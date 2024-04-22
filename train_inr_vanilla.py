import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, TotalVariation
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
from pathlib import Path
from tqdm import tqdm
import os
from os import makedirs

from datasets import SliceDataset, ImageFitting, protocol_filter
from transforms import INRDataTransform, image_to_kspace

from fastmri.data.subsample import create_mask_for_mask_type
from degradation import get_shift
from inr.inr_model import SelfSiren

from loss import laplacian_edge
from datetime import datetime


def seed_everything(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed',
    )

    parser.add_argument(
        '--region',
        type=str,
        choices=["knee"],
        default='knee',
        help='The part of the body.',
    )

    parser.add_argument(
        '--af',
        type=int,
        default=4,
        help='Acceleration factor.',
    )

    parser.add_argument(
        '--acs',
        type=int,
        default=0.08,
        help='ACS region, default to 0.08.',
    )

    parser.add_argument(
        '--slice_num',
        type=int,
        default=50,
        help='The number of slice used to train.',
    )

    parser.add_argument(
        '--data_root',
        type=str,
        default="/path/to/data/",
        help='Path to data.',
    )

    parser.add_argument(
        '--dataset_cache_path',
        type=str,
        default="dataset_cache.pkl",
        help='Path to dataset cache.',
    )

    parser.add_argument(
        '--dim_pe',
        type=int,
        default=4,
        help='The dimension of positional encoding.',
    )

    parser.add_argument(
        '--num_iter',
        type=int,
        default=4000,
        help='Number of iterations.',
    )

    parser.add_argument(
        '--sum_interval',
        type=int,
        default=1,
        help='The interval between summary.',
    )

    parser.add_argument(
        '--loss_weights',
        type=list,
        default=[1, 1e-4, 1, 1e-3, 1e-5],
        help='The weights of losses.',
    )

    parser.add_argument(
        '--summary_path',
        type=str,
        default="./log/",
        help='Path to save summary.',
    )

    args = parser.parse_args()
    return args


def train_self_siren(args, net, optimizer, dataloader, num, lr_scheduler, image_size, total_steps=4000, steps_til_summary=1):
    coords, pos_encoding, d_pixels, pixels, d_image, gt_image, mask = next(iter(dataloader))
    coords, pos_encoding, d_pixels, pixels, d_image, gt_image, mask = (coords.cuda(), pos_encoding.cuda(),
                                                                       d_pixels.cuda(), pixels.cuda(), d_image.cuda(),
                                                                       gt_image.cuda(), mask.cuda())
    # Initialize best value
    best_ssim = 0
    best_kr_ssim = 0
    # Initialize image shape
    B, C, H, W = 1, 1, image_size, image_size
    # Initialize metrics
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()
    tv = TotalVariation().cuda()
    # Prepare summary and save path
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    summary_save_path = str(Path(args.summary_path) / run_id)
    makedirs(summary_save_path, exist_ok=True)
    writer = SummaryWriter(summary_save_path)
    writer.add_image('GT', gt_image.squeeze()[None])
    writer.add_image('Input', d_image.squeeze()[None])
    # Begin to train
    net.cuda()
    net.train()
    with (tqdm(total=total_steps) as pbar):
        pbar.set_description('Training')
        for step in range(total_steps):
            model_input = torch.cat((pos_encoding, d_pixels), dim=-1)
            output_pixels, conv_img = net(model_input)
            filtered_img_np = d_image.squeeze().cpu().detach().numpy()
            conv_img_np = conv_img.squeeze().cpu().detach().numpy()
            shift = get_shift(filtered_img_np, conv_img_np)
            aligned_conv_img = torch.roll(conv_img.squeeze(), shift, dims=(0, 1))
            aligned_conv_img = torch.clamp(aligned_conv_img, 0.0, 1.0)
            aligned_conv_pixels = aligned_conv_img[None].permute(1, 2, 0).view(-1, 1)
            output_img = output_pixels.view(C, H, W)
            output_img = torch.clamp(output_img, 0.0, 1.0)
            output_kspace = image_to_kspace(output_img)
            input_kspace = image_to_kspace(d_image)
            output_kr_kspace = output_kspace * (1 - mask) + input_kspace
            output_kr_img = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(output_kr_kspace)))
            output_kr_img = torch.clamp(output_kr_img, 0.0, 1.0)
            kr_ssim = ssim(output_kr_img.view(B, C, H, W), gt_image.view(B, C, H, W))
            kr_psnr = psnr(output_kr_img.view(B, C, H, W), gt_image.view(B, C, H, W))
            # Calculate losses
            loss_f = (torch.abs(output_kspace * mask - input_kspace) ** 2).mean()
            loss_i = ((d_pixels - aligned_conv_pixels) ** 2).mean()
            loss_perc = 1 - ssim(aligned_conv_img[None, None], d_image)
            loss_blur = laplacian_edge(output_img[None])
            loss_tv = tv(output_img[None])
            loss = (args.loss_weights[0] * loss_i) + (args.loss_weights[1] * loss_f) + (args.loss_weights[2] *
                    loss_perc) + (args.loss_weights[3] * loss_blur) + (args.loss_weights[4] * loss_tv)
            # Write summary
            writer.add_scalar("Kspace Diff", loss_f, global_step=step)
            writer.add_scalar("Kspace Replaced SSIM", kr_ssim, global_step=step)
            writer.add_scalar("Train_loss", loss, global_step=step)
            if not (step+1) % steps_til_summary:
                ssim_value = ssim(output_pixels.view(B, C, H, W), gt_image.view(B, C, H, W))
                psnr_value = psnr(output_pixels.view(B, C, H, W), gt_image.view(B, C, H, W))
                pbar.set_description(f"Best SSIM: {best_ssim:.5f}, Best KR SSIM: {best_kr_ssim:.5f}")
                writer.add_scalar("SSIM", ssim_value, global_step=step)
                writer.add_scalar("PSNR", psnr_value, global_step=step)
                if ssim_value > best_ssim or kr_ssim > best_kr_ssim:
                    if ssim_value > best_ssim:
                        checkpoint = {
                            'epoch': step,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'ssim': ssim_value,
                            'psnr': psnr_value,
                        }
                        torch.save(checkpoint, f"ckpt/best_output_ssim_AF{args.af}_{num}.pth")
                        best_ssim = ssim_value
                        # print("Updated best ssim, Best SSIM: {}".format(best_ssim))
                    if kr_ssim > best_kr_ssim:
                        checkpoint = {
                            'epoch': step,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'ssim': kr_ssim,
                            'psnr': kr_psnr,
                        }
                        torch.save(checkpoint, f"ckpt/best_kr_ssim_AF{args.af}_{num}.pth")
                        best_kr_ssim = kr_ssim
                        # print("Updated best kr ssim, Best KR SSIM: {}".format(kr_ssim))
                    writer.add_image('output', output_img, global_step=step)
                    writer.add_image('output_kr', output_kr_img.squeeze()[None], global_step=step)
                    writer.add_image('aligned_conv_img', aligned_conv_img[None], global_step=step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)
            lr_scheduler.step()


def main(args):
    protocols = {'knee': ['CORPDFS_FBK', 'CORPD_FBK']}
    recon_sizes = {'knee': 320}
    p_filter = protocol_filter(protocols[args.region][1:])  # PD Only
    seed_everything(args.seed)
    mask_func = create_mask_for_mask_type("equispaced_fraction", [args.acs], [args.af])
    root = Path(args.data_root)
    data_transform = INRDataTransform(mask_func=mask_func)
    dataset = SliceDataset(root, data_transform, p_filter, args.dataset_cache_path)
    num = args.slice_num
    # protocol = "pd"
    sample = dataset.__getitem__(num)  # 50
    dataset_for_siren = ImageFitting(recon_sizes[args.region], sample, pos_dim=args.dim_pe)
    dataloader = DataLoader(dataset_for_siren, batch_size=1, pin_memory=True, num_workers=0)
    print(f"[INFO] Input SSIM: {StructuralSimilarityIndexMeasure(data_range=1.0)(sample.image[None, None], sample.target[None, None])}")
    in_dim = 2 + 1 + args.dim_pe * 2 * 2
    img_siren = SelfSiren(in_features=in_dim, out_features=1, kernel=sample.kernel, hidden_layers=5, hidden_features=256)
    optim = torch.optim.Adam(lr=1e-4, params=filter(lambda p: p.requires_grad, img_siren.parameters()))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=300, gamma=0.5)
    train_self_siren(args, img_siren, optim, dataloader, num=num, image_size=recon_sizes[args.region], lr_scheduler=scheduler, total_steps=args.num_iter, steps_til_summary=args.sum_interval)


if __name__ == '__main__':
    args = get_args()
    main(args)
