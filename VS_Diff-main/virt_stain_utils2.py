import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from PIL import Image
import glob
import torch.nn as nn
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
import random
from skimage.metrics import structural_similarity as ssim # converts to grayscale
# from torchmetrics import StructuralSimilarityIndexMeasure
import shutil
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import MS_SSIM
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import LambdaLR
import torchvision
import glob, csv
from torch import no_grad
from torchmetrics.image import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def save_checkpoint(state, is_best, checkpoint_dir='checkpoints'):
    # state should include 'epoch', 'step', 'model_state_dict', 'optimizer_state_dict', etc.
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save(state, path)
    if is_best:
        shutil.copy(path, os.path.join(checkpoint_dir, 'best.pth'))

def load_checkpoint(model, optimizer, scaler, checkpoint_path='checkpoints/checkpoint.pth'):
    if os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt.get('scaler_state_dict', scaler.state_dict()))
        start_epoch = ckpt.get('epoch', 0)
        start_step  = ckpt.get('step', 0)
        best_val_ssim = ckpt.get('best_val_ssim', 0.0)
        print(f"→ Resumed from epoch {start_epoch}, step {start_step}")
        return start_epoch, start_step, best_val_ssim
    else:
        print("→ No checkpoint found, starting at epoch 0, step 0")
        return 0, 0, 0.0
    

def make_warmup_cosine_scheduler(optimizer, total_steps, warmup_frac=0.05):
    """
    LR multiplier schedule: linear warmup from 0→1 over warmup_frac*total_steps,
    then cosine decay from 1→0 over the rest.
    """
    warmup_steps = int(warmup_frac * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / warmup_steps  # ramp up
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay

    return LambdaLR(optimizer, lr_lambda)




def train_one_epoch(
    model,
    loader,      # yields (phase, stained_gt)
    optimizer,
    scaler,
    noise_scheduler,
    alpha,       # weight for noise MSE
    beta,        # weight for Lab chroma loss
    device,
    lr_scheduler,
    global_step,
    sigma
):
    model.train()

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    running = {
        'train_mse_r': 0., 'train_mse_g': 0., 'train_mse_b': 0., 'train_mse': 0.,
        'train_ssim_r': 0., 'train_ssim_g': 0., 'train_ssim_b': 0., 'train_ssim': 0.,
        'train_loss': 0.,
        'train_lab_chroma': 0.,
    }

    for i, (phase, stained_gt) in enumerate(loader, 1):
        phase, stained_gt = phase.to(device), stained_gt.to(device)
        noise = torch.randn_like(stained_gt)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (stained_gt.size(0),),
            device=device
        ).long()
        x_noisy = noise_scheduler.add_noise(stained_gt, noise, timesteps)

        optimizer.zero_grad()
        with autocast():
            inp = torch.cat([x_noisy, phase], dim=1)
            out = model(inp, timesteps)
            noise_pred = out.sample if hasattr(out, "sample") else out

            # core diffusion loss: predicted noise MSE
            loss_mse = F.mse_loss(noise_pred, noise)

            # one-step denoised image
            denoised_list = []
            for b in range(noise_pred.shape[0]):
                t_b = int(timesteps[b].item())
                step_output = noise_scheduler.step(noise_pred[b:b+1], t_b, x_noisy[b:b+1])
                prev = step_output.prev_sample if hasattr(step_output, "prev_sample") else step_output["prev_sample"]
                denoised_list.append(prev)
            x_den_raw = torch.clamp(torch.cat(denoised_list, dim=0), 0.0, 1.0)  # in normalized space, clamped

            # prepare for Lab chroma loss (helpers expected to be in this module)
            one_step_vis = prepare_vis(x_den_raw)    # [0,1]
            gt_vis = prepare_vis(stained_gt)         # [0,1]
            w = soft_tissue_weight(gt_vis, sigma)
            loss_lab = lab_chroma_loss(one_step_vis, gt_vis, weight=w)

            loss = alpha * loss_mse + beta * loss_lab

        # stability guard
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[WARN] skipping bad batch at global_step {global_step}; loss={loss}")
            global_step += 1
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # metrics on denoised image: use same preparation
        deno = prepare_vis(x_den_raw)
        gt = prepare_vis(stained_gt)

        mse_r = F.mse_loss(deno[:,0:1], gt[:,0:1]).item()
        mse_g = F.mse_loss(deno[:,1:2], gt[:,1:2]).item()
        mse_b = F.mse_loss(deno[:,2:3], gt[:,2:3]).item()
        mse_avg = (mse_r + mse_g + mse_b) / 3.0

        sr = ssim_metric(deno[:,0:1], gt[:,0:1]).item()
        sg = ssim_metric(deno[:,1:2], gt[:,1:2]).item()
        sb = ssim_metric(deno[:,2:3], gt[:,2:3]).item()
        ssim_avg = (sr + sg + sb) / 3.0

        # accumulate
        running['train_mse_r'] += mse_r
        running['train_mse_g'] += mse_g
        running['train_mse_b'] += mse_b
        running['train_mse'] += mse_avg
        running['train_ssim_r'] += sr
        running['train_ssim_g'] += sg
        running['train_ssim_b'] += sb
        running['train_ssim'] += ssim_avg
        running['train_loss'] += loss.item()
        running['train_lab_chroma'] += loss_lab.item()

        global_step += 1

    # average
    for k in running:
        running[k] /= i

    if lr_scheduler is not None:
        running['lr'] = lr_scheduler.get_last_lr()[0]

    return running, global_step

    

def validate_metrics(model, loader, noise_scheduler, device):
    """
    Validation loop: computes per-channel and overall MSE and SSIM on validation loader.
    Returns a dict with keys:
      val_mse_r, val_mse_g, val_mse_b, val_mse,
      val_ssim_r, val_ssim_g, val_ssim_b, val_ssim
    """
    model.eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    running = {
        'val_mse_r': 0.0, 'val_mse_g': 0.0, 'val_mse_b': 0.0, 'val_mse': 0.0,
        'val_ssim_r': 0.0, 'val_ssim_g': 0.0, 'val_ssim_b': 0.0, 'val_ssim': 0.0
    }
    count = 0
    with torch.no_grad():
        for phase, stained in loader:
            phase, stained = phase.to(device), stained.to(device)

            # add noise + sample timesteps
            noise     = torch.randn_like(stained)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (stained.size(0),),
                device=device
            ).long()
            x_noisy = noise_scheduler.add_noise(stained, noise, timesteps)

            # predict
            inp  = torch.cat([x_noisy, phase], dim=1)
            pred = model(inp, timesteps).sample

            # per-sample one-step denoise
            denoised_list = []
            for b in range(pred.shape[0]):
                t_b   = int(timesteps[b].item())
                out_b = noise_scheduler.step(pred[b:b+1], t_b, x_noisy[b:b+1])
                denoised_list.append(out_b.prev_sample)
            x_den = torch.clamp(torch.cat(denoised_list, dim=0), 0.0, 1.0)

            # de-normalize for metrics
            deno = x_den * 0.5 + 0.5
            gt   = stained * 0.5 + 0.5

            # per-channel MSE
            mse_r   = F.mse_loss(deno[:,0:1], gt[:,0:1]).item()
            mse_g   = F.mse_loss(deno[:,1:2], gt[:,1:2]).item()
            mse_b   = F.mse_loss(deno[:,2:3], gt[:,2:3]).item()
            mse_all = F.mse_loss(deno,        gt).item()

            # per-channel SSIM
            sr       = ssim_metric(deno[:,0:1], gt[:,0:1]).item()
            sg       = ssim_metric(deno[:,1:2], gt[:,1:2]).item()
            sb       = ssim_metric(deno[:,2:3], gt[:,2:3]).item()
            ssim_all = ssim_metric(deno,        gt).item()

            # accumulate
            running['val_mse_r'] += mse_r
            running['val_mse_g'] += mse_g
            running['val_mse_b'] += mse_b
            running['val_mse']   += mse_all
            running['val_ssim_r'] += sr
            running['val_ssim_g'] += sg
            running['val_ssim_b'] += sb
            running['val_ssim']   += ssim_all
            count += 1

    # average
    for k in running:
        running[k] /= count
    return running

def inference_batch(
    model,
    phase,          # Tensor [B,1,H,W]
    stained,        # Tensor [B,3,H,W]
    noise_scheduler,
    device
):
    """
    Pure inference: returns denoised output x_den in [0,1].
    No plotting or logging. More for simple visual checks. 
    """
    model.eval()
    phase, stained = phase.to(device), stained.to(device)
    with torch.no_grad():
        noise   = torch.randn_like(stained)
        ts      = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (stained.size(0),), device=device
        ).long()
        x_noisy = noise_scheduler.add_noise(stained, noise, ts)
        inp     = torch.cat([x_noisy, phase], dim=1)
        out     = model(inp, ts).sample
        out     = noise_scheduler.step(out, ts, x_noisy)
        x_den   = torch.clamp(out.prev_sample, 0, 1)

    return x_den


def visualize_batch(
    phase,          # Tensor [B,1,H,W]
    stained,        # Tensor [B,3,H,W]
    x_den,          # Tensor [B,3,H,W], output of inference_batch
    denorm_fn,      # function mapping [-1,1]→[0,1]
    visualize=True, # if True, show inline Matplotlib
    writer=None,    # SummaryWriter, if logging to TensorBoard
    global_step=None
):
    """
    Displays (and/or logs) the phase | ground-truth | denoised images.
    - denorm_fn: e.g. lambda x: (x*0.5 + 0.5).clamp(0,1)
    - If writer & global_step provided, logs image grids.
    """
    # Prepare CPU tensors in [0,1]
    ph_vis = denorm_fn(phase).squeeze(1).cpu()    # [B,H,W]
    gt_vis = denorm_fn(stained).permute(0,2,3,1).cpu()  # [B,H,W,3]
    dn_vis = denorm_fn(x_den).permute(0,2,3,1).cpu()     # [B,H,W,3]

    if visualize:
        import matplotlib.pyplot as plt
        B = min(ph_vis.size(0), 4)
        fig, axs = plt.subplots(B, 3, figsize=(9, 3*B))
        for i in range(B):
            axs[i,0].imshow(ph_vis[i], cmap='gray');    axs[i,0].axis('off'); axs[i,0].set_title('Phase')
            axs[i,1].imshow(gt_vis[i]);                axs[i,1].axis('off'); axs[i,1].set_title('GT')
            axs[i,2].imshow(dn_vis[i]);                axs[i,2].axis('off'); axs[i,2].set_title('Denoised')
        plt.tight_layout()
        plt.show()

    if writer is not None and global_step is not None:
        # log as image grids: expects [B,C,H,W]
        writer.add_images('infer/phase',
                          ph_vis.unsqueeze(1), global_step)
        writer.add_images('infer/gt',
                          torch.from_numpy(gt_vis).permute(0,3,1,2), global_step)
        writer.add_images('infer/denoised',
                          torch.from_numpy(dn_vis).permute(0,3,1,2), global_step)
        

def _denorm_tensor(x):
    """Assumes input is in [-1,1], returns in [0,1]."""
    return x * 0.5 + 0.5

def compute_ssim_per_sample(a_tensor, b_tensor):
    """
    Computes SSIM between two tensors of shape [B,C,H,W], expects both in [0,1].
    Returns the average SSIM over the batch.
    """
    ssim_module = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").to(a_tensor.device)
    with torch.no_grad():
        ssim_vals = ssim_module(a_tensor, b_tensor)  # shape: (B,)
        return float(ssim_vals.mean().item())

def compute_mse_per_sample(a_tensor, b_tensor):
    """
    MSE between two tensors [B,C,H,W], returns average over batch.
    Assumes inputs are in [0,1].
    """
    return float(F.mse_loss(a_tensor, b_tensor).item())

@torch.no_grad()
def ddim_sample_full(model, phase, scheduler, device, num_inference_steps=50, eta=0.0):
    """
    Full conditional DDIM sampling given `phase`.
    Returns sampled stained image in same range as model output (e.g., [-1,1]).
    """
    model.eval()
    B = phase.size(0)
    sample = torch.randn(B, 3, phase.shape[2], phase.shape[3], device=device)

    scheduler.set_timesteps(num_inference_steps, device=device)

    for t in scheduler.timesteps:
        inp = torch.cat([sample, phase], dim=1)
        out = model(inp, t)
        noise_pred = out.sample if hasattr(out, "sample") else out
        # try passing eta if supported
        try:
            step_output = scheduler.step(noise_pred, t, sample, eta=eta)
        except TypeError:
            step_output = scheduler.step(noise_pred, t, sample)
        if hasattr(step_output, "prev_sample"):
            sample = step_output.prev_sample
        else:
            sample = step_output["prev_sample"]
    model.train()
    return sample

import matplotlib.pyplot as plt  

def _to_gray(x):
    # simple average across RGB channels, expects [B,3,H,W]
    return x.mean(dim=1, keepdim=True)

@torch.no_grad()
def run_sanity_check(
    model,
    val_loader,
    train_noise_scheduler,    # e.g., DDPMScheduler used during training
    inference_scheduler,      # e.g., DDIMScheduler for full sampling
    writer,
    epoch,
    device,
    num_visualize=4,
    num_inference_steps=50,
    eta=0.0,
    log_dir=None,
    verbose=False,            # toggle detailed diagnostics & inline plots
):
    """
    Executes sanity check: one-step denoise vs full DDIM sample.
    Logs image grid to TensorBoard, returns SSIM/MSE metrics.
    If verbose=True, also prints ranges and shows per-component visualizations.
    """
    model.eval()
    with torch.no_grad():
        phase, stained = next(iter(val_loader))
        phase = phase.to(device)[:num_visualize]
        stained = stained.to(device)[:num_visualize]

        B = phase.size(0)

        # --- One-step denoise (training-style) ---
        # pick a single representative scalar timestep (middle)
        t_idx = len(train_noise_scheduler.timesteps) // 2
        t_scalar = train_noise_scheduler.timesteps[t_idx]
        timesteps_vec = torch.full((B,), int(t_scalar.item()), dtype=torch.long, device=device)

        noise = torch.randn_like(stained)
        x_noisy = train_noise_scheduler.add_noise(stained, noise, timesteps_vec)

        inp = torch.cat([x_noisy, phase], dim=1)
        out = model(inp, timesteps_vec)
        noise_pred = out.sample if hasattr(out, "sample") else out

        # step requires scalar timestep for this scheduler
        step_output = train_noise_scheduler.step(noise_pred, int(t_scalar.item()), x_noisy)
        if hasattr(step_output, "prev_sample"):
            x_den = step_output.prev_sample
        else:
            x_den = step_output["prev_sample"]

        # --- Full inference sampling ---
        full_sample = ddim_sample_full(
            model=model,
            phase=phase,
            scheduler=inference_scheduler,
            device=device,
            num_inference_steps=num_inference_steps,
            eta=eta
        )

        # --- Clamp to expected range before denorm ---
        stained_clamped = torch.clamp(stained, -1.0, 1.0)
        x_den_clamped = torch.clamp(x_den, -1.0, 1.0)
        full_clamped = torch.clamp(full_sample, -1.0, 1.0)
        phase_clamped = torch.clamp(phase, -1.0, 1.0)

        # Denormalize to [0,1]
        phase_vis = _denorm_tensor(phase_clamped.repeat(1, 3, 1, 1))  # 3-channel for display
        gt_vis = _denorm_tensor(stained_clamped)
        one_step_vis = _denorm_tensor(x_den_clamped)
        full_vis = _denorm_tensor(full_clamped)

        # --- Compute metrics ---
        ssim_one_step = compute_ssim_per_sample(gt_vis, one_step_vis)
        ssim_full = compute_ssim_per_sample(gt_vis, full_vis)
        mse_one_step = compute_mse_per_sample(gt_vis, one_step_vis)
        mse_full = compute_mse_per_sample(gt_vis, full_vis)

        # --- Build grid and log ---
        combined = torch.cat([phase_vis, gt_vis, one_step_vis, full_vis], dim=0)  # 4*B
        grid = torchvision.utils.make_grid(combined, nrow=B, padding=4, pad_value=1.0)
        writer.add_image("sanity/phase_gt_onestep_full", grid, epoch)

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            from torchvision.utils import save_image
            save_image(grid, os.path.join(log_dir, f"sanity_epoch{epoch+1:03d}.png"))

        # --- VERBOSE DEBUGGING ---
        if verbose:
            def print_stats(name, tensor):
                t = tensor.detach()
                print(f"{name}: min {t.min().item():.4f}, max {t.max().item():.4f}, "
                      f"mean {t.mean().item():.4f}, std {t.std().item():.4f}")

            # Raw (pre-clamp) stats
            print("\n[Sanity DEBUG] Raw ranges before clamp/denorm:")
            print_stats("GT (raw)", stained)
            print_stats("One-step (raw)", x_den)
            print_stats("Full sample (raw)", full_sample)
            print_stats("Phase (raw)", phase)

            # Post-clamp + denorm stats
            print("\n[Sanity DEBUG] After clamp + denorm (should be in [0,1]):")
            print_stats("GT vis", gt_vis)
            print_stats("One-step vis", one_step_vis)
            print_stats("Full vis", full_vis)
            print_stats("Phase vis (replicated)", phase_vis)

            # Inline visualizations: phase / GT / one-step / full
            def show_tensor(tensor, title):
                img = tensor[0].permute(1, 2, 0).cpu().clamp(0, 1).numpy()
                plt.figure(figsize=(3, 3))
                plt.imshow(img)
                plt.title(title)
                plt.axis("off")
                plt.tight_layout()
                plt.show()

            print("\n[Sanity DEBUG] Visual components:")
            show_tensor(phase_vis, "Phase input")
            show_tensor(gt_vis, "GT H&E")
            show_tensor(one_step_vis, "One-step denoise")
            show_tensor(full_vis, "Full DDIM sample")

            # Channel-wise difference between full sample and GT
            print("\n[Sanity DEBUG] Channel-wise differences (Full - GT):")
            diff_full = full_vis - gt_vis  # in [0,1]
            for i, cname in enumerate(["R", "G", "B"]):
                plt.figure(figsize=(2.5, 2.5))
                plt.imshow(diff_full[0, i].cpu(), cmap="bwr", vmin=-0.2, vmax=0.2)
                plt.title(f"Full - GT diff ({cname})")
                plt.colorbar()
                plt.axis("off")
                plt.tight_layout()
                plt.show()

            # Grayscale structural check
            print("\n[Sanity DEBUG] Grayscale structure comparison:")
            show_tensor(_to_gray(gt_vis).repeat(1, 3, 1, 1), "GT gray")
            show_tensor(_to_gray(full_vis).repeat(1, 3, 1, 1), "Full gray")

    model.train()

    metrics = {
        "ssim_one_step": ssim_one_step,
        "ssim_full": ssim_full,
        "mse_one_step": mse_one_step,
        "mse_full": mse_full,
    }
    return metrics

@torch.no_grad()
def validate_diffusion_math(noise_scheduler, phase, stained, device):
    B = phase.size(0) 
    # pick a middle timestep
    t_idx = len(noise_scheduler.timesteps) // 2
    t_scalar = noise_scheduler.timesteps[t_idx]
    timesteps_vec = torch.full((B,), int(t_scalar.item()), dtype=torch.long, device=device)

    # sample noise and add it
    noise = torch.randn_like(stained)
    x_noisy = noise_scheduler.add_noise(stained, noise, timesteps_vec)

    # perfect inversion: feed true noise into step
    step_output = noise_scheduler.step(noise, int(t_scalar.item()), x_noisy)
    reconstructed = step_output.prev_sample if hasattr(step_output, "prev_sample") else step_output["prev_sample"]

    # clamp + denorm for comparison
    def prepare(x):
        return torch.clamp((torch.clamp(x, -1.0, 1.0) * 0.5 + 0.5), 0.0, 1.0)

    recon_vis = prepare(reconstructed)
    gt_vis = prepare(stained)

    # compute SSIM / MSE
    from torchmetrics import StructuralSimilarityIndexMeasure
    ssim_module = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_val = ssim_module(gt_vis, recon_vis).item()
    mse_val = F.mse_loss(gt_vis, recon_vis).item()
    print(f"[Diffusion math check] Perfect reconstruction SSIM: {ssim_val:.4f}, MSE: {mse_val:.6f}")
    return ssim_val, mse_val

@torch.no_grad()
def check_one_step_t1(noise_scheduler, stained, device):
    """
    Perfect reconstruction test at t=1: add noise at timestep 1 and reverse it
    using the true noise. Should recover stained (x0) with high SSIM.
    """
    B = stained.size(0)

    # find timestep closest to 1 in the scheduler's timesteps array
    target_val = 1
    if (noise_scheduler.timesteps == target_val).any():
        t_scalar = target_val
    else:
        # fallback: smallest positive timestep
        positive = noise_scheduler.timesteps[noise_scheduler.timesteps > 0]
        t_scalar = int(positive.min().item())

    timesteps_vec = torch.full((B,), int(t_scalar), dtype=torch.long, device=device)

    # sample true noise and form x_t
    noise = torch.randn_like(stained)
    x_t = noise_scheduler.add_noise(stained, noise, timesteps_vec)

    # reverse using the true noise at that timestep
    step_output = noise_scheduler.step(noise, int(t_scalar), x_t)
    if hasattr(step_output, "prev_sample"):
        reconstructed = step_output.prev_sample
    else:
        reconstructed = step_output["prev_sample"]

    # prepare for comparison: clamp then denorm (assumes normalized in [-1,1])
    def prepare(x):
        x = torch.clamp(x, -1.0, 1.0)
        x = x * 0.5 + 0.5  # denorm to [0,1]
        return torch.clamp(x, 0.0, 1.0)

    recon_vis = prepare(reconstructed)
    gt_vis = prepare(stained)

    ssim_module = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_val = ssim_module(gt_vis, recon_vis).item()
    mse_val = F.mse_loss(gt_vis, recon_vis).item()

    print(f"[Diffusion math t=1 check] SSIM: {ssim_val:.4f}, MSE: {mse_val:.6f}")
    return ssim_val, mse_val

def beta_ramp(epoch, ramp_epochs=20, target_beta=0.5):
    return target_beta * min(1.0, (epoch + 1) / ramp_epochs)


#-------- CIELAB Chroma Loss Helper Functions --------------------------------------------------   

def _f(t):
    delta = 6 / 29
    return torch.where(t > delta ** 3, t.pow(1/3), (t / (3 * delta ** 2)) + 4/29)

def rgb_to_lab(img):
    """
    img: [B,3,H,W] in [0,1]
    returns Lab tensor [B,3,H,W]; L in ~[0,100], a/b centered around 0
    """
    # sRGB to linear
    mask = img > 0.04045
    linear = torch.where(mask, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)

    # linear RGB to XYZ (D65)
    rgb_to_xyz = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], device=img.device, dtype=img.dtype)  # 3x3

    B, C, H, W = img.shape
    linear_flat = linear.permute(0,2,3,1).reshape(-1,3)  # (N,3)
    xyz = (linear_flat @ rgb_to_xyz.T)  # (N,3)

    # normalize by reference white D65
    white = torch.tensor([0.95047, 1.0, 1.08883], device=img.device, dtype=img.dtype)
    xyz_scaled = xyz / white

    f_x = _f(xyz_scaled[:,0])
    f_y = _f(xyz_scaled[:,1])
    f_z = _f(xyz_scaled[:,2])

    L = (116 * f_y) - 16
    a = 500 * (f_x - f_y)
    b = 200 * (f_y - f_z)

    lab = torch.stack([L, a, b], dim=1)  # (N,3)
    lab = lab.reshape(B, H, W, 3).permute(0,3,1,2)  # [B,3,H,W]
    return lab

def prepare_vis(x):
    """
    x: model output or GT in normalized [-1,1]
    returns: clamped + denormalized in [0,1]
    """
    x = torch.clamp(x, -1.0, 1.0)
    x = x * 0.5 + 0.5
    return torch.clamp(x, 0.0, 1.0)


def lab_chroma_loss(pred_rgb, gt_rgb, weight=None):
    pred_lab = rgb_to_lab(pred_rgb)
    gt_lab   = rgb_to_lab(gt_rgb)
    ab_diff  = pred_lab[:,1:] - gt_lab[:,1:]           # a,b
    if weight is None:
        return torch.mean(ab_diff**2)
    w2 = weight.expand(-1, 2, -1, -1)
    return (w2 * (ab_diff**2)).sum() / w2.sum().clamp_min(1.0)

#-------------------MASKING HELPER FUNCTIONS -------------------------------------

def soft_tissue_weight(gt_rgb, sigma):
    """
    gt_rgb: [B,3,H,W] in [0,1]
    returns w in [0,1]; ~0 near white, →1 as chroma/distance from white increases
    """
    gt_lab = rgb_to_lab(gt_rgb)                 # [B,3,H,W]
    L, a, b = gt_lab[:,0:1], gt_lab[:,1:2], gt_lab[:,2:3]
    deltaE = torch.sqrt((100.0 - L)**2 + a**2 + b**2)  # distance to white (100,0,0)
    w = 1.0 - torch.exp(-(deltaE**2) / (2 * (sigma**2)))  # soft 0→1
    return w


#-------------INFERENCE TEST STAGE ------------------------------------------

@torch.no_grad()
def infer_and_save_ddim(
    model,
    loader,                       # e.g. test_loader
    scheduler,                    # your DDIM scheduler (inference_noise_scheduler)
    device,
    out_dir,
    steps,
    eta,
    use_dataset_names=True,       # use dataset filenames if available
    save_triptych=False           # also save [phase | GT | pred] panels
):
    """
    Runs DDIM sampling for every batch in `loader` and saves images.
    Expects inputs normalized to [-1,1]; saves PNGs in [0,1].

    If the dataset has `.image_paths`, filenames are based on those;
    otherwise sequential names are used.
    """
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    has_paths = hasattr(loader.dataset, "image_paths")
    counter = 0

    for batch_idx, (phase, stained_gt) in enumerate(loader):
        phase = phase.to(device)

        # full DDIM sample (your existing utility)
        pred = ddim_sample_full(
            model=model,
            phase=phase,
            scheduler=scheduler,
            device=device,
            num_inference_steps=steps,
            eta=eta,
        )  # [-1,1], shape [B,3,H,W]

        # clamp & denorm to [0,1] for saving
        pred_vis = torch.clamp(pred, -1.0, 1.0) * 0.5 + 0.5

        # optional visuals
        if save_triptych:
            gt_vis    = torch.clamp(stained_gt.to(device), -1.0, 1.0) * 0.5 + 0.5
            phase_vis = torch.clamp(phase, -1.0, 1.0).repeat(1,3,1,1) * 0.5 + 0.5

        B = pred_vis.size(0)
        for b in range(B):
            if use_dataset_names and has_paths:
                base = os.path.splitext(os.path.basename(loader.dataset.image_paths[counter]))[0]
                name = base
            else:
                name = f"sample_{counter:06d}"

            # save prediction
            torchvision.utils.save_image(pred_vis[b], os.path.join(out_dir, f"{name}_pred.tif"))

            if save_triptych:
                panel = torch.stack([phase_vis[b], gt_vis[b], pred_vis[b]], dim=0)  # [3,3,H,W]
                grid  = torchvision.utils.make_grid(panel, nrow=3, padding=2)
                torchvision.utils.save_image(grid, os.path.join(out_dir, f"{name}_panel.tif"))

            counter += 1


#------------------COMPARISONS -------------------------------
def _norm_name(p: str) -> str:
    s = os.path.splitext(os.path.basename(p))[0].lower()
    for suf in ("_pred","_out","_fake","_ddim","_gan"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s

def deltaE76(pred01: torch.Tensor, gt01: torch.Tensor) -> torch.Tensor:
    """Return ΔE76 map per pixel. pred01, gt01: [3,H,W] in [0,1]."""
    pred_lab = rgb_to_lab(pred01.unsqueeze(0))[0]  # [3,H,W]
    gt_lab   = rgb_to_lab(gt01.unsqueeze(0))[0]
    d = torch.sqrt((pred_lab[0]-gt_lab[0])**2 + (pred_lab[1]-gt_lab[1])**2 + (pred_lab[2]-gt_lab[2])**2)
    return d  # [H,W]

def psnr_from_mse(mse: float, eps: float = 1e-12) -> float:
    return 10.0 * math.log10(1.0 / max(mse, eps))

def compute_metrics_batch(pred01: torch.Tensor, gt01: torch.Tensor, mask: torch.Tensor = None,
                          ssim_metric: StructuralSimilarityIndexMeasure = None):
    """
    pred01, gt01: [B,3,H,W] in [0,1]
    mask: [B,1,H,W] in [0,1] (optional, tissue weights)
    returns dict of per-image metrics (lists) and batch means
    """
    B = pred01.size(0)
    if ssim_metric is None:
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred01.device)

    # global MSE per image
    diff2 = (pred01 - gt01)**2
    mse_img = diff2.reshape(B, -1).mean(dim=1).cpu().tolist()

    # tissue-weighted MSE per image
    if mask is not None:
        w = mask.expand_as(pred01)
        num = (w * diff2).flatten(1).sum(dim=1)
        den = w.flatten(1).sum(dim=1).clamp_min(1.0)
        mse_tissue = (num / den).cpu().tolist()
    else:
        mse_tissue = [float('nan')] * B

    # SSIM (RGB)
    ssim_img = []
    for i in range(B):
        ssim_img.append(ssim_metric(pred01[i].unsqueeze(0), gt01[i].unsqueeze(0)).item())

    # PSNR from global MSE
    psnr_img = [psnr_from_mse(m) for m in mse_img]

    # ΔE76 mean per image
    dE_mean = []
    for i in range(B):
        dE = deltaE76(pred01[i], gt01[i])
        dE_mean.append(dE.mean().item())

    return {
        "mse": mse_img,
        "mse_tissue": mse_tissue,
        "ssim": ssim_img,
        "psnr": psnr_img,
        "deltaE": dE_mean,
        "mse_mean": float(np.mean(mse_img)),
        "mse_tissue_mean": float(np.nanmean(mse_tissue)),
        "ssim_mean": float(np.mean(ssim_img)),
        "psnr_mean": float(np.mean(psnr_img)),
        "deltaE_mean": float(np.mean(dE_mean)),
    }

def _save_panel(phase01, gt01, pred01, title, save_path, other01=None):
    """
    phase01: [1,H,W] or [3,H,W] in [0,1]
    gt01, pred01, other01: [3,H,W] in [0,1]
    Saves a panel PNG with ΔE maps.
    """
    H, W = gt01.shape[-2:]
    if phase01.shape[0] == 1:
        phase3 = phase01.repeat(3,1,1)
    else:
        phase3 = phase01
    dE_pred = deltaE76(pred01, gt01)
    fig_cols = 5 if other01 is None else 7

    plt.figure(figsize=(3*fig_cols, 3))
    # Phase, GT, Pred, ΔE(pred)
    plt.subplot(1, fig_cols, 1); plt.imshow(phase3.permute(1,2,0).cpu()); plt.title("Phase"); plt.axis("off")
    plt.subplot(1, fig_cols, 2); plt.imshow(gt01.permute(1,2,0).cpu());    plt.title("GT");    plt.axis("off")
    plt.subplot(1, fig_cols, 3); plt.imshow(pred01.permute(1,2,0).cpu());  plt.title("Pred");  plt.axis("off")
    plt.subplot(1, fig_cols, 4); im=plt.imshow(dE_pred.cpu(), cmap="magma"); plt.title("ΔE(Pred,GT)"); plt.axis("off"); plt.colorbar(im, fraction=0.046, pad=0.04)
    col = 5
    if other01 is not None:
        dE_other = deltaE76(other01, gt01)
        plt.subplot(1, fig_cols, 5); plt.imshow(other01.permute(1,2,0).cpu()); plt.title("Other"); plt.axis("off")
        plt.subplot(1, fig_cols, 6); im=plt.imshow(dE_other.cpu(), cmap="magma"); plt.title("ΔE(Other,GT)"); plt.axis("off"); plt.colorbar(im, fraction=0.046, pad=0.04)
        col = 7
    plt.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()




# expects you already have rgb_to_lab, soft_tissue_weight, deltaE76, compute_metrics_batch, _save_panel

@torch.no_grad()
def evaluate_saved_dir_vs_gt(pred_dir, dataset, device, out_dir,
                             label="model", sigma=None, visualize_n=16):
    os.makedirs(out_dir, exist_ok=True)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # collect prediction files (recursively) and support TIFF
    files = []
    for ext in ("*.png","*.jpg","*.jpeg","*.tif","*.tiff"):
        files += glob.glob(os.path.join(pred_dir, "**", ext), recursive=True)

    # map normalized basename -> path
    pred_map = { _norm_name(p): p for p in files }

    # dataset basenames, normalized the same way
    basenames = [_norm_name(p) for p in dataset.image_paths]

    # quick debug
    matches = sum(1 for n in basenames if n in pred_map)
    if matches == 0:
        print("[eval] No filename matches.\n  examples dataset:",
              basenames[:5], "\n  examples preds:", list(pred_map.keys())[:5])
        return

    rows, vis = [], 0
    for i, name in enumerate(basenames):
        if name not in pred_map:
            continue

        # GT/phase from dataset ([-1,1] -> [0,1])
        phase, gt = dataset[i]
        phase01 = (phase.clamp(-1,1).unsqueeze(0)*0.5 + 0.5).to(device)
        gt01    = (gt.clamp(-1,1).unsqueeze(0)*0.5 + 0.5).to(device)

        # load saved pred with PIL (TIFF-safe) -> [0,1] tensor
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor
        img = Image.open(pred_map[name]).convert("RGB")
        pred = pil_to_tensor(img).float()/255.0  # [3,H,W]
        pred = pred.unsqueeze(0).to(device)

        # optional tissue mask
        w = soft_tissue_weight(gt01, sigma=sigma) if (sigma is not None) else None

        m = compute_metrics_batch(pred, gt01, mask=w, ssim_metric=ssim)
        rows.append([name, m["mse"][0], m["mse_tissue"][0], m["ssim"][0], m["psnr"][0], m["deltaE"][0]])

        if vis < visualize_n:
            _save_panel(phase01[0], gt01[0], pred[0],
                        f"{label} (sigma={sigma}) : {name}",
                        os.path.join(out_dir, f"panel_{name}.png"))
            vis += 1

    # write CSV
    csv_path = os.path.join(out_dir, f"metrics_{label}_sigma_{'none' if sigma is None else sigma}.csv")
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["name","mse","mse_tissue","ssim","psnr","deltaE"])
        wcsv.writerows(rows)

    print(f"[{label}] {len(rows)} images scored. CSV: {csv_path}. Panels in {out_dir}.")



@no_grad()
def evaluate_saved_dir_both(pred_dir, dataset, device, out_root,
                            label="model", sigma_mask=6.0, visualize_n=16):
    """
    Runs BOTH: global metrics (no mask) and tissue-masked (sigma_mask).
    Writes two CSVs and two panel sets in subfolders.
    """
    out_global = os.path.join(out_root, f"{label}_global")
    out_masked = os.path.join(out_root, f"{label}_masked_sigma{sigma_mask}")

    evaluate_saved_dir_vs_gt(pred_dir, dataset, device, out_global,
                             label=f"{label}_global", sigma=None, visualize_n=visualize_n)
    evaluate_saved_dir_vs_gt(pred_dir, dataset, device, out_masked,
                             label=f"{label}_masked", sigma=sigma_mask, visualize_n=visualize_n)
    


def plot_metric_distributions(dir_a, dir_b, csv_name="comparison.csv", label_a="Model A", label_b="Model B", out_dir=None):
    """
    Compare per-image metrics between two models given CSV outputs.
    
    Parameters
    ----------
    dir_a : str
        Directory containing first CSV file.
    dir_b : str
        Directory containing second CSV file.
    csv_name : str
        CSV filename (must exist in both dirs). Default "comparison.csv".
    label_a : str
        Label for first model (for plotting).
    label_b : str
        Label for second model (for plotting).
    out_dir : str or None
        If provided, save plots here. If None, just show interactively.
    """
    # load csvs
    csv_a = os.path.join(dir_a, csv_name)
    csv_b = os.path.join(dir_b, csv_name)
    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)
    
    # Add label column to distinguish models
    df_a["model"] = label_a
    df_b["model"] = label_b
    
    # Long-form dataframe
    df = pd.concat([df_a, df_b], ignore_index=True)
    
    # Identify metric columns (skip 'name' and 'model')
    metric_cols = [c for c in df.columns if c not in ["name", "model"]]
    
    # Plot distribution for each metric
    for metric in metric_cols:
        plt.figure(figsize=(6, 4))
        sns.kdeplot(data=df, x=metric, hue="model", fill=True, common_norm=False, alpha=0.4)
        plt.title(f"Distribution of {metric.upper()}")
        plt.xlabel(metric)
        plt.ylabel("Density")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, f"{metric}_distribution.png"), dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


#-----------EXTENDED PER CHANNEL METRICS------------

import re 

def _auto_label_from_dir(dir_path: str, prefix="VS-Diff-"):
    """
    Infer a label from a directory with 'exp<NUMBER>' in path, e.g. runs/exp35/... -> VS-Diff-35.
    If not found, returns just the prefix (e.g., 'VS-Diff').
    """
    m = re.search(r"exp(\d+)", dir_path)
    return f"{prefix}{m.group(1)}" if m else prefix


def _safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


# --------------------- distribution / violin from CSV ------------------------

def plot_metric_distributions(dir_a, dir_b, csv_name="metrics.csv",
                              label_a=None, label_b="Model B", out_dir=None):
    """
    KDE distributions for every metric column in the CSV.
    dir_a/dir_b: directories containing csv_name
    If label_a is None, infer from dir_a like 'VS-Diff-<exp>'.
    """
    csv_a = os.path.join(dir_a, csv_name)
    csv_b = os.path.join(dir_b, csv_name)
    df_a = _safe_read_csv(csv_a)
    df_b = _safe_read_csv(csv_b)

    if label_a is None:
        label_a = _auto_label_from_dir(dir_a, prefix="VS-Diff-")

    df_a["model"] = label_a
    df_b["model"] = label_b
    df = pd.concat([df_a, df_b], ignore_index=True)

    metric_cols = [c for c in df.columns if c.lower() not in ["name", "model"]]

    for metric in metric_cols:
        plt.figure(figsize=(6, 4))
        sns.kdeplot(data=df, x=metric, hue="model", fill=True, common_norm=False, alpha=0.4)
        metric_label = metric.upper()
        plt.title(f"Distribution of {metric_label}")
        plt.xlabel(metric_label)
        plt.ylabel("Density")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, f"{metric_label}_distribution.png"), dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_violin_metrics(dir_a, dir_b, csv_name="metrics.csv",
                        label_a=None, label_b="Model B", out_dir=None):
    """
    Violin plots (with quartiles) for every metric column in the CSV.
    """
    csv_a = os.path.join(dir_a, csv_name)
    csv_b = os.path.join(dir_b, csv_name)
    df_a = _safe_read_csv(csv_a)
    df_b = _safe_read_csv(csv_b)

    if label_a is None:
        label_a = _auto_label_from_dir(dir_a, prefix="VS-Diff-")

    df_a["model"] = label_a
    df_b["model"] = label_b
    df = pd.concat([df_a, df_b], ignore_index=True)

    metric_cols = [c for c in df.columns if c.lower() not in ["name", "model"]]

    for metric in metric_cols:
        plt.figure(figsize=(5, 4))
        sns.violinplot(data=df, x="model", y=metric, inner="quartile", cut=0)
        plt.title(f"Violin Plot of {metric.upper()}")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, f"{metric.upper()}_violin.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


# --------- extended computation: per-channel + Lab component metrics ----------

@torch.no_grad()
def compute_channel_and_lab_metrics(
    pred_dir: str,
    dataset,
    device: torch.device,
    out_csv: str,
    batch_size: int = 16,
    num_workers: int = 4,
    csv_of_globals: str = None,   # optional: merge with your existing per-image CSV
):
    """
    For each test image, load pred from `pred_dir` and GT from `dataset`; compute:
      - per-channel MSE/PSNR/SSIM: mse_r/g/b, psnr_r/g/b, ssim_r/g/b
      - Lab component RMSE: dL_rmse, da_rmse, db_rmse
    Saves to out_csv. Optionally merges with global metrics csv (csv_of_globals) on 'name'.

    Works with datasets that return either:
      • tuple/list: (phase, stained_gt)  → use index 1 as GT
      • dict: containing GT under one of ['gt01','stained01','stained','gt','target','image']
    Names are matched to predictions by normalizing basenames with `_norm_name`
    (so files like '0_pred.tif' map to key '0').
    """
    import os, glob
    import numpy as np
    import pandas as pd
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # ---------- index predictions by normalized basename ----------
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.npy")
    pred_files = []
    for pat in exts:
        pred_files += glob.glob(os.path.join(pred_dir, "**", pat), recursive=True)
    pred_map = { _norm_name(p): p for p in pred_files }
    if not pred_map:
        raise FileNotFoundError(f"No prediction files found under {pred_dir}")

    # ---------- helpers to extract name + GT ----------
    gt_key_candidates = ["gt01", "stained01", "stained", "gt", "target", "image"]

    def _name_for_idx(sample, idx: int) -> str:
        # Prefer dataset filenames; fall back to dict name; else index.
        if hasattr(dataset, "image_paths"):
            return _norm_name(dataset.image_paths[idx])
        if isinstance(sample, dict) and ("name" in sample):
            return _norm_name(str(sample["name"]))
        return _norm_name(str(idx))  # gives "0", "1", ... to match "0_pred.tif" → "0"

    def _gt_tensor_from_sample(sample) -> torch.Tensor:
        # tuple/list: (phase, stained_gt)
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            gt = torch.as_tensor(sample[1]).float()
        elif isinstance(sample, dict):
            for k in gt_key_candidates:
                if k in sample:
                    gt = torch.as_tensor(sample[k]).float()
                    break
            else:
                raise KeyError(f"Dict sample missing GT keys {gt_key_candidates}. Got: {list(sample.keys())}")
        else:
            raise TypeError("Sample must be (phase, gt) tuple/list or a dict with a GT key.")

        # Ensure CHW float in [0,1]
        if gt.ndim == 3 and gt.shape[0] in (1, 3):
            pass
        elif gt.ndim == 3 and gt.shape[-1] in (1, 3):
            gt = gt.permute(2, 0, 1)  # HWC → CHW
        else:
            raise ValueError(f"Unexpected GT shape {tuple(gt.shape)}; expected CHW or HWC with 3 channels.")
        return gt.clamp(0, 1)

    # wrapper dataset that yields normalized name + gt01
    class _NameGT(torch.utils.data.Dataset):
        def __len__(self): return len(dataset)
        def __getitem__(self, idx):
            s = dataset[idx]
            name = _name_for_idx(s, idx)     # normalized key like "0"
            gt01 = _gt_tensor_from_sample(s) # [3,H,W] in [0,1]
            return {"name": name, "gt01": gt01}

    def _collate(batch):
        names = [b["name"] for b in batch]
        gt01  = torch.stack([b["gt01"] for b in batch], dim=0)  # [B,3,H,W]
        return {"names": names, "gt01": gt01}

    loader = DataLoader(dataset=_NameGT(), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, collate_fn=_collate)

    ssim = SSIM(data_range=1.0).to(device)

    # load pred by lookup in pred_map
    def _load_pred_tensor_by_key(name_key: str) -> torch.Tensor:
        p = pred_map.get(name_key)
        if p is None:
            raise FileNotFoundError(f"No prediction file matches key '{name_key}' in {pred_dir}")
        if p.endswith(".npy"):
            arr = np.load(p)
            if arr.ndim == 3 and arr.shape[0] in (1, 3):       # CHW
                ten = torch.from_numpy(arr).float()
            elif arr.ndim == 3 and arr.shape[-1] in (1, 3):    # HWC
                ten = torch.from_numpy(arr).permute(2, 0, 1).float()
            else:
                raise ValueError(f"Unexpected npy shape for {p}: {arr.shape}")
            if ten.max() > 1.5: ten = ten / 255.0
            return ten.unsqueeze(0)  # [1,3,H,W]
        else:
            im = Image.open(p).convert("RGB")
            ten = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
            return ten.unsqueeze(0)  # [1,3,H,W]

    rows = []
    missing = 0

    for batch in loader:
        names = batch["names"]                   # already normalized keys
        gt01  = batch["gt01"].to(device)         # [B,3,H,W] in [0,1]

        pred_tensors = []
        for n in names:
            try:
                pred_tensors.append(_load_pred_tensor_by_key(n))
            except FileNotFoundError:
                missing += 1
                pred_tensors.append(None)

        # filter out any missing
        keep_idx = [i for i, t in enumerate(pred_tensors) if t is not None]
        if not keep_idx:
            continue
        names = [names[i] for i in keep_idx]
        gt01  = gt01[keep_idx]
        predB = torch.cat([pred_tensors[i] for i in keep_idx], dim=0).to(device).clamp(0, 1)

        # Per-channel MSE (B,3)
        diff  = predB - gt01
        mse_c = (diff ** 2).flatten(2).mean(-1)

        # Per-channel PSNR
        eps = 1e-12
        psnr_c = 10.0 * torch.log10(1.0 / (mse_c + eps))

        # Per-image, per-channel SSIM
        ssim_img_c = torch.empty_like(mse_c)
        for c in range(3):
            vals = []
            for i in range(predB.size(0)):
                vals.append(ssim(predB[i:i+1, c:c+1], gt01[i:i+1, c:c+1]).item())
            ssim_img_c[:, c] = torch.tensor(vals, device=device)

        # Lab components
        lab_pred = rgb_to_lab(predB)  # [B,3,H,W]
        lab_gt   = rgb_to_lab(gt01)
        dL = (lab_pred[:, 0] - lab_gt[:, 0]).flatten(1)
        da = (lab_pred[:, 1] - lab_gt[:, 1]).flatten(1)
        db = (lab_pred[:, 2] - lab_gt[:, 2]).flatten(1)
        dL_rmse = torch.sqrt((dL ** 2).mean(1))
        da_rmse = torch.sqrt((da ** 2).mean(1))
        db_rmse = torch.sqrt((db ** 2).mean(1))

        for i, n in enumerate(names):
            rows.append({
                "name": n,
                "mse_r": float(mse_c[i, 0].item()),
                "mse_g": float(mse_c[i, 1].item()),
                "mse_b": float(mse_c[i, 2].item()),
                "psnr_r": float(psnr_c[i, 0].item()),
                "psnr_g": float(psnr_c[i, 1].item()),
                "psnr_b": float(psnr_c[i, 2].item()),
                "ssim_r": float(ssim_img_c[i, 0].item()),
                "ssim_g": float(ssim_img_c[i, 1].item()),
                "ssim_b": float(ssim_img_c[i, 2].item()),
                "dL_rmse": float(dL_rmse[i].item()),
                "da_rmse": float(da_rmse[i].item()),
                "db_rmse": float(db_rmse[i].item()),
            })

    df_new = pd.DataFrame(rows)

    # Merge with existing global per-image CSV if provided
    if csv_of_globals and os.path.exists(csv_of_globals):
        base = pd.read_csv(csv_of_globals)

        # --- NEW: force same dtype for the join key ---
        base["name"]   = base["name"].astype(str)
        df_new["name"] = df_new["name"].astype(str)

        df = base.merge(df_new, on="name", how="left")
    else:
        df = df_new

    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}  ({len(df)} rows). Skipped missing preds: {missing}.")


def plot_violin_extended(csv_a, csv_b, label_a="Model A", label_b="Model B", out_dir=None):
    """
    Violin plots for:
      - per-channel MSE/PSNR/SSIM
      - Lab component RMSE (dL/da/db)
    from two extended CSVs (as produced by compute_channel_and_lab_metrics).
    """
    df_a = _safe_read_csv(csv_a); df_a["model"] = label_a
    df_b = _safe_read_csv(csv_b); df_b["model"] = label_b
    df = pd.concat([df_a, df_b], ignore_index=True)

    groups = {
        "MSE (per channel)": ["mse_r","mse_g","mse_b"],
        "PSNR (per channel)": ["psnr_r","psnr_g","psnr_b"],
        "SSIM (per channel)": ["ssim_r","ssim_g","ssim_b"],
        "Lab component RMSE": ["dL_rmse","da_rmse","db_rmse"],
    }

    for title, cols in groups.items():
        present = [c for c in cols if c in df.columns]
        for c in present:
            plt.figure(figsize=(5, 4))
            sns.violinplot(data=df, x="model", y=c, inner="quartile", cut=0)
            plt.title(f"{title}: {c.upper()}")
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                plt.savefig(os.path.join(out_dir, f"violin_{c}.png"),
                            dpi=150, bbox_inches="tight")
                plt.close()
            else:
                plt.show()

