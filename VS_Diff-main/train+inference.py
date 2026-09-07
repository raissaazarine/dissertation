#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install diffusers
# ! pip install tensorboard
# ! pip install seaborn 


# In[35]:


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
from torchmetrics import StructuralSimilarityIndexMeasure
import shutil
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import MS_SSIM
from torch.cuda.amp import autocast, GradScaler

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import virt_stain_utils2 as vsu



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ##### Data loading and preprocessing

# In[5]:


# Dataset class
class VirtualStainingDataset(Dataset):
    def __init__(self, root_dir, phase_transform=None, stain_transform=None):
        paths = glob.glob(os.path.join(root_dir, '*.tif'))
        self.image_paths = sorted(
            paths,
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )
        self.phase_transform = phase_transform
        self.stain_transform = stain_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = image.resize((512, 256))  
        image = np.array(image)

        stained = image[:, :256, :]       # Left stained image
        phase = image[:, 256:, 0]         # Right FPM image

        stained = Image.fromarray(stained)
        phase = Image.fromarray(phase)

        if self.phase_transform:
            phase = self.phase_transform(phase)
        if self.stain_transform:
            stained = self.stain_transform(stained)

        return phase, stained



# In[6]:


# 2. Transform
stain_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

phase_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])



# In[ ]:


# 3. Dataset + Dataloader
full_dataset = VirtualStainingDataset(
    '/v7/train',
    phase_transform=phase_transform,
    stain_transform=stain_transform
)

test_dataset = VirtualStainingDataset(
    '/v7/val',
    phase_transform=phase_transform,
    stain_transform=stain_transform
)   

g = torch.Generator().manual_seed(42)                                
train_len = int(0.7 * len(full_dataset))
val_len = len(full_dataset) - train_len
train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len], generator=g)
debug_dataset = torch.utils.data.Subset(full_dataset, range(5))
debug_loader = DataLoader(debug_dataset, batch_size=16, shuffle=True, num_workers=8)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=16, num_workers=8) 
test_loader = DataLoader(test_dataset, batch_size=16, num_workers=8)

ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


# ##### Defining Model 
# 
# - UNet
# - Current batch size = 16
# - Epochs variable
# - Timesteps = 1000 

# In[9]:


# 4. Model from diffusers (huggingface)

model = UNet2DModel(
    sample_size=256,  
    in_channels=4,    # 3 stained + 1 phase
    out_channels=3,   # Predict noise on RGB stained image
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
).cuda()

noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="linear"
)

noise_scheduler.set_timesteps(
    noise_scheduler.config.num_train_timesteps,
    device=device
)

inference_noise_scheduler = DDIMScheduler(
    num_train_timesteps=noise_scheduler.config.num_train_timesteps,
    beta_schedule="linear"
)


# ##### Training Loop

# In[7]:


from torch.cuda.amp import autocast, GradScaler



# Initialize once
scaler = GradScaler() 

checkpoint = "/cs/student/projects2/aisd/2024/amanivan/virtual_staining/meu/runs/exp1/checkpoints/checkpoint.pth"


# ##### LR Scheduler

# In[6]:


def denorm(tensor):
    return tensor * 0.5 + 0.5


# ##### Driver Cell - Training

# In[ ]:


# autoreload so edits to virt_stain_utils.py take effect immediately
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import math
import torch

# ——— HYPERPARAMS ———
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
total_epochs = 200
log_dir = "runs/exp35"
alpha = 1.0  # loss mix weight for noise MSE
sigma = 1.0  
# ——————————————

writer = SummaryWriter(log_dir)
# warmup / lr scheduler
steps_per_epoch = len(train_loader)
total_steps = total_epochs * steps_per_epoch
scheduler = vsu.make_warmup_cosine_scheduler(optimizer, total_steps, warmup_frac=0.05)
global_step = 0
best_val_ssim = 0.0

# resume if possible
ckpt_path = f"{log_dir}/checkpoints/checkpoint.pth"
start_epoch, global_step, best_val_ssim = vsu.load_checkpoint(
    model, optimizer, scaler,
    checkpoint_path=ckpt_path
)

for epoch in range(start_epoch, total_epochs):
    beta = vsu.beta_ramp(epoch, ramp_epochs=50, target_beta=0.002)
    torch.cuda.empty_cache()


    # ─── 1) TRAIN ────────────────────────────────────────────
    try:
        train_stats, global_step = vsu.train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            noise_scheduler,
            alpha,
            beta,
            device,
            lr_scheduler=scheduler,
            global_step=global_step,
            sigma=sigma
        )

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[OOM] Epoch {epoch+1}, step {global_step}. Aborting.")
            # optional: save a crash checkpoint for clean resuming
            state = {
                "epoch": epoch,               # current epoch not completed
                "step":  global_step,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict":    scaler.state_dict(),
                "lr_scheduler_state_dict": scheduler.state_dict(),
                "best_val_ssim":        best_val_ssim,
            }
            vsu.save_checkpoint(state, is_best=False, checkpoint_dir=f"{log_dir}/checkpoints")
            raise  # or: raise SystemExit(1)
        else:
            raise
        
    # skip if training produced NaNs
    if math.isnan(train_stats.get("train_mse", float("nan"))) or math.isnan(train_stats.get("train_ssim", float("nan"))):
        print(f"[WARN] NaN in training stats at epoch {epoch+1}, skipping validation/checkpoint")
        continue

    # ─── 2) VALIDATE ─────────────────────────────────────────
    val_stats = vsu.validate_metrics(
        model=model,
        loader=val_loader,
        noise_scheduler=noise_scheduler,
        device=device
    )

    # ─── 2.1) SANITY CHECK ───────────────────────────────────
    if (epoch + 1) % 5 == 0:
        sanity_metrics = vsu.run_sanity_check(
            model=model,
            val_loader=val_loader,
            train_noise_scheduler=noise_scheduler,
            inference_scheduler=inference_noise_scheduler,
            writer=writer,
            epoch=epoch,
            device=device,
            num_visualize=4,
            num_inference_steps=50,
            eta=0.0,
            log_dir=log_dir,
            verbose=False,  # set True for detailed debug plots
        )
        print(
            f"[Sanity] Epoch {epoch+1}: "
            f"SSIM one-step {sanity_metrics['ssim_one_step']:.4f}, full {sanity_metrics['ssim_full']:.4f}; "
            f"MSE one-step {sanity_metrics['mse_one_step']:.5f}, full {sanity_metrics['mse_full']:.5f}"
        )
        writer.add_scalars("sanity/ssim", {
            "one_step": sanity_metrics["ssim_one_step"],
            "full": sanity_metrics["ssim_full"],
        }, epoch)
        writer.add_scalars("sanity/mse", {
            "one_step": sanity_metrics["mse_one_step"],
            "full": sanity_metrics["mse_full"],
        }, epoch)

    # ─── 3) PRINT SUMMARY ─────────────────────────────────────
    print(
        f"Epoch {epoch+1}/{total_epochs}  "
        f"Train MSE={train_stats['train_mse']:.4f}, SSIM={train_stats['train_ssim']:.4f}  |  "
        f"Val   MSE={val_stats['val_mse']:.4f}, SSIM={val_stats['val_ssim']:.4f}"
    )

    # ─── 4) LOG TO TENSORBOARD ───────────────────────────────
    writer.add_scalars("mse", {
        "train": train_stats["train_mse"],
        "val": val_stats["val_mse"],
        "train_lab_chroma": train_stats["train_lab_chroma"]
    }, epoch)
    writer.add_scalars("ssim", {
        "train": train_stats["train_ssim"],
        "val": val_stats["val_ssim"]
    }, epoch)

    writer.add_scalars("mse_channels", {
        "train_r": train_stats["train_mse_r"],
        "train_g": train_stats["train_mse_g"],
        "train_b": train_stats["train_mse_b"],
        "val_r": val_stats["val_mse_r"],
        "val_g": val_stats["val_mse_g"],
        "val_b": val_stats["val_mse_b"],
    }, epoch)
    writer.add_scalars("ssim_channels", {
        "train_r": train_stats["train_ssim_r"],
        "train_g": train_stats["train_ssim_g"],
        "train_b": train_stats["train_ssim_b"],
        "val_r": val_stats["val_ssim_r"],
        "val_g": val_stats["val_ssim_g"],
        "val_b": val_stats["val_ssim_b"],
    }, epoch)

    writer.flush()

    # ─── 5) CHECKPOINT ────────────────────────────────────────
    is_best = val_stats["val_ssim"] > best_val_ssim
    best_val_ssim = max(best_val_ssim, val_stats["val_ssim"])

    state = {
        "epoch": epoch + 1, 
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_ssim": best_val_ssim,
        "lr_scheduler_state_dict": scheduler.state_dict(),
    }

    vsu.save_checkpoint(state, is_best, checkpoint_dir=f"{log_dir}/checkpoints")




# ##### Driver - Inference

# In[ ]:


log_dir = "runs/exp32"
ckpt_path = f"{log_dir}/checkpoints/checkpoint.pth"
start_epoch, global_step, best_val_ssim = vsu.load_checkpoint(
    model, optimizer, scaler,
    checkpoint_path=ckpt_path
)

out_dir = f"{log_dir}/test_preds_ddim"
vsu.infer_and_save_ddim(
    model=model,
    loader=test_loader,
    scheduler=inference_noise_scheduler,  #  DDIM scheduler
    device=device,
    out_dir=out_dir,
    steps=100,
    eta=0.0,                # deterministic DDIM
    use_dataset_names=True, # uses dataset.image_paths for names if available
    save_triptych=True     # set True to also save [phase|GT|pred] panels
)
print("Saved predictions to:", out_dir)


# In[ ]:


# fixed reference sample
phase_val, stained_val = next(iter(val_loader))
stained_val = stained_val.to(device)[:1]  # single example

# run the t=1 diffusion math validation
ssim_val, mse_val = vsu.check_one_step_t1(noise_scheduler, stained_val, device)



# ##### Driver - Comparison with GT and Pix2Pix

# In[ ]:


log_dir = "runs/exp34"


# results write location
eval_root = f"{log_dir}/eval_saved_diff"

# Run both comparisons on saved predictions
vsu.evaluate_saved_dir_both(
    pred_dir=f"{log_dir}/test_preds_ddim",  
    dataset=test_dataset,                  
    device=device,
    out_root=eval_root,
    label="diffusion",
    sigma_mask=6.0,                         # tissue-masked run
    visualize_n=24
)


# In[22]:


# Global (no mask)
log_dir = "runs/exp34"
pix2pix_dir = "/cs/student/projects2/aisd/2024/amanivan/virtual_staining/Pix2Pix Polypv7 Inference/val"
vsu.evaluate_saved_dir_vs_gt(
    pred_dir=f"{log_dir}/test_preds_ddim",
    dataset=test_dataset, device=device,
    out_dir=f"{log_dir}/eval_saved_diff/global",
    label="diffusion", sigma=None, visualize_n=24
)


# In[ ]:


# Global (no mask)
log_dir = "runs/exp34"
pix2pix_dir = "/cs/student/projects2/aisd/2024/amanivan/virtual_staining/Pix2Pix Polypv7 Inference/val"
vsu.evaluate_saved_dir_vs_gt(
    pred_dir=f"{pix2pix_dir}",
    dataset=test_dataset, device=device,
    out_dir=f"{pix2pix_dir}/eval_saved_diff/global",
    label="Pix2pix", sigma=None, visualize_n=24
)


# In[ ]:


import re

dir_a = "runs/exp34/eval_saved_diff/global"
dir_b = "/cs/student/projects2/aisd/2024/amanivan/virtual_staining/Pix2Pix Polypv7 Inference/val/eval_saved_diff/global"

match = re.search(r"exp(\d+)", dir_a)
if match:
    exp_num = match.group(1)
    label_a = f"VS-Diff-{exp_num}"
else:
    label_a = "VS-Diff"

vsu.plot_metric_distributions(
    dir_a=dir_a,
    dir_b=dir_b,
    csv_name="metrics_sigma_none.csv",  # change if file isn’t named e.g. "comparison.csv"
    label_a=label_a,
    label_b="Pix2Pix",
    out_dir=f"plots/exp{exp_num}_vs_other"  # set None to just display
)

