import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from diffusers import UNet2DModel, DDIMScheduler

import virt_stain_utils2 as vsu

# ---- Dataset (copied from train+inference.ipynb, cells 2-4, not in virt_stain_utils2.py) ----

stain_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

phase_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


class VirtualStainingDataset(Dataset):
    def __init__(self, root_dir, phase_transform=phase_transform, stain_transform=stain_transform):
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

        stained = image[:, :256, :]       # left half: stained/GT
        phase = image[:, 256:, 0]         # right half: phase (single channel)

        stained = Image.fromarray(stained)
        phase = Image.fromarray(phase)

        if self.phase_transform:
            phase = self.phase_transform(phase)
        if self.stain_transform:
            stained = self.stain_transform(stained)

        return phase, stained


# ---- Model + scheduler (copied from train+inference.ipynb, cell 7) ----

def build_model_and_scheduler(device):
    model = UNet2DModel(
        sample_size=256,
        in_channels=4,     # 3 stained + 1 phase
        out_channels=3,    # predicted noise on RGB stained image
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    return model, scheduler


def load_checkpoint(model, checkpoint_path, device):
    """Inference-only checkpoint load (no optimizer/scaler, unlike vsu.load_checkpoint)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"restored checkpoint: {checkpoint_path} "
          f"(epoch {ckpt.get('epoch', '?')}, step {ckpt.get('step', '?')})")


def sample(model, phase, scheduler, device, steps, eta):
    """phase: [B, 1, H, W] in [-1, 1]. Returns [B, 3, H, W] in [-1, 1].
    Delegates to virt_stain_utils2.ddim_sample_full so the sampling loop
    isn't duplicated here."""
    return vsu.ddim_sample_full(model, phase, scheduler, device,
                                 num_inference_steps=steps, eta=eta)
