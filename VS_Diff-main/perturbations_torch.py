import torch
import torch.nn.functional as F

# PyTorch port of ../perturbations.py (same formulas, same [-1, 1] convention).
# All functions expect/return a torch tensor image shaped [C, H, W].


def perturb_gaussian(image, std=0.1):
    noise = torch.randn_like(image) * std
    return torch.clamp(image + noise, -1, 1)


def perturb_blur(image, kernel_size=5):
    x = image.unsqueeze(0)  # [1, C, H, W]
    pad = kernel_size // 2
    blurred = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad, count_include_pad=False)
    return blurred.squeeze(0)


def perturb_salt_pepper(image, amount=0.05):
    mask = torch.rand_like(image)
    image = torch.where(mask < amount / 2, torch.full_like(image, -1.0), image)
    image = torch.where(mask > 1 - (amount / 2), torch.full_like(image, 1.0), image)
    return image


def perturb_contrast(image, factor=1.0):
    return torch.clamp(image * factor, -1.0, 1.0)
