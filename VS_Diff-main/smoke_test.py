import torch
from vsdiff_model import build_model_and_scheduler, load_checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m, s = build_model_and_scheduler(device)
print("model built OK", device)
