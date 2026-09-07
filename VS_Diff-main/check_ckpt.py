import torch
print(torch.__version__)
with open('checkpoints/best.pth','rb') as f:
	print(f.read(8))
d = torch.load('checkpoints/best.pth', map_location='cpu', weights_only=True)
print(type(d))
