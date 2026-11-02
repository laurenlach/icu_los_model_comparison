import torch
import torch.nn as nn
from icu_benchmarks.models.dl_models.layers import Chomp1d

# Dummy data matching our real shapes
B, F, T = 8, 48, 24
X = torch.randn(B, F, T)  # channels-first, as Conv1d expects
print("Input X shape:", X.shape)

kernel_size = 3
temp_kernels = 4   # how many output channels PER SENSOR the temporal conv produces
point_size = 8     # how many output features the pointwise layer produces

dilation = 1
padding = (kernel_size - 1) * dilation  # causal padding amount


temp_conv = nn.Conv1d(
    in_channels=F,
    out_channels=F * temp_kernels,   # each sensor produces temp_kernels outputs
    kernel_size=kernel_size,
    padding=padding,
    dilation=dilation,
    groups=F,
)
chomp = Chomp1d(padding)  # trims the extra padding so output stays causal

temp_out = chomp(temp_conv(X))
print("Temporal conv output shape (expect (B, F*temp_kernels, T)):", temp_out.shape)


X_for_pointwise = X.permute(0, 2, 1).reshape(B * T, F)  # (B*T, F)
pointwise = nn.Linear(F, point_size)
point_out = pointwise(X_for_pointwise)
point_out = point_out.view(B, T, point_size)  # back to (B, T, point_size)

print("Pointwise output shape (expect (B, T, point_size)):", point_out.shape)