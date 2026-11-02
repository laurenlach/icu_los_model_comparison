import torch
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.dl_models.tpc import TPC

m = TPC(input_size=(64,48,24), static_size=4, n_layers=3, kernel_size=3, temp_kernels=4, point_size=8, dropout=0.1, run_mode=RunMode.regression)
m.eval()

x = torch.randn(2, 48, 24, requires_grad=True)
static = torch.randn(2, 4)

out = m(x, static)
out.sum().backward()

grad_per_hour = x.grad.abs().sum(dim=(0, 1))
print("Gradient magnitude per hour:")
for h in range(24):
    print(f"  hour {h}: {grad_per_hour[h].item():.6f}")