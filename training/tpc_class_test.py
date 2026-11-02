# import torch
# from icu_benchmarks.models.dl_models.tpc import TPC

# model = TPC(
#     input_size=(8, 48, 24),
#     static_size=4,
#     n_layers=3,
#     kernel_size=3,
#     temp_kernels=4,
#     point_size=8,
#     dropout=0.1,
#     run_mode="regression",
# )

# x = torch.randn(8, 48, 24)
# static = torch.randn(8, 4)

# output = model(x, static)
# print("Output shape (expect (8, 1)):", output.shape)
# print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))



import torch
from icu_benchmarks.models.dl_models.tpc import TPC
from icu_benchmarks.constants import RunMode

model = TPC(
    input_size=(8, 48, 24),
    static_size=4,
    n_layers=3,
    kernel_size=3,
    temp_kernels=4,
    point_size=8,
    dropout=0.1,
    run_mode=RunMode.regression,
)

x = torch.randn(8, 48, 24)
static = torch.randn(8, 4)

output = model(x, static)
print("Output shape (expect (8, 1)):", output.shape)
print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))