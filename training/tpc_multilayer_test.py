
import torch
import torch.nn as nn
from icu_benchmarks.models.dl_models.layers import Chomp1d

B, F, T = 8, 48, 24
X = torch.randn(B, F, T)

n_layers = 3
kernel_size = 3
temp_kernels = 4
point_size = 8

Y = 0
Zt = 0

current_X = X.unsqueeze(2)  # (B, F, 1, T) - "1" slot for original value, will grow each layer
point_skip_list = []

for i in range(n_layers):
    dilation = 2 ** i
    padding = (kernel_size - 1) * dilation

    # Reshape (B, F, 1+Y, T) -> (B, F*(1+Y), T) for the grouped conv
    conv_input = current_X.reshape(B, F * (1 + Y), T)

    temp_conv = nn.Conv1d(
        in_channels=F * (1 + Y),
        out_channels=F * temp_kernels,
        kernel_size=kernel_size,
        padding=padding,
        dilation=dilation,
        groups=F,
    )
    chomp = Chomp1d(padding)
    temp_out = chomp(temp_conv(conv_input))  # (B, F*temp_kernels, T)
    print(f"Layer {i} - temporal output shape:", temp_out.shape)

    # Pointwise layer 
    flat_X = X.permute(0, 2, 1).reshape(B * T, F)
    if point_skip_list:
        prior_points = torch.cat(point_skip_list, dim=-1).permute(0, 2, 1).reshape(B * T, Zt)
        pointwise_input = torch.cat([flat_X, prior_points], dim=-1)
    else:
        pointwise_input = flat_X

    pointwise = nn.Linear(pointwise_input.shape[-1], point_size)
    point_out = pointwise(pointwise_input).view(B, T, point_size)
    print(f"Layer {i} - pointwise output shape:", point_out.shape)

    point_skip_list.append(point_out)
    Zt += point_size

    #Build next layer's temporal input
    temp_out_reshaped = temp_out.view(B, F, temp_kernels, T)  # per-sensor grouping restored
    original_slot = X.unsqueeze(2)  # (B, F, 1, T)
    current_X = torch.cat([original_slot, temp_out_reshaped], dim=2)  # (B, F, 1+temp_kernels, T)

    Y = temp_kernels  # for next iteration, if we were accumulating - kept simple here per-layer

print("\nFinal Zt (total accumulated pointwise features):", Zt)
print("Number of skip connections stored:", len(point_skip_list))



static_features = torch.randn(B, 4) # 4 static features per patient

# Combine all pointwise skip-connection outputs into one representation
combined_points = torch.cat(point_skip_list, dim=-1)  # (B, T, Zt)

# Take the representation from the LAST timestep only, since we want
# ONE prediction per patient (not per-timestep, per our earlier BAT decision)
final_timestep_repr = combined_points[:, -1, :]  # (B, Zt)

# Now concatenate with static features
final_combined = torch.cat([final_timestep_repr, static_features], dim=-1)
print("\nFinal combined representation shape (expect (B, Zt+4)):", final_combined.shape)

# Final prediction layer: one number per patient
final_linear = nn.Linear(final_combined.shape[-1], 1)
prediction = final_linear(final_combined)
print("Final prediction shape (expect (B, 1)):", prediction.shape)