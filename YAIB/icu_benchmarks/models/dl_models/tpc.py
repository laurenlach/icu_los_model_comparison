import gin
import torch
import torch.nn as nn

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.dl_models.layers import Chomp1d
from icu_benchmarks.models.wrappers import DLPredictionWrapper


@gin.configurable
class TPC(DLPredictionWrapper):
    """
    Temporal Pointwise Convolution (TPC), adapted from Rocheteau et al. (2021)
    for single-value-per-stay Length of Stay regression.

    Architecture per layer:
      - Temporal conv: each sensor gets its own independent causal filter (groups=F)
      - Pointwise conv (Linear): looks across all sensors at each timestep
      - Skip connections: each layer's pointwise output is retained and passed
        forward to every subsequent layer, so later layers see everything
        computed so far (not just the immediately previous layer)

    Static patient features (age, sex, height, weight) are folded in only at
    the very end, combined with the final timestep's accumulated pointwise
    representation, before the final prediction layer.

    Unlike Rocheteau's original per-timestep design, this version predicts a
    single LOS value per patient, using the first 24 hours as input - matching
    the same "early warning" framing used for BAT in this study.
    """

    _supported_run_modes = [RunMode.regression]

    def __init__(
        self,
        input_size,             # (batch, sensors, timepoints)
        static_size,            # number of static features (e.g. 4)
        n_layers=3,
        kernel_size=3,
        temp_kernels=4,
        point_size=8,
        dropout=0.1,
        lr=1e-4,
        optimizer=torch.optim.Adam,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            input_size=input_size,
            static_size=static_size,
            n_layers=n_layers,
            kernel_size=kernel_size,
            temp_kernels=temp_kernels,
            point_size=point_size,
            dropout=dropout,
            lr=lr,
            optimizer=optimizer,
            **kwargs,
        )
        self.save_hyperparameters()

        self.F = input_size[1]          # number of sensors
        self.T = input_size[2]          # number of timepoints
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.temp_kernels = temp_kernels
        self.point_size = point_size

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        # Build each layer's temporal conv + pointwise linear
        self.layer_modules = nn.ModuleDict()

        Y = 0    # temp_kernels produced by the previous layer (0 for layer 0)
        Zt = 0   # cumulative pointwise features from all previous layers

        for i in range(n_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            channels_per_sensor = 1 + Y  # original value slot + prior temporal outputs

            temp_conv = nn.Conv1d(
                in_channels=self.F * channels_per_sensor,
                out_channels=self.F * temp_kernels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=self.F,
            )
            chomp = Chomp1d(padding)

            pointwise_input_dim = self.F + Zt
            pointwise = nn.Linear(pointwise_input_dim, point_size)

            self.layer_modules[str(i)] = nn.ModuleDict({
                "temp_conv": temp_conv,
                "chomp": chomp,
                "pointwise": pointwise,
            })

            Zt += point_size
            Y = temp_kernels

        self.Zt_final = Zt

        # Final prediction layer: combines last-timestep representation
        # with static features, outputs one LOS value per patient.
        # self.final_linear = nn.Linear(self.Zt_final + static_size, 1)
        self.final_linear = nn.Linear(self.Zt_final + (self.F * temp_kernels) + static_size, 1)

        # Helps CustomDLPredictionWrapper-style metric code that may look for `.logit`
        self.logit = nn.Identity()
        
    def forward(self, x, static):
        # x: (B, F, T) - sensor data, channels-first
        # static: (B, static_size)
        B, F, T = x.shape

        current_X = x.unsqueeze(2)  # (B, F, 1, T) - "slot" dimension for per-sensor channels
        point_skip_list = []
        temp_skip_list = []
        Y = 0

        for i in range(self.n_layers):
            layer = self.layer_modules[str(i)]
            channels_per_sensor = 1 + Y

            conv_input = current_X.reshape(B, F * channels_per_sensor, T)
            temp_out = layer["chomp"](layer["temp_conv"](conv_input))  # (B, F*temp_kernels, T)

            # Pointwise layer: original sensor values at each timestep + everything
            # accumulated by previous layers so far
            flat_X = x.permute(0, 2, 1).reshape(B * T, F)
            if point_skip_list:
                prior_points = torch.cat(point_skip_list, dim=-1)  # (B, T, Zt_so_far)
                prior_points = prior_points.reshape(B * T, -1)
                pointwise_input = torch.cat([flat_X, prior_points], dim=-1)
            else:
                pointwise_input = flat_X

            point_out = self.dropout(self.relu(layer["pointwise"](pointwise_input)))
            point_out = point_out.view(B, T, self.point_size)
            point_skip_list.append(point_out)

            # Prepare next layer's temporal input: original value + all temporal
            # outputs produced so far, kept grouped per sensor
            temp_out_reshaped = temp_out.view(B, F, self.temp_kernels, T)
            temp_skip_list.append(temp_out_reshaped)
            original_slot = x.unsqueeze(2)
            current_X = torch.cat([original_slot, temp_out_reshaped], dim=2)

            Y = self.temp_kernels

        # Combine all layers' pointwise outputs, take the last timestep only
        combined_points = torch.cat(point_skip_list, dim=-1)  # (B, T, Zt_final)
        final_timestep_points = combined_points[:, -1, :]     # (B, Zt_final)

        # Final layer's temporal-convolution output, at the last timestep -
        # this is the piece that was being computed but never used before
        last_temp_out = temp_skip_list[-1]                    # (B, F, temp_kernels, T)
        final_timestep_temp = last_temp_out[:, :, :, -1].reshape(B, -1)  # (B, F*temp_kernels)

        final_combined = torch.cat([final_timestep_points, final_timestep_temp, static], dim=-1)
        prediction = self.final_linear(final_combined)  # (B, 1)

        return prediction
