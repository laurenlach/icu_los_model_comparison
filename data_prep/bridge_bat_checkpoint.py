import torch
from icu_benchmarks.models.dl_models.bat import BAT, RegressionHead

# Step A: Load the pretrained checkpoint
ckpt = torch.load("pretrained_checkpoints/bat_pretrained.ckpt", map_location="cpu")
pretrained_state_dict = ckpt["state_dict"]

# Step B: Keep only the encoder weights, discard the forecasting head
encoder_weights = {
    k: v for k, v in pretrained_state_dict.items()
    if k.startswith("model.encoder_class.")
}
print(f"Found {len(encoder_weights)} encoder weight tensors to transplant.")

# Step C: Create a fresh BAT model, matching the pretrained architecture exactly,
# but with a regression head instead of a classification/forecasting head
new_model = BAT(
    input_size=(64, 48, 24),          # (batch, sensors, timepoints) - matches pretrained
    value_embed_size=128,
    layers=2,
    heads=1,
    dropout=0.36427,
    attn_dropout=0.20659,
    use_mask=False,
    prediction_head=RegressionHead,
    prediction_head_kwargs={"output_dim": 1},
    static_count=4,
)

# Step D: Load the encoder weights into the new model
# strict=False because the new model's head doesn't match the old one - that's expected
result = new_model.load_state_dict(encoder_weights, strict=False)

print("Missing keys (expected: the new regression head, since it's untrained):")
print(result.missing_keys)
print("\nUnexpected keys (should be empty if everything matched cleanly):")
print(result.unexpected_keys)