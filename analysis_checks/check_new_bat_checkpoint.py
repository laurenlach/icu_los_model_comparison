"""
Step 1 of the BAT re-run: check the MIMIC-IV + eICU checkpoint (never saw MIMIC-III)
can be bridged into your BAT regression model exactly like the old one.

Run from your YAIB folder (same place you ran bridge_bat_checkpoint.py):
    python check_new_bat_checkpoint.py

Nothing is overwritten. The new checkpoint is saved under a new name.
"""
import os
import urllib.request
import torch
from icu_benchmarks.models.dl_models.bat import BAT, RegressionHead

OLD_CKPT = "pretrained_checkpoints/bat_pretrained.ckpt"
NEW_CKPT = "pretrained_checkpoints/bat_pretrained_eicu_mimiciv.ckpt"
NEW_URL = ("https://huggingface.co/Katja-Jagd/bat-pretrained-eicu-mimiciv-base"
           "/resolve/main/model.ckpt")

# ---------------------------------------------------------------- 1. download
# Why: get the checkpoint pretrained only on MIMIC-IV and eICU.
if not os.path.exists(NEW_CKPT):
    print(f"Downloading new checkpoint to {NEW_CKPT} ...")
    urllib.request.urlretrieve(NEW_URL, NEW_CKPT)
print(f"New checkpoint size: {os.path.getsize(NEW_CKPT) / 1e6:.1f} MB\n")

old = torch.load(OLD_CKPT, map_location="cpu", weights_only=False)
new = torch.load(NEW_CKPT, map_location="cpu", weights_only=False)

# ---------------------------------------------------------------- 2. settings
# Why: the bridge hardcodes embed size, layers, heads and dropouts from the OLD
# checkpoint. If the new one was pretrained with different settings, we need to know.
print("=== Hyperparameters that DIFFER between old and new checkpoint ===")
hp_old = dict(old.get("hyper_parameters", {}))
hp_new = dict(new.get("hyper_parameters", {}))
diffs = 0
for k in sorted(set(hp_old) | set(hp_new)):
    a, b = hp_old.get(k, "<missing>"), hp_new.get(k, "<missing>")
    if str(a) != str(b):
        print(f"  {k}:  old={a}   new={b}")
        diffs += 1
print("  (none)" if diffs == 0 else f"  {diffs} difference(s)")
print()

# ---------------------------------------------------------------- 3. weights
# Why: every encoder tensor must exist in both with the same shape,
# otherwise the new weights cannot go into the same model.
enc_old = {k: v for k, v in old["state_dict"].items() if k.startswith("model.encoder_class.")}
enc_new = {k: v for k, v in new["state_dict"].items() if k.startswith("model.encoder_class.")}
print("=== Encoder weight comparison ===")
print(f"  Encoder tensors: old={len(enc_old)}  new={len(enc_new)}")
only_old = sorted(set(enc_old) - set(enc_new))
only_new = sorted(set(enc_new) - set(enc_old))
shape_bad = [k for k in set(enc_old) & set(enc_new) if enc_old[k].shape != enc_new[k].shape]
print(f"  Only in old: {only_old if only_old else 'none'}")
print(f"  Only in new: {only_new if only_new else 'none'}")
print(f"  Shape mismatches: {[(k, tuple(enc_old[k].shape), tuple(enc_new[k].shape)) for k in shape_bad] or 'none'}")

# Why: confirm it really is a different model, not the same file twice.
same = [k for k in set(enc_old) & set(enc_new) if torch.equal(enc_old[k], enc_new[k])]
print(f"  Tensors identical to old checkpoint: {len(same)} of {len(enc_new)} (should be 0 or close to 0)\n")

# ---------------------------------------------------------------- 4. bridge
# Why: same construction as bridge_bat_checkpoint.py, so the only thing that changes
# in the re-run is which pretrained weights go in.
# Built with the NEW checkpoint's own architecture (6 layers, embed size 64),
# because its weights only fit a model of that shape.
new_model = BAT(
    input_size=(64, 48, 24),
    value_embed_size=hp_new["value_embed_size"],
    layers=hp_new["layers"],
    heads=hp_new.get("heads", 1),
    dropout=hp_new["dropout"],
    attn_dropout=hp_new["attn_dropout"],
    use_mask=False,
    prediction_head=RegressionHead,
    prediction_head_kwargs={"output_dim": 1},
    static_count=4,
)
print("=== Bridging into BAT regression model ===")
try:
    result = new_model.load_state_dict(enc_new, strict=False)
    print(f"  Missing keys ({len(result.missing_keys)}), expected = the untrained regression head only:")
    for k in result.missing_keys:
        print("    ", k)
    print(f"  Unexpected keys ({len(result.unexpected_keys)}), must be 0:")
    for k in result.unexpected_keys:
        print("    ", k)
except RuntimeError as e:
    print("  FAILED, shapes do not fit the model:\n", e)
    raise SystemExit(1)

# Why: strict=False silently skips keys whose names don't match, so check the
# weights actually landed in the model.
model_sd = new_model.state_dict()
landed = sum(1 for k, v in enc_new.items() if k in model_sd and torch.equal(model_sd[k], v))
print(f"  Encoder tensors actually loaded into model: {landed} of {len(enc_new)}")

# Why: the new checkpoint is a different size, so the model's parameter count
# (reported in your Methodology) will change.
n_new = sum(p.numel() for p in new_model.parameters() if p.requires_grad)
print(f"  Trainable parameters, new BAT model: {n_new:,}")

# Pass depends only on the new weights fitting the new-shaped model.
# (Differences from the OLD checkpoint above are expected now and just for information.)
ok = len(result.unexpected_keys) == 0 and landed == len(enc_new)
print("\nRESULT:", "PASS, safe to retrain with this checkpoint" if ok else "CHECK OUTPUT ABOVE before going further")