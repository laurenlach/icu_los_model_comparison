import pandas as pd
from scipy.stats import spearmanr

ORGANS = ["sofa_cardio_comp", "sofa_renal_comp", "sofa_liver_comp",
          "sofa_coag_comp", "sofa_resp_comp"]

print(f"{'organ':20s}{'stays':>8s}{'TPC':>9s}{'BAT':>9s}{'p TPC':>12s}{'p BAT':>12s}")
for o in ORGANS:
    t = pd.read_csv(f"alignment_{o}_TPC_fixed.csv")
    b = pd.read_csv(f"alignment_{o}_BAT_healthybaseline.csv")
    tr, tp_ = spearmanr(t[o], t["attribution_mass"])
    br, bp_ = spearmanr(b[o], b["attribution_mass"])
    print(f"{o:20s}{len(t):8d}{tr:9.3f}{br:9.3f}{tp_:12.1e}{bp_:12.1e}")