"""Checks that no forecast window uses the record's leading, never-observed hours.

  python check_leading_prefix.py --repo /path/to/STORM-PhysNet-L2

_ffill_leading() fills the hours before the first observed GOES flux (the first 139 h, March 2010)
with the first observed value. valid_starts() only requires the last input hour and the three target
hours to be observed, so this script confirms from the data that every window in every block starts
after those hours. GOES-15 flux is almost unobserved until January 2011, so the first valid window
starts on 2011-01-05 in all three blocks.
"""
import argparse, sys
import numpy as np
sys.path.insert(0, ".")
from storm_pipeline.data import load_raw, build_features, label_rows, valid_starts

ap = argparse.ArgumentParser(); ap.add_argument("--repo", default=".."); a = ap.parse_args()
df = build_features(*load_raw(a.repo)); obs = df["flux_obs"].values
lead = int(np.argmax(obs > 0)); print(f"leading unobserved hours: {lead}")
for b in "ABC":
    st, _ = valid_starts(label_rows(df.index, b), obs)
    print(f"block {b}: first window starts {df.index[st.min()]} (row {st.min()}); windows touching leading hours: {int((st < lead).sum())}")
    assert (st >= lead).all()
print("OK: the leading fill never enters a forecast window")
