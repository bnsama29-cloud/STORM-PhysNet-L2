# STORM-PhysNet journal campaign 4, account 2: classical baseline (gradient-boosted trees, LightGBM)
# Blocks A, B and C, ten seeds, the same test windows and robustness probes as the deep models. CPU only.
# HOW TO USE: Kaggle -> Create -> New Notebook -> paste this whole file into the first cell.
# Settings (right sidebar): Accelerator = None (CPU is enough; GPU T4 x2 also works), Internet = ON.
# Then Save Version -> Save & Run All (Commit). Expected time: about 4 hours.
# The last log line says "ALL WORK DONE" (download journal_l2_c4_account2.zip and send it) or "NOT FINISHED".
# TO CONTINUE ANOTHER DAY: Add Input -> Notebook Output Files -> this notebook -> Save Version -> Save & Run All.
SESSION_HOURS = 10

import os, subprocess, shutil, glob, time, zipfile
T0 = time.time()
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-1500:]); return r
sh("pip -q install cdflib lightgbm==4.7.0 2>&1 | tail -1")
REPO = "/kaggle/working/STORM-PhysNet-L2"
if not os.path.exists(REPO):
    sh("cd /kaggle/working && git clone -q https://github.com/bnsama29-cloud/STORM-PhysNet-L2 STORM-PhysNet-L2")
sh(f"cd {REPO} && git fetch -q origin && git checkout -q f46bb27db1f3683be23bd50f2729bdd6d2695c79")
assert sh(f"cd {REPO} && git rev-parse HEAD").stdout.strip() == "f46bb27db1f3683be23bd50f2729bdd6d2695c79", "wrong code version"
PIPE = REPO + "/pipeline"; W = "/kaggle/working"; print("CPUs:", os.cpu_count())
for name in ("j2_A", "j2_B", "j2_C"):
    for src in glob.glob(f"/kaggle/input/**/{name}", recursive=True):
        shutil.copytree(src, f"{W}/{name}", dirs_exist_ok=True); print("resumed", name, "from", src)
for f in glob.glob(f"{W}/*/*.tmp.npz"): os.remove(f)

def n_preds(b): return len(glob.glob(f"{W}/j2_{b}/preds_{b}_gbm_res_s*.npz"))
for b in ("A", "B", "C"):
    if n_preds(b) >= 10: print(f"block {b}: done"); continue
    left = SESSION_HOURS - (time.time() - T0) / 3600 - 0.3
    if left <= 0: break
    r = subprocess.run(f"cd {PIPE} && python -u classical_baselines.py --repo {REPO} --block {b} --out {W}/j2_{b} "
                       f"--horizons 1,6,12,24 --seeds 42-51 --families gbm_res --stop_after_hours {left:.2f}",
                       shell=True, capture_output=True, text=True)
    print((r.stdout + r.stderr)[-3000:]); assert r.returncode == 0, "classical_baselines.py failed: read the lines above"
    print(f"block {b}: {n_preds(b)} of 10 seeds")

name = f"{W}/journal_l2_c4_account2.zip"
with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as zf:
    for d in ("j2_A", "j2_B", "j2_C"):
        for f in glob.glob(f"{W}/{d}/*"):
            if os.path.isfile(f) and ".tmp." not in f: zf.write(f, os.path.relpath(f, W))
print(round(os.path.getsize(name) / 1e6, 1), "MB ->", name, f"| session time {(time.time() - T0) / 3600:.1f} h")
print("ALL WORK DONE: send", name) if all(n_preds(b) >= 10 for b in "ABC") else print("NOT FINISHED: add this notebook's output as an Input and Save & Run All again.")
