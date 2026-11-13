#!/usr/bin/env python
"""Write the Kaggle scripts of the journal campaign (one per Kaggle account).

  python make_journal_kaggle.py <commit>              ->  kaggle/journal_account{1,2,3}.py   (campaign 1)
  python make_journal_kaggle.py <commit> --campaign 2 ->  kaggle/journal2_account{1,2}.py   (LSTM-encoder attribution)
  python make_journal_kaggle.py <commit> --campaign 3 ->  kaggle/journal3_account{1,2,3}.py (final: campaign 2 + physfix)
  python make_journal_kaggle.py <commit> --campaign 4 ->  kaggle/journal4_account1.py         (residual-LSTM width)

Each account trains a group of model families on all three test blocks (A, B, C; 10 seeds; horizons 1, 6, 12, 24 h)
with the noise and driver probes, then uses its block-A checkpoints for cross-satellite transfer to GOES-16, GOES-18
and GSAT-19 GRASP (zero-shot for every family; heads-only and full fine-tuning for the STORM-PhysNet variants).
The nine conference families were already run with this pipeline (journal24_[ABC].zip, journal_transfer.zip).
A session stops starting new work after SESSION_HOURS and resumes in the next session from its own output.
"""
import sys
from pathlib import Path

REPO_URL = "https://github.com/bnsama29-cloud/STORM-PhysNet-L2"
CAMPAIGN1 = {   # account -> (families, families fine-tuned in the transfer step)
    1: ("storm_cathode,storm_cathode_spec,storm_radiotrophic", "storm_cathode,storm_cathode_spec,storm_radiotrophic"),
    2: ("storm_no_physics,storm_magnetopause,storm_lstm_backbone", "storm_no_physics,storm_magnetopause,storm_lstm_backbone"),
    3: ("transformer,tf_ffn256,mlp,cnn", "transformer"),
}
CAMPAIGN2 = {   # why does the LSTM encoder help? residual LSTM control and one-component ablations of the LSTM variant
    1: ("lstm_res,storm_lstm_no_gate", "lstm_res"),
    2: ("storm_lstm_no_delay,storm_lstm_no_physics", ""),
}
CAMPAIGN3 = {   # final campaign: campaign 2 (if not yet run) + the physics loss evaluated in physical units ("physfix")
    1: ("lstm_res,storm_lstm_no_gate", "lstm_res"),
    2: ("storm_lstm_no_delay,storm_lstm_no_physics", ""),
    3: ("storm_physfix,storm_lstm_physfix", "storm_physfix,storm_lstm_physfix"),
}
CAMPAIGN4 = {   # width sensitivity of the residual LSTM (the recommended model); account 2 runs the classical baseline
    1: ("lstm_res_w64,lstm_res_w256", ""),
    2: ("gbm_res", "CLASSICAL"),
}

TEMPLATE = r'''# STORM-PhysNet journal campaign __CAMP__, account __ACC__: families __FAMS__
# Blocks A, B and C (10 seeds each, horizons 1/6/12/24 h, noise and driver probes), then cross-satellite transfer
# (GOES-16, GOES-18, GSAT-19 GRASP) from the block-A models.
# HOW TO USE: Kaggle -> Create -> New Notebook -> paste this whole file into the first cell.
# Settings (right sidebar): Accelerator = GPU T4 x2, Internet = ON (needs a phone-verified account).
# Then Save Version -> Save & Run All (Commit). Expected time: about __HOURS__ hours.
# The last log line says "ALL WORK DONE" (download __ZIP__ and send it) or "NOT FINISHED".
# TO CONTINUE ANOTHER DAY: open the notebook -> Add Input -> Notebook Output Files -> pick this notebook (its latest
# version) -> Save Version -> Save & Run All. Finished runs are skipped; nothing is repeated.
SESSION_HOURS = 10          # keep <= 10.5 (Kaggle stops a session at 12 h)

import os, subprocess, shutil, glob, time, zipfile
T0 = time.time()
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-1500:]); return r
sh("pip -q install cdflib tensorboard tqdm tabulate 2>&1 | tail -1")
REPO = "/kaggle/working/STORM-PhysNet-L2"
if not os.path.exists(REPO):
    sh("cd /kaggle/working && git clone -q __REPO_URL__ STORM-PhysNet-L2")
sh(f"cd {REPO} && git fetch -q origin && git checkout -q __COMMIT__")
assert sh(f"cd {REPO} && git rev-parse HEAD").stdout.strip() == "__COMMIT__", "wrong code version"
PIPE = REPO + "/pipeline"
import torch
NGPU = torch.cuda.device_count(); print("GPUs:", NGPU, [torch.cuda.get_device_name(i) for i in range(NGPU)])
assert NGPU >= 1, "No GPU: set Accelerator = GPU T4 x2 in the notebook settings"
SEED_SPLITS = ["42-46", "47-51"] if NGPU >= 2 else ["42-51"]
FAMS, TFAMS = "__FAMS__", "__TFAMS__"
NFAM = len(FAMS.split(","))
W = "/kaggle/working"

# ---- resume: copy finished work from a previous version of this notebook, if attached as an Input ----
for name in ("j2_A", "j2_B", "j2_C", "tj2"):
    for src in glob.glob(f"/kaggle/input/**/{name}", recursive=True):
        shutil.copytree(src, f"{W}/{name}", dirs_exist_ok=True); print("resumed", name, "from", src)
for f in glob.glob(f"{W}/*/*.tmp.*"): os.remove(f)

def budget():
    return max(0.0, SESSION_HOURS - (time.time() - T0) / 3600 - 0.4)   # 0.4 h margin for the last run started

def run_parallel(cmd_for_split, tag):
    procs = []
    for i, seeds in enumerate(SEED_SPLITS):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        log = open(f"{W}/{tag}_gpu{i}.log", "a")
        procs.append(subprocess.Popen(cmd_for_split(seeds), shell=True, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=PIPE))
        time.sleep(20)
    while any(p.poll() is None for p in procs):
        time.sleep(600)
        for i in range(len(procs)):
            lines = open(f"{W}/{tag}_gpu{i}.log").read().splitlines()
            print(time.strftime("%H:%M"), f"{tag} gpu{i}:", lines[-1][:150] if lines else "")
    codes = [p.returncode for p in procs]; print(tag, "exit codes", codes)
    for i in range(len(procs)):
        print(f"--- last lines {tag} gpu{i}"); print("\n".join(open(f"{W}/{tag}_gpu{i}.log").read().splitlines()[-6:]))
    assert all(c == 0 for c in codes), "a process failed: read the log lines above"

def n_preds(b): return len(glob.glob(f"{W}/j2_{b}/preds_*.npz"))

# ---- 1. train every family of this account on blocks A, B, C ----
for b in ("A", "B", "C"):
    if n_preds(b) >= 10 * NFAM: print(f"block {b}: done"); continue
    if budget() <= 0: break
    run_parallel(lambda seeds, b=b: f"python -u train_block.py --repo {REPO} --block {b} --out {W}/j2_{b} --horizons 1,6,12,24 "
                 f"--families {FAMS} --noise_families {FAMS} --probes --seeds {seeds} --keep_ckpt A "
                 f"--stop_after_hours {budget():.2f}", f"train_{b}")
    print(f"block {b}: {n_preds(b)} of {10 * NFAM} runs")

# ---- 2. cross-satellite transfer from the block-A checkpoints ----
def n_tr(tag): return len(glob.glob(f"{W}/tj2/transfer_preds{tag}_s*.npz"))
if n_preds("A") >= 10 * NFAM:
    for tag, fams, variants in (("_zs", FAMS, "zeroshot"), ("_ft", TFAMS, "heads,full")):
        if not fams or n_tr(tag) >= 10 or budget() <= 0: continue
        run_parallel(lambda seeds, fams=fams, variants=variants, tag=tag:
                     f"python -u transfer_journal.py --repo {REPO} --ckpt_dir {W}/j2_A/ckpt_A --meta {W}/j2_A/meta_A.npz "
                     f"--out {W}/tj2 --seeds {seeds} --horizons 1,6,12,24 --families {fams} --variants {variants} "
                     f"--tag {tag} --stop_after_hours {max(0.0, budget() - 0.6):.2f}", f"transfer{tag}")

# ---- 3. zip everything except checkpoints (download __ZIP__ from the Output tab) ----
name = f"{W}/__ZIP__"
with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as zf:
    for d in ("j2_A", "j2_B", "j2_C", "tj2"):
        for f in glob.glob(f"{W}/{d}/*"):
            if os.path.isfile(f) and not f.endswith(".pt") and ".tmp." not in f: zf.write(f, os.path.relpath(f, W))
    for f in glob.glob(f"{W}/*.log"): zf.write(f, "logs/" + os.path.basename(f))
done = [n_preds(b) >= 10 * NFAM for b in "ABC"] + [n_tr("_zs") >= 10, (not TFAMS) or n_tr("_ft") >= 10]
print(round(os.path.getsize(name) / 1e6, 1), "MB ->", name, f"| session time {(time.time() - T0) / 3600:.1f} h")
print(f"runs: A {n_preds('A')}, B {n_preds('B')}, C {n_preds('C')} of {10 * NFAM} each; transfer seeds: zero-shot {n_tr('_zs')}/10, fine-tune {n_tr('_ft')}/10")
print("ALL WORK DONE: send", name) if all(done) else print("NOT FINISHED: add this notebook's output as an Input and Save & Run All again.")
'''


CLASSICAL_TEMPLATE = r'''# STORM-PhysNet journal campaign 4, account __ACC__: classical baseline (gradient-boosted trees, LightGBM)
# Blocks A, B and C, ten seeds, the same test windows and robustness probes as the deep models. CPU only.
# HOW TO USE: Kaggle -> Create -> New Notebook -> paste this whole file into the first cell.
# Settings (right sidebar): Accelerator = None (CPU is enough; GPU T4 x2 also works), Internet = ON.
# Then Save Version -> Save & Run All (Commit). Expected time: about __HOURS__ hours.
# The last log line says "ALL WORK DONE" (download __ZIP__ and send it) or "NOT FINISHED".
# TO CONTINUE ANOTHER DAY: Add Input -> Notebook Output Files -> this notebook -> Save Version -> Save & Run All.
SESSION_HOURS = 10

import os, subprocess, shutil, glob, time, zipfile
T0 = time.time()
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-1500:]); return r
sh("pip -q install cdflib lightgbm==4.7.0 2>&1 | tail -1")
REPO = "/kaggle/working/STORM-PhysNet-L2"
if not os.path.exists(REPO):
    sh("cd /kaggle/working && git clone -q __REPO_URL__ STORM-PhysNet-L2")
sh(f"cd {REPO} && git fetch -q origin && git checkout -q __COMMIT__")
assert sh(f"cd {REPO} && git rev-parse HEAD").stdout.strip() == "__COMMIT__", "wrong code version"
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

name = f"{W}/__ZIP__"
with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as zf:
    for d in ("j2_A", "j2_B", "j2_C"):
        for f in glob.glob(f"{W}/{d}/*"):
            if os.path.isfile(f) and ".tmp." not in f: zf.write(f, os.path.relpath(f, W))
print(round(os.path.getsize(name) / 1e6, 1), "MB ->", name, f"| session time {(time.time() - T0) / 3600:.1f} h")
print("ALL WORK DONE: send", name) if all(n_preds(b) >= 10 for b in "ABC") else print("NOT FINISHED: add this notebook's output as an Input and Save & Run All again.")
'''


def main(commit, campaign=1):
    out = Path(__file__).resolve().parent / "kaggle"; out.mkdir(exist_ok=True)
    groups = {1: CAMPAIGN1, 2: CAMPAIGN2, 3: CAMPAIGN3, 4: CAMPAIGN4}[campaign]
    hours = {1: {1: 7, 2: 7, 3: 5}, 2: {1: 4, 2: 3}, 3: {1: 4, 2: 3, 3: 7}, 4: {1: 3, 2: 4}}[campaign]
    for acc, (fams, tfams) in groups.items():
        stem = f"journal_account{acc}" if campaign == 1 else f"journal{campaign}_account{acc}"
        zipname = f"journal_l2_account{acc}.zip" if campaign == 1 else f"journal_l2_c{campaign}_account{acc}.zip"
        s = ((CLASSICAL_TEMPLATE if tfams == "CLASSICAL" else TEMPLATE).replace("__ACC__", str(acc)).replace("__FAMS__", fams).replace("__TFAMS__", tfams)
             .replace("__COMMIT__", commit).replace("__REPO_URL__", REPO_URL).replace("__HOURS__", str(hours[acc]))
             .replace("__CAMP__", str(campaign)).replace("__ZIP__", zipname))
        left = [k for k in ("__ACC__", "__FAMS__", "__TFAMS__", "__COMMIT__", "__REPO_URL__", "__HOURS__", "__CAMP__", "__ZIP__") if k in s]
        assert not left, f"unfilled placeholders {left}"
        p = out / f"{stem}.py"; p.write_text(s); print("wrote", p)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("commit"); ap.add_argument("--campaign", type=int, default=1)
    a = ap.parse_args(); main(a.commit, a.campaign)
