# STORM-PhysNet journal campaign 3, account 2: families storm_lstm_no_delay,storm_lstm_no_physics
# Blocks A, B and C (10 seeds each, horizons 1/6/12/24 h, noise and driver probes), then cross-satellite transfer
# (GOES-16, GOES-18, GSAT-19 GRASP) from the block-A models.
# HOW TO USE: Kaggle -> Create -> New Notebook -> paste this whole file into the first cell.
# Settings (right sidebar): Accelerator = GPU T4 x2, Internet = ON (needs a phone-verified account).
# Then Save Version -> Save & Run All (Commit). Expected time: about 3 hours.
# The last log line says "ALL WORK DONE" (download journal_l2_c3_account2.zip and send it) or "NOT FINISHED".
# TO CONTINUE ANOTHER DAY: open the notebook -> Add Input -> Notebook Output Files -> pick this notebook (its latest
# version) -> Save Version -> Save & Run All. Finished runs are skipped; nothing is repeated.
SESSION_HOURS = 10          # keep <= 10.5 (Kaggle stops a session at 12 h)

import os, subprocess, shutil, glob, time, zipfile
T0 = time.time()
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-1500:]); return r
sh("pip -q install cdflib tensorboard tqdm tabulate 2>&1 | tail -1")
REPO = "/kaggle/working/STORM-PhysNet-L2"
if not os.path.exists(REPO):
    sh("cd /kaggle/working && git clone -q https://github.com/bnsama29-cloud/STORM-PhysNet-L2 STORM-PhysNet-L2")
sh(f"cd {REPO} && git fetch -q origin && git checkout -q 22c1d242604620e08a2b3aa63f55b9fd7490c380")
assert sh(f"cd {REPO} && git rev-parse HEAD").stdout.strip() == "22c1d242604620e08a2b3aa63f55b9fd7490c380", "wrong code version"
PIPE = REPO + "/pipeline"
import torch
NGPU = torch.cuda.device_count(); print("GPUs:", NGPU, [torch.cuda.get_device_name(i) for i in range(NGPU)])
assert NGPU >= 1, "No GPU: set Accelerator = GPU T4 x2 in the notebook settings"
SEED_SPLITS = ["42-46", "47-51"] if NGPU >= 2 else ["42-51"]
FAMS, TFAMS = "storm_lstm_no_delay,storm_lstm_no_physics", ""
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

# ---- 3. zip everything except checkpoints (download journal_l2_c3_account2.zip from the Output tab) ----
name = f"{W}/journal_l2_c3_account2.zip"
with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as zf:
    for d in ("j2_A", "j2_B", "j2_C", "tj2"):
        for f in glob.glob(f"{W}/{d}/*"):
            if os.path.isfile(f) and not f.endswith(".pt") and ".tmp." not in f: zf.write(f, os.path.relpath(f, W))
    for f in glob.glob(f"{W}/*.log"): zf.write(f, "logs/" + os.path.basename(f))
done = [n_preds(b) >= 10 * NFAM for b in "ABC"] + [n_tr("_zs") >= 10, (not TFAMS) or n_tr("_ft") >= 10]
print(round(os.path.getsize(name) / 1e6, 1), "MB ->", name, f"| session time {(time.time() - T0) / 3600:.1f} h")
print(f"runs: A {n_preds('A')}, B {n_preds('B')}, C {n_preds('C')} of {10 * NFAM} each; transfer seeds: zero-shot {n_tr('_zs')}/10, fine-tune {n_tr('_ft')}/10")
print("ALL WORK DONE: send", name) if all(done) else print("NOT FINISHED: add this notebook's output as an Input and Save & Run All again.")
