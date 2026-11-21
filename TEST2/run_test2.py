r"""
TEST2: retrain the ensemble on a RANDOMLY sampled EyePACS subset (no sharpness-based selection).

Motivation: the thesis 15k subset was chosen sharpest-first within each grade, so sharpness fell
with grade (median 245 for grade 0 vs 42-49 for grades 3-4) and the models learned to use it.
TEST2 draws the same total (15,000 images) at random within each grade, so image quality is
distributed as in EyePACS itself, then repeats the thesis training recipe unchanged.

Grade counts (GRADE_COUNTS, total 15,000 like the thesis subset):
  0: 5000   one third of the set, the screening-majority class
  1: 2250 \ grades 1 and 2 kept at the full-EyePACS ratio (6,205 : 13,152 = 1 : 2.1);
  2: 4500 /  grade 2 is the referral boundary, so it gets the larger share
  3: 1750 \ close to all usable grade 3/4 images in EyePACS (2,087 and 1,914 before crop
  4: 1500 /  rejects), so the rare, clinically urgent classes are as large as the data allow

Pipeline (each step is cached / resumable; re-running skips finished steps):
  1. sample      random images per grade from all_labels.csv (crop rejects excluded)
  2. preprocess  512x512 crops with the thesis crop_and_resize; an image already cropped in any
                 REUSE_DIRS folder is reused, otherwise it is created in E:\\TEST2\\images_512
  3. split       patient-level 70/15/15 train/val/test, stratified by grade
  4. train       ResNet-50 and ViT-B/16 with the thesis recipe (Table 4.1); resumable
  5. predict     softmax outputs for val and test
  6. evaluate    ensemble weight sweep (val), temperature (val), then on test: per-model and
                 ensemble accuracy/QWK, per-grade metrics, referable and vision-threatening DR,
                 calibration, triage gates (top-grade 0.85; binary gate re-selected on val by the
                 thesis rule; and the thesis 0.94 for reference)

Models and images live on E: (E:\\TEST2). Small result files go to TEST2/results in the repo.

Usage (repo root):   python TEST2/run_test2.py
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "dataset_prep"))

# ---------------- CONFIG ----------------
EYEPACS_DIR = Path(r"E:\EyePACS")
RAW_IMAGES = EYEPACS_DIR / "Images"
ALL_LABELS = EYEPACS_DIR / "all_labels.csv"               # image, level
REJECTS_CSV = EYEPACS_DIR / "crop_rejects.csv"
SHARPNESS_CSV = EYEPACS_DIR / "sharpness_manifest.csv"    # only used to report image quality by grade

WORK = Path(r"E:\TEST2")                                   # everything large for TEST2 goes here
PREP_DIR = WORK / "images_512"
CKPT_DIR = WORK / "checkpoints"
MANIFEST_DIR = WORK / "manifests"
REUSE_DIRS = [                                             # existing 512x512 crops made by the same code
    EYEPACS_DIR / "holdout_preprocessed_512",
    EYEPACS_DIR / "shift_check_512",
    Path(r"D:\Thesis Dataset\final dataset\images"),       # thesis 15k crops (skipped if missing)
]
RESULTS = HERE / "results"                                 # in the repo

GRADE_COUNTS = {0: 5000, 1: 2250, 2: 4500, 3: 1750, 4: 1500}
SPLIT = (0.70, 0.15, 0.15)
SEED = 42

# thesis training recipe (Table 4.1)
RECIPE = {
    "resnet50": dict(lr=1e-4, weight_decay=1e-4, dropout=0.0),
    "vit_b16": dict(lr=3e-5, weight_decay=5e-4, dropout=0.3),
}
INPUT_SIZE = 224
BATCH_SIZE = 32
MAX_EPOCHS = 30
PATIENCE = 7
NUM_WORKERS = 4
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TOP_GRADE_T = 0.85
THESIS_BINARY_T = 0.94
MANDATORY = 2
N_BOOT = 2000
EXTS = (".jpeg", ".jpg", ".png", ".tif", ".tiff")
# -----------------------------------------


def log(msg):
    print(msg, flush=True)


def strip_ext(s):
    return s.astype(str).str.replace(r"\.(png|jpe?g|tiff?)$", "", regex=True, case=False)


# ============================================================ 1. sample
def step_sample():
    out = MANIFEST_DIR / "sample.csv"
    if out.exists():
        log(f"  using existing {out}")
        return pd.read_csv(out)
    lab = pd.read_csv(ALL_LABELS)
    lab["image"] = strip_ext(lab["image"])
    lab = lab[["image", "level"]].dropna()
    lab["level"] = lab["level"].astype(int)
    rej = pd.read_csv(REJECTS_CSV)
    rejected = set(strip_ext(rej["filename" if "filename" in rej.columns else rej.columns[0]]))
    on_disk = {p.stem for p in RAW_IMAGES.iterdir() if p.suffix.lower() in EXTS}
    pool = lab[~lab.image.isin(rejected) & lab.image.isin(on_disk)]
    log(f"  pool: {len(pool)} usable images; grades {pool.level.value_counts().sort_index().to_dict()}")
    rng = np.random.default_rng(SEED)
    parts = []
    for g, n in GRADE_COUNTS.items():
        avail = pool[pool.level == g]
        if len(avail) < n:
            raise SystemExit(f"grade {g}: only {len(avail)} usable images, {n} requested - lower GRADE_COUNTS[{g}]")
        parts.append(avail.iloc[rng.choice(len(avail), n, replace=False)])
    s = pd.concat(parts).reset_index(drop=True)
    s["patient_id"] = s.image.str.split("_").str[0]
    s.to_csv(out, index=False)
    log(f"  sampled {len(s)} images from {s.patient_id.nunique()} patients")
    return s


# ============================================================ 2. preprocess (reuse existing crops)
def step_preprocess(s):
    from crop_and_resize import crop_and_resize
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    reuse = [d for d in REUSE_DIRS if d.exists()]
    paths, made, reused, bad = [], 0, 0, []
    for i, n in enumerate(s.image, 1):
        own = PREP_DIR / (n + ".png")
        hit = own if own.exists() else next((d / (n + ".png") for d in reuse if (d / (n + ".png")).exists()), None)
        if hit is not None:
            paths.append(str(hit)); reused += hit != own
        else:
            src = next((RAW_IMAGES / (n + e) for e in EXTS if (RAW_IMAGES / (n + e)).exists()), None)
            img = None if src is None else __import__("cv2").imread(str(src))
            if img is None:
                paths.append(None); bad.append((n, "unreadable")); continue
            out, valid, reason = crop_and_resize(img)
            if not valid:
                bad.append((n, reason))
            __import__("cv2").imwrite(str(own), out)
            paths.append(str(own)); made += 1
        if i % 1000 == 0:
            log(f"  {i}/{len(s)} images ready")
    s = s.assign(path=paths)
    if bad:
        log(f"  WARNING: {len(bad)} images failed the crop check and are dropped: {bad[:5]}...")
        s = s[~s.image.isin({b[0] for b in bad})]
    log(f"  reused {reused} existing crops, created {made}; {len(s)} images kept")
    return s.reset_index(drop=True)


# ============================================================ 3. patient-level stratified split
def step_split(s):
    files = {k: MANIFEST_DIR / f"{k}_manifest.csv" for k in ["train", "val", "test"]}
    if all(f.exists() for f in files.values()):
        log("  using existing split manifests")
        return {k: pd.read_csv(f) for k, f in files.items()}
    from sklearn.model_selection import StratifiedGroupKFold
    # 20 folds of ~5%: 3 folds val, 3 folds test, 14 folds train -> 70/15/15, grade-stratified,
    # with every patient in exactly one split
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    fold = np.empty(len(s), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(s, s.level, groups=s.patient_id)):
        fold[idx] = k
    split = np.where(fold < 3, "val", np.where(fold < 6, "test", "train"))
    out = {}
    for k in ["train", "val", "test"]:
        out[k] = s[split == k].reset_index(drop=True)
        out[k].to_csv(files[k], index=False)
    for a in ["train", "val"]:
        for b in ["val", "test"]:
            if a != b:
                assert not set(out[a].patient_id) & set(out[b].patient_id), f"patient overlap {a}/{b}"
    tab = pd.DataFrame({k: v.level.value_counts().sort_index() for k, v in out.items()})
    log("  images per grade and split:\n" + tab.to_string())
    return out


# ============================================================ 4. training (thesis recipe)
class ManifestDataset:
    """Module-level so DataLoader workers can pickle it on Windows."""
    def __init__(self, df, transform):
        self.paths, self.labels, self.transform = list(df.path), list(df.level.astype(int)), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        return self.transform(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def transforms_for(train):
    from torchvision import transforms as T
    base = [T.Resize((INPUT_SIZE, INPUT_SIZE))]
    if train:
        base += [T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5), T.RandomRotation(180),
                 T.ColorJitter(brightness=0.1, contrast=0.1)]
    return T.Compose(base + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def build(arch, pretrained=True):
    import torch.nn as nn
    from torchvision import models
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 5)
    else:
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        m.heads.head = nn.Sequential(nn.Dropout(RECIPE[arch]["dropout"]), nn.Linear(m.heads.head.in_features, 5))
    return m


def step_train(arch, splits):
    import torch
    import torch.nn as nn
    from sklearn.metrics import cohen_kappa_score
    from torch.utils.data import DataLoader
    d = CKPT_DIR / arch
    d.mkdir(parents=True, exist_ok=True)
    best_path, last_path, log_path, done = d / "best_model.pt", d / "last_checkpoint.pt", d / "training_log.csv", d / "DONE"
    if done.exists():
        log(f"  {arch}: already trained ({best_path})")
        return best_path
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  {arch}: device {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else "  <-- no GPU!"))
    tr = DataLoader(ManifestDataset(splits["train"], transforms_for(True)), batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0)
    va = DataLoader(ManifestDataset(splits["val"], transforms_for(False)), batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0)
    counts = splits["train"].level.value_counts().reindex(range(5), fill_value=0)
    w = counts.sum() / (5 * counts.replace(0, 1)); w = w / w.mean()
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w.values, dtype=torch.float32, device=dev))
    model = build(arch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=RECIPE[arch]["lr"], weight_decay=RECIPE[arch]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    start, best, since_best, rows = 0, -1.0, 0, []
    if last_path.exists():
        ck = torch.load(last_path, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start, best, since_best = ck["epoch"] + 1, ck["best"], ck["since_best"]
        rows = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
        log(f"  {arch}: resuming at epoch {start + 1}, best val kappa {best:.4f}")

    def run(loader, train):
        model.train(train)
        tot, preds, ys = 0.0, [], []
        with torch.set_grad_enabled(train):
            for b, (x, y) in enumerate(loader, 1):
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                    out = model(x); loss = crit(out, y)
                if train:
                    opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                tot += loss.item() * len(y)
                preds += out.argmax(1).tolist(); ys += y.tolist()
        return tot / len(ys), cohen_kappa_score(ys, preds, weights="quadratic"), float(np.mean(np.array(ys) == np.array(preds)))

    for ep in range(start, MAX_EPOCHS):
        t0 = time.time()
        tl, tk, _ = run(tr, True)
        vl, vk, vacc = run(va, False)
        rows.append({"epoch": ep + 1, "train_loss": tl, "train_kappa": tk, "val_loss": vl, "val_kappa": vk, "val_acc": vacc})
        pd.DataFrame(rows).to_csv(log_path, index=False)
        if vk > best:
            best, since_best = vk, 0
            torch.save(model.state_dict(), best_path)
        else:
            since_best += 1
        torch.save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(), "best": best,
                    "since_best": since_best}, last_path)
        log(f"  {arch} epoch {ep + 1}/{MAX_EPOCHS} ({time.time() - t0:.0f}s): train kappa {tk:.4f} | "
            f"val loss {vl:.4f} kappa {vk:.4f} acc {vacc:.4f}" + ("  *best*" if since_best == 0 else ""))
        if since_best >= PATIENCE:
            log(f"  {arch}: early stop (no val-kappa gain for {PATIENCE} epochs)")
            break
    done.write_text(f"best val kappa {best:.4f}\n")
    shutil.copy(log_path, RESULTS / f"{arch}_training_log.csv")
    return best_path


# ============================================================ 5. predictions
def step_predict(arch, weights, splits):
    out = RESULTS / f"{arch}_predictions.csv"
    if out.exists():
        log(f"  {arch}: using existing {out.name}")
        return pd.read_csv(out)
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build(arch, pretrained=False)
    model.load_state_dict(torch.load(weights, map_location=dev))
    model.to(dev).eval()
    frames = []
    for k in ["val", "test"]:
        dl = DataLoader(ManifestDataset(splits[k], transforms_for(False)), batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS)
        with torch.no_grad():
            P = np.concatenate([F.softmax(model(x.to(dev)), 1).cpu().numpy() for x, _ in dl])
        f = splits[k][["image", "patient_id", "level"]].copy().rename(columns={"level": "true_label"})
        f["split"] = k
        for c in range(5):
            f[f"prob_class{c}"] = P[:, c]
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out, index=False)
    return df


# ============================================================ 6. evaluation
def qwk(y, p):
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y, p, weights="quadratic")


def pboot(df, fn, n=N_BOOT, seed=0):
    """Patient-level bootstrap 95% CI of fn(sub-dataframe)."""
    rng = np.random.default_rng(seed)
    groups = df.groupby("patient_id").indices
    keys = np.array(list(groups))
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        try:
            vals.append(fn(df.iloc[idx]))
        except ValueError:
            pass
    return np.percentile(vals, [2.5, 97.5])


def fit_temperature(P, y):
    from scipy.optimize import minimize_scalar
    z = np.log(np.clip(P, 1e-7, 1))
    def nll(logT):
        s = z / np.exp(logT); s = s - s.max(1, keepdims=True)
        return -(s[np.arange(len(y)), y] - np.log(np.exp(s).sum(1))).mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def calibrate(P, T):
    z = np.log(np.clip(P, 1e-7, 1)) / T
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def ece(conf, correct, bins=15):
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return e


def gate(P, rule, t):
    pred = P.argmax(1)
    score = P.max(1) if rule == "top" else P[:, :2].sum(1)
    return (pred < MANDATORY) & (score >= t)


def gate_row(name, clear, y, P):
    ref = y >= MANDATORY
    miss = clear & ref
    return {"policy": name, "auto_cleared": int(clear.sum()), "missed_referable": int(miss.sum()),
            "miss_rate_%": round(100 * miss.sum() / max(clear.sum(), 1), 2),
            "referral_recall_%": round(100 * (ref & ~clear).sum() / max(ref.sum(), 1), 2),
            "severe_or_PDR_cleared": int((clear & (y >= 3)).sum()),
            "workload_%": round(100 * (~clear).mean(), 2)}


def step_evaluate(preds):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
    rn, vt = preds["resnet50"], preds["vit_b16"]
    assert (rn.image.values == vt.image.values).all()
    base = rn[["image", "patient_id", "true_label", "split"]].copy()
    Prn = rn[[f"prob_class{c}" for c in range(5)]].values
    Pvt = vt[[f"prob_class{c}" for c in range(5)]].values
    val, test = (base.split == "val").values, (base.split == "test").values
    y = base.true_label.values
    out = []

    # --- ensemble weight on val
    sweep = []
    for w in np.round(np.arange(0, 1.0001, 0.05), 2):
        p = (w * Prn + (1 - w) * Pvt)[val].argmax(1)
        sweep.append({"w_resnet": w, "val_QWK": qwk(y[val], p), "val_acc": float((p == y[val]).mean())})
    sweep = pd.DataFrame(sweep)
    w = float(sweep.loc[sweep.val_QWK.idxmax(), "w_resnet"])
    sweep.to_csv(RESULTS / "ensemble_weight_sweep_val.csv", index=False)
    Pens = w * Prn + (1 - w) * Pvt
    T = fit_temperature(Pens[val], y[val])
    Pcal = calibrate(Pens, T)
    out += [f"Ensemble weight (val-selected): w_ResNet = {w:.2f}; temperature T = {T:.4f} (fitted on val)", ""]

    # --- grading
    rows = []
    for split, m in [("val", val), ("test", test)]:
        for name, P in [("ResNet-50", Prn), ("ViT-B/16", Pvt), ("Ensemble", Pcal)]:
            sub = base[m].assign(pred=P[m].argmax(1))
            lo, hi = pboot(sub, lambda d: qwk(d.true_label, d.pred))
            alo, ahi = pboot(sub, lambda d: float((d.true_label == d.pred).mean()))
            rows.append({"split": split, "model": name, "accuracy": f"{(sub.pred == sub.true_label).mean():.3f} [{alo:.3f}, {ahi:.3f}]",
                         "QWK": f"{qwk(sub.true_label, sub.pred):.3f} [{lo:.3f}, {hi:.3f}]"})
    out += ["## Grading (patient-level bootstrap 95% CI)", pd.DataFrame(rows).to_string(index=False), ""]

    yt, pt = y[test], Pcal[test].argmax(1)
    cm = pd.DataFrame(confusion_matrix(yt, pt, labels=range(5)), index=[f"true {g}" for g in range(5)],
                      columns=[f"pred {g}" for g in range(5)])
    pr, rc, f1, sup = precision_recall_fscore_support(yt, pt, labels=range(5), zero_division=0)
    per = pd.DataFrame({"grade": range(5), "precision": pr.round(3), "recall": rc.round(3), "F1": f1.round(3), "n": sup})
    out += ["## Test confusion matrix (ensemble)", cm.to_string(), "", "## Per-grade (test)", per.to_string(index=False), ""]

    # --- binary readouts
    rows = []
    for label, k in [("Referable, grade >= 2", 2), ("Vision-threatening, grade >= 3", 3)]:
        for split, m in [("val", val), ("test", test)]:
            yy, P = y[m] >= k, Pcal[m]
            pp, score = P.argmax(1) >= k, P[:, k:].sum(1)
            tp, tn = (pp & yy).sum(), (~pp & ~yy).sum()
            rows.append({"outcome": label, "split": split, "prevalence_%": round(100 * yy.mean(), 1),
                         "sensitivity_%": round(100 * tp / yy.sum(), 1), "specificity_%": round(100 * tn / (~yy).sum(), 1),
                         "PPV_%": round(100 * tp / max(pp.sum(), 1), 1), "NPV_%": round(100 * tn / max((~pp).sum(), 1), 1),
                         "AUC": round(roc_auc_score(yy, score), 4)})
    out += ["## Binary readouts (argmax operating point; AUC of P(grade >= k))", pd.DataFrame(rows).to_string(index=False), ""]

    # --- calibration
    rows = []
    for name, P in [("uncalibrated", Pens), (f"T = {T:.3f}", Pcal)]:
        Pt = P[test]
        rows.append({"probabilities": name, "ECE_5class": round(ece(Pt.max(1), (Pt.argmax(1) == yt).astype(float)), 4),
                     "ECE_binary_P_nr": round(ece(Pt[:, :2].sum(1), (yt < 2).astype(float)), 4)})
    out += ["## Calibration (test)", pd.DataFrame(rows).to_string(index=False), ""]

    # --- gates: top-grade 0.85; binary re-selected on val with the thesis rule; thesis 0.94
    yv, Pv = y[val], Pcal[val]
    m_top = int((gate(Pv, "top", TOP_GRADE_T) & (yv >= MANDATORY)).sum())
    grid = np.round(np.arange(0.800, 0.9901, 0.005), 3)
    t_star = next((t for t in grid if (gate(Pv, "bin", t) & (yv >= MANDATORY)).sum() <= m_top), grid[-1])
    out += [f"Binary-gate threshold re-selected on val (lowest t with val misses <= top-grade val misses = {m_top}): t* = {t_star:.3f}", ""]
    rows = []
    for split, m in [("val", val), ("test", test)]:
        P, yy = Pcal[m], y[m]
        for name, rule, t in [(f"Top-grade: max p >= {TOP_GRADE_T}", "top", TOP_GRADE_T),
                              (f"P_nr >= {TOP_GRADE_T} (same threshold)", "bin", TOP_GRADE_T),
                              (f"Binary: P_nr >= t* = {t_star:.3f}", "bin", t_star),
                              (f"Binary: P_nr >= {THESIS_BINARY_T} (thesis value)", "bin", THESIS_BINARY_T)]:
            rows.append({"split": split, **gate_row(name, gate(P, rule, t), yy, P)})
    out += ["## Triage gates (no verification)", pd.DataFrame(rows).to_string(index=False), ""]

    # --- image quality by grade (should no longer fall with grade)
    if SHARPNESS_CSV.exists():
        sh = pd.read_csv(SHARPNESS_CSV)
        sh["image"] = strip_ext(sh["image"])
        q = base.image.map(dict(zip(sh.image, sh.quality_score)))
        t = pd.DataFrame({"TEST2 median sharpness": q.groupby(base.true_label).median().round(1),
                          "thesis 15k median (for reference)": pd.Series({0: 244.9, 1: 95.0, 2: 111.6, 3: 49.3, 4: 41.9})})
        out += ["## Image sharpness by grade (val + test images)", t.to_string(), ""]

    pd.DataFrame({"image": base.image, "patient_id": base.patient_id, "split": base.split, "true_label": y,
                  **{f"ensemble_prob_class{c}": Pcal[:, c] for c in range(5)}}).to_csv(RESULTS / "ensemble_calibrated_predictions.csv", index=False)
    thesis = ("Thesis 15k (sharpness-selected), test: ensemble accuracy 0.727, QWK 0.872; referable sens 86.6%, "
              "spec 88.1%, AUC 0.948; top-grade gate 396 cleared / 16 missed; binary 0.94 gate 423 / 13.")
    text = "\n".join(["# TEST2: ensemble retrained on a randomly sampled 15k EyePACS subset", "", thesis, ""] + out)
    (RESULTS / "TEST2_SUMMARY.md").write_text(text, encoding="utf-8")
    log(text)


def main():
    for d in [WORK, PREP_DIR, CKPT_DIR, MANIFEST_DIR, RESULTS]:
        d.mkdir(parents=True, exist_ok=True)
    log("=== 1. sample"); s = step_sample()
    log("=== 2. preprocess"); s = step_preprocess(s)
    log("=== 3. split"); splits = step_split(s)
    pd.concat([v.assign(split=k) for k, v in splits.items()])[["image", "patient_id", "level", "split"]] \
        .to_csv(RESULTS / "TEST2_split.csv", index=False)
    preds = {}
    for arch in ["resnet50", "vit_b16"]:
        log(f"=== 4. train {arch}"); w = step_train(arch, splits)
        log(f"=== 5. predict {arch}"); preds[arch] = step_predict(arch, w, splits)
    log("=== 6. evaluate"); step_evaluate(preds)


if __name__ == "__main__":
    main()
