"""
Held-out EyePACS evaluation of the frozen system on the TOP_X sharpest EyePACS images that
were not used anywhere in the 15k subset.

Selection: rank sharpness_manifest.csv by quality_score (descending) and take the first TOP_X
images that are
  - not in selected_15k_manifest.csv,
  - not in crop_rejects.csv,
  - from a patient with no image in the 15k (EXCLUDE_USED_PATIENTS; prevents patient leakage).

Frozen (selected on the original val split, never re-tuned): ensemble weights 0.6/0.4,
temperature T, binary gate P(grade 0+1) >= 0.94, top-grade gate 0.85 (baseline),
lesion rule MA_count > 11, fellow-eye rule (other eye predicted >= 2).

Caching (so raising TOP_X from 10k to 20k only processes the new images):
  - preprocessed 512x512 crops : PREP_DIR on E: (an image is skipped if its file exists)
  - ensemble probabilities     : CACHE_DIR/predictions_cache.csv (one row per image, appended)
  - lesion features            : CACHE_DIR/lesions_cache.csv     (one row per image, appended)
Results for each run go to agentic/results/eyepacs_holdout_<TOP_X>/, so runs never overwrite.

Usage (repo root):  python agentic/eval_eyepacs_holdout.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
for sub in ["dataset_prep", "diagnosis_agent", "idrid", "agentic"]:
    sys.path.insert(0, str(ROOT / sub))

# ---------------- CONFIG ----------------
TOP_X = 10000                                    # <-- number of held-out images; raise later if wanted

EYEPACS_DIR = Path(r"E:\EyePACS")
RAW_IMAGES = EYEPACS_DIR / "Images"              # original (uncropped) EyePACS images
SHARPNESS_CSV = EYEPACS_DIR / "sharpness_manifest.csv"
USED_CSV = EYEPACS_DIR / "selected_15k_manifest.csv"
REJECTS_CSV = EYEPACS_DIR / "crop_rejects.csv"
EXCLUDE_USED_PATIENTS = True

PREP_DIR = EYEPACS_DIR / "holdout_preprocessed_512"   # outside the repo
CACHE_DIR = EYEPACS_DIR / "holdout_cache"             # per-image model outputs, outside the repo
OUT_DIR = ROOT / "agentic" / "results" / f"eyepacs_holdout_{TOP_X}"

RESNET_WEIGHTS = ROOT / "diagnosis_agent" / "resnet50" / "best_model.pt"
VIT_WEIGHTS = ROOT / "diagnosis_agent" / "vitb" / "run2" / "best_model.pt"
RESNET_W, VIT_W = 0.6, 0.4
BATCH_SIZE = 32
NUM_WORKERS = 4             # image-loading processes; set to 0 if the DataLoader errors on Windows

GATE_T = 0.94
V1_T = 0.85
LESION_RULE = ("MA_count", 11.0)
MANDATORY = 2
SWEEP = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]
N_BOOT = 2000
EXTS = (".jpeg", ".jpg", ".png", ".tif", ".tiff")
# -----------------------------------------

TEMPERATURE = json.load(open(ROOT / "referral_agent" / "temperature_scaling.json"))["temperature"]
PRED_CACHE = CACHE_DIR / "predictions_cache.csv"
LES_CACHE = CACHE_DIR / "lesions_cache.csv"
CROP_LOG = CACHE_DIR / "crop_log.csv"


def strip_ext(s):
    return s.astype(str).str.replace(r"\.(png|jpe?g|tiff?)$", "", regex=True, case=False)


def read_cache(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["image"])


def append_cache(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


# ============================================================ 1. select
def select():
    sh = pd.read_csv(SHARPNESS_CSV)
    sh["image"] = strip_ext(sh["image"])
    if "image_exists" in sh.columns:
        sh = sh[sh["image_exists"].astype(bool)]
    used = set(strip_ext(pd.read_csv(USED_CSV)["image"]))
    used_pids = {i.split("_")[0] for i in used}
    rej = pd.read_csv(REJECTS_CSV)
    rejected = set(strip_ext(rej["filename" if "filename" in rej.columns else rej.columns[0]]))

    pool = sh[~sh.image.isin(used) & ~sh.image.isin(rejected)].copy()
    pool["patient_id"] = pool.image.str.split("_").str[0]
    n_before = len(pool)
    if EXCLUDE_USED_PATIENTS:
        pool = pool[~pool.patient_id.isin(used_pids)]
    pool = pool.sort_values("quality_score", ascending=False)
    print(f"  pool: {n_before} unused, non-rejected images; {len(pool)} after removing patients in the 15k")
    if len(pool) < TOP_X:
        print(f"  WARNING: only {len(pool)} eligible images, fewer than TOP_X = {TOP_X}")
    sel = pool.head(TOP_X).rename(columns={"level": "true_label"}).reset_index(drop=True)
    sel = sel[["image", "patient_id", "true_label", "quality_score"]]
    sel.to_csv(OUT_DIR / "holdout_manifest.csv", index=False)
    print(f"  selected {len(sel)} images / {sel.patient_id.nunique()} patients; "
          f"grades {sel.true_label.value_counts().sort_index().to_dict()}; "
          f"quality_score range {sel.quality_score.min():.1f}-{sel.quality_score.max():.1f}")
    return sel


# ============================================================ 2. preprocess (skips existing files)
def preprocess(sel):
    from crop_and_resize import crop_and_resize
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    todo = [n for n in sel.image if not (PREP_DIR / (n + ".png")).exists()]
    print(f"  {len(sel) - len(todo)} already preprocessed, {len(todo)} to do")
    log = []
    for i, n in enumerate(todo, 1):
        src = next((RAW_IMAGES / (n + e) for e in EXTS if (RAW_IMAGES / (n + e)).exists()), None)
        img = cv2.imread(str(src)) if src else None
        if img is None:
            log.append({"image": n, "crop_valid": False, "crop_reason": "missing_or_unreadable"})
            continue
        out, valid, reason = crop_and_resize(img)
        cv2.imwrite(str(PREP_DIR / (n + ".png")), out)
        log.append({"image": n, "crop_valid": bool(valid), "crop_reason": reason})
        if i % 500 == 0 or i == len(todo):
            print(f"  preprocessed {i}/{len(todo)}", flush=True)
    append_cache(CROP_LOG, log)


# ============================================================ 3. ensemble (cached per image)
def predict(sel):
    done = set(read_cache(PRED_CACHE).image)
    todo = [n for n in sel.image if n not in done and (PREP_DIR / (n + ".png")).exists()]
    print(f"  {len(sel) - len(todo)} predictions cached, {len(todo)} to run")
    if not todo:
        return
    print("  importing PyTorch ...", flush=True)
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from ensemble_eval import build_resnet50, build_vit_b, get_eval_transform, load_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else
                                   "  <-- WARNING: no GPU visible to PyTorch, this will be slow"))

    class DS(Dataset):
        def __init__(self, tf): self.tf = tf
        def __len__(self): return len(todo)
        def __getitem__(self, i): return self.tf(Image.open(PREP_DIR / (todo[i] + ".png")).convert("RGB"))

    @torch.no_grad()
    def probs(model, name):
        dl = DataLoader(DS(get_eval_transform(224)), batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
        out, n_batches = [], len(dl)
        for b, x in enumerate(dl, 1):
            out.append(F.softmax(model(x.to(device, non_blocking=True)), 1).cpu().numpy())
            if b % 20 == 0 or b == n_batches:
                print(f"  {name}: batch {b}/{n_batches}", flush=True)
        return np.concatenate(out)

    print("  loading ResNet-50 ...", flush=True)
    rn = load_model(build_resnet50(5), RESNET_WEIGHTS, device); p_rn = probs(rn, "ResNet-50"); del rn
    print("  loading ViT-B/16 ...", flush=True)
    vit = load_model(build_vit_b(5), VIT_WEIGHTS, device); p_vit = probs(vit, "ViT-B/16")
    out = pd.DataFrame({"image": todo})
    for name, P in [("resnet", p_rn), ("vit", p_vit), ("ensemble", RESNET_W * p_rn + VIT_W * p_vit)]:
        for k in range(5):
            out[f"{name}_prob_class{k}"] = P[:, k]
    append_cache(PRED_CACHE, out.to_dict("records"))


# ============================================================ 4. lesions (cached per image)
def lesions(sel):
    done = set(read_cache(LES_CACHE).image)
    todo = [n for n in sel.image if n not in done and (PREP_DIR / (n + ".png")).exists()]
    print(f"  {len(sel) - len(todo)} lesion results cached, {len(todo)} to run")
    if not todo:
        return
    import segmentation_models_pytorch as smp
    import torch
    from eyepacs_lesion_analysis import MODEL_PATH, THRESHOLDS_PATH, analyse_image, retina_mask, run_model
    from train_lesion_unet import ENCODER, LESIONS, preprocess as ben_graham
    thresholds = json.load(open(THRESHOLDS_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(ENCODER, encoder_weights=None, in_channels=3, classes=len(LESIONS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    for start in range(0, len(todo), 8):
        names = todo[start:start + 8]
        imgs = [cv2.imread(str(PREP_DIR / (n + ".png"))) for n in names]
        P = run_model(model, [ben_graham(im) for im in imgs], device)
        rows = []
        for k, (n, im) in enumerate(zip(names, imgs)):
            feat, _ = analyse_image(P[k], retina_mask(im), thresholds)
            rows.append({"image": n, **feat})
        append_cache(LES_CACHE, rows)            # written per batch, so an interrupted run resumes
        if (start // 8) % 100 == 0:
            print(f"  lesions {min(start + 8, len(todo))}/{len(todo)}", flush=True)


# ============================================================ 5. analyse
def calibrate(P, t=TEMPERATURE):
    z = np.log(P + 1e-12) / t
    c = np.exp(z - z.max(1, keepdims=True))
    return c / c.sum(1, keepdims=True)


def build_df(sel):
    pred = read_cache(PRED_CACHE).drop_duplicates("image")
    les = read_cache(LES_CACHE).drop_duplicates("image")
    df = sel.merge(pred, on="image").merge(les, on="image")
    if CROP_LOG.exists():
        log = read_cache(CROP_LOG).drop_duplicates("image", keep="last")
        bad = set(log.loc[~log.crop_valid.astype(bool), "image"])
        if bad & set(df.image):
            print(f"  excluding {len(bad & set(df.image))} images flagged by the crop check")
        df = df[~df.image.isin(bad)]
    df = df.reset_index(drop=True)
    print(f"  analysing {len(df)} of {len(sel)} selected images")
    cal = calibrate(df[[f"ensemble_prob_class{k}" for k in range(5)]].values)
    df["pred"], df["p_nr"], df["p_top"] = cal.argmax(1), cal[:, :2].sum(1), cal.max(1)
    side = df.image.str.extract(r"_(left|right)$")[0]
    other = df.image.str.replace(r"_(left|right)$", "", regex=True) + "_" + side.map({"left": "right", "right": "left"})
    df["fellow_pred"] = other.map(dict(zip(df.image, df.pred)))
    keep = ["image", "patient_id", "true_label", "quality_score", "pred", "p_nr", "p_top", "fellow_pred",
            LESION_RULE[0]] + [f"ensemble_prob_class{k}" for k in range(5)]
    df[keep].to_csv(OUT_DIR / "holdout_per_image.csv", index=False)
    return df


def clears(df, rule, t, lesion=False, fellow=False):
    score = df.p_top if rule == "v1" else df.p_nr
    cand = (df.pred < MANDATORY) & (score >= t)
    over = pd.Series(False, index=df.index)
    if lesion:
        over |= cand & (df[LESION_RULE[0]] > LESION_RULE[1])
    if fellow:
        over |= cand & (df.fellow_pred >= MANDATORY)
    return cand & ~over, over


def matched_t(df, n):
    s = df.loc[df.pred < MANDATORY, "p_nr"].sort_values(ascending=False).values
    return 1.0 if n <= 0 else (float(np.nextafter(s[n - 1], 1.0)) if n < len(s) else 0.0)


def row(df, name, c, over):
    ref = (df.true_label >= MANDATORY).values
    c = c.values
    return {"policy": name, "auto_cleared": int(c.sum()), "missed_referable": int((c & ref).sum()),
            "miss_rate_%": round(100 * (c & ref).sum() / max(c.sum(), 1), 2),
            "missed_per_1000": round(1000 * (c & ref).sum() / len(df), 2),
            "clear_rate_%": round(100 * c.mean(), 1),
            "referral_recall_%": round(100 * (ref & ~c).sum() / max(ref.sum(), 1), 2),
            "overrides": int(over.sum()), "overrides_referable": int((over.values & ref).sum()),
            "severe_or_PDR_cleared": int((c & (df.true_label.values >= 3)).sum())}


def paired(df, a, b, name):
    ref = (df.true_label >= MANDATORY).values
    ma, mb = a.values & ref, b.values & ref
    n01, n10 = int((ma & ~mb).sum()), int((~ma & mb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    pids = df.patient_id.values
    up = np.unique(pids)
    groups = pd.Series(np.arange(len(df))).groupby(pids).apply(np.array).to_dict()
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(N_BOOT):
        ii = np.concatenate([groups[p] for p in rng.choice(up, len(up))])
        diffs.append(int(ma[ii].sum() - mb[ii].sum()))
    return {"comparison": name, "misses_first": int(ma.sum()), "misses_second": int(mb.sum()),
            "missed_only_by_first": n01, "missed_only_by_second": n10, "misses_avoided": n01 - n10,
            "avoided_95CI": f"[{np.percentile(diffs, 2.5):.0f}, {np.percentile(diffs, 97.5):.0f}]",
            "exact_McNemar_p": round(p, 4)}


def analyse(sel):
    from sklearn.metrics import cohen_kappa_score, roc_auc_score
    df = build_df(sel)
    y = df.true_label.values
    out = {}
    out["A_summary"] = pd.Series({
        "images": len(df), "patients": df.patient_id.nunique(),
        "grades": {int(k): int(v) for k, v in df.true_label.value_counts().sort_index().items()},
        "fellow_eye_available_%": round(100 * df.fellow_pred.notna().mean(), 1),
        "accuracy": round(float((df.pred == y).mean()), 4),
        "QWK": round(cohen_kappa_score(y, df.pred, weights="quadratic"), 4),
        "AUC_referable": round(roc_auc_score(y >= 2, 1 - df.p_nr), 4) if 0 < (y >= 2).sum() < len(y) else None,
        "sens_referable_%": round(100 * ((df.pred >= 2) & (y >= 2)).sum() / max((y >= 2).sum(), 1), 1),
        "spec_referable_%": round(100 * ((df.pred < 2) & (y < 2)).sum() / max((y < 2).sum(), 1), 1),
    }).to_frame("value")

    c_v1, o_v1 = clears(df, "v1", V1_T)
    c_g, o_g = clears(df, "v2", GATE_T)
    c_l, o_l = clears(df, "v2", GATE_T, lesion=True)
    c_f, o_f = clears(df, "v2", GATE_T, fellow=True)
    c_lf, o_lf = clears(df, "v2", GATE_T, lesion=True, fellow=True)
    tm = matched_t(df, int(c_lf.sum()))
    c_m, o_m = clears(df, "v2", tm)
    out["B_policies"] = pd.DataFrame([
        row(df, f"top-grade gate (max p >= {V1_T}) [baseline]", c_v1, o_v1),
        row(df, f"binary gate (P_nr >= {GATE_T})", c_g, o_g),
        row(df, "binary + lesion check", c_l, o_l),
        row(df, "binary + fellow eye", c_f, o_f),
        row(df, "binary + lesion + fellow eye (fixed rule)", c_lf, o_lf),
        row(df, f"threshold only, matched clears (t={tm:.4f})", c_m, o_m)])
    out["C_paired"] = pd.DataFrame([
        paired(df, c_v1, c_lf, "top-grade gate vs verify-before-clear (thesis: p = 0.11)"),
        paired(df, c_g, c_lf, "binary gate vs verify-before-clear"),
        paired(df, c_m, c_lf, "matched threshold vs verify-before-clear")])
    ref = y >= MANDATORY
    rows = []
    for t in SWEEP:
        cv, _ = clears(df, "v2", t, lesion=True, fellow=True)
        ct, _ = clears(df, "v2", t)
        cm, _ = clears(df, "v2", matched_t(df, int(cv.sum())))
        rows.append({"t": t, "threshold_only (clears/missed)": f"{int(ct.sum())}/{int((ct & ref).sum())}",
                     "verify (clears/missed)": f"{int(cv.sum())}/{int((cv & ref).sum())}",
                     "threshold_only_matched (clears/missed)": f"{int(cm.sum())}/{int((cm & ref).sum())}"})
    out["D_threshold_sweep"] = pd.DataFrame(rows)

    lines = [f"# Held-out EyePACS evaluation: top {TOP_X} sharpest unused images", ""]
    for k, v in out.items():
        lines += [f"## {k}", v.to_string(), ""]
        v.to_csv(OUT_DIR / f"{k}.csv")
    text = "\n".join(lines)
    (OUT_DIR / "HOLDOUT_SUMMARY.md").write_text(text, encoding="utf-8")
    print(text)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== select (TOP_X = {TOP_X})"); sel = select()
    print("=== preprocess"); preprocess(sel)
    print("=== predict"); predict(sel)
    print("=== lesions"); lesions(sel)
    print("=== analyse"); analyse(sel)


if __name__ == "__main__":
    main()