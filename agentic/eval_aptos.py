"""
External evaluation of the frozen system on APTOS 2019 (all labelled images, every split combined).

Nothing is re-trained or re-tuned: the ensemble weights (0.6/0.4), temperature T, the
binary gate (P(grade 0+1) >= 0.94) and the lesion rule (MA_count > 11) are all the
values selected on the EyePACS validation split.

Stages (each caches its output, so later stages can be re-run without the GPU):
  1. preprocess : same crop_and_resize as EyePACS -> 512x512 PNGs (+ crop validity log)
  2. predict    : ResNet-50 + ViT-B/16 softmax  -> aptos_predictions.csv      (GPU)
  3. lesions    : IDRiD U-Net lesion features    -> aptos_lesion_features.csv  (GPU)
  4. analyse    : grading, calibration, binary referable DR, gate + lesion check,
                  threshold sweep, lesion-vs-grade                             (CPU)

APTOS has no patient IDs / left-right pairing, so the fellow-eye check cannot run.
The LLM planner is not needed: on EyePACS it was identical to the fixed rule.

Usage (from the repo root):
    python agentic/eval_aptos.py            # runs every stage whose output is missing
    python agentic/eval_aptos.py analyse    # only re-run the analysis
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
for sub in ["dataset_prep", "diagnosis_agent", "idrid", "agentic"]:
    sys.path.insert(0, str(ROOT / sub))

# ---------------- CONFIG ----------------
APTOS_DIR = Path(r"E:\APTOS_2019")
# Every labelled split is used: nothing is trained or tuned on APTOS, so all of it is test data.
# All images sit in one folder; the three CSVs only supply the labels.
APTOS_IMAGES = APTOS_DIR / "all_images"
APTOS_SPLITS = [
    (APTOS_DIR / "train_1.csv", APTOS_IMAGES),
    (APTOS_DIR / "valid.csv",   APTOS_IMAGES),
    (APTOS_DIR / "test.csv",    APTOS_IMAGES),
]
ID_COL, LABEL_COL = "id_code", "diagnosis"
EXTS = (".png", ".jpg", ".jpeg")

OUT_DIR = ROOT / "agentic" / "results" / "aptos"
PREP_DIR = OUT_DIR / "images_512"                  # preprocessed images (same pipeline as EyePACS)

RESNET_WEIGHTS = ROOT / "diagnosis_agent" / "resnet50" / "best_model.pt"
VIT_WEIGHTS = ROOT / "diagnosis_agent" / "vitb" / "run2" / "best_model.pt"
RESNET_W, VIT_W = 0.6, 0.4
BATCH_SIZE = 32

GATE_T = 0.94                                      # frozen, selected on EyePACS val
LESION_RULE = ("MA_count", 11.0)                   # frozen, selected on EyePACS val
MANDATORY = 2
SWEEP = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]
N_BOOT = 2000
SEED = 0
# -----------------------------------------

TEMPERATURE = json.load(open(ROOT / "referral_agent" / "temperature_scaling.json"))["temperature"]
PRED_CSV = OUT_DIR / "aptos_predictions.csv"
LES_CSV = OUT_DIR / "aptos_lesion_features.csv"
CROP_CSV = OUT_DIR / "aptos_crop_log.csv"


# ============================================================ 1. preprocessing
def load_labels():
    """All labelled APTOS splits combined into one table (image id, label, source split, file path)."""
    parts = []
    for csv_path, img_dir in APTOS_SPLITS:
        d = pd.read_csv(csv_path)
        assert {ID_COL, LABEL_COL} <= set(d.columns), f"{csv_path} needs {ID_COL} and {LABEL_COL} columns"
        assert d[LABEL_COL].notna().all(), f"{csv_path} has missing labels"
        d = d[[ID_COL, LABEL_COL]].rename(columns={ID_COL: "image", LABEL_COL: "true_label"})
        d["image"] = d["image"].astype(str).str.replace(r"\.(png|jpe?g)$", "", regex=True)
        d["source_split"] = Path(csv_path).stem
        d["path"] = [next((img_dir / (n + e) for e in EXTS if (img_dir / (n + e)).exists()), None)
                     for n in d["image"]]
        parts.append(d)
    lab = pd.concat(parts, ignore_index=True)
    dup = lab["image"].duplicated(keep=False)
    if dup.any():
        same = lab[dup].groupby("image")["true_label"].nunique().eq(1).all()
        print(f"  WARNING: {int(dup.sum())} rows share an id across splits"
              f" ({'same' if same else 'CONFLICTING'} labels); keeping the first of each")
        lab = lab.drop_duplicates("image").reset_index(drop=True)
    missing = lab["path"].isna()
    if missing.any():
        print(f"  WARNING: {int(missing.sum())} labelled images not found on disk, skipped")
        lab = lab[~missing].reset_index(drop=True)
    print(f"  {len(lab)} labelled images from {lab.source_split.value_counts().to_dict()}")
    print(f"  grades: {lab.true_label.value_counts().sort_index().to_dict()}")
    return lab


def stage_preprocess():
    from crop_and_resize import crop_and_resize
    labels = load_labels()
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in enumerate(labels.itertuples(index=False), 1):
        img = cv2.imread(str(r.path))
        base = {"image": r.image, "true_label": int(r.true_label), "source_split": r.source_split}
        if img is None:
            rows.append({**base, "readable": False, "crop_valid": False, "crop_reason": "unreadable"})
            continue
        out, valid, reason = crop_and_resize(img)
        cv2.imwrite(str(PREP_DIR / (r.image + ".png")), out)   # keep invalid crops too; flagged below
        rows.append({**base, "readable": True, "crop_valid": bool(valid), "crop_reason": reason})
        if i % 250 == 0 or i == len(labels):
            print(f"  preprocessed {i}/{len(labels)}")
    log = pd.DataFrame(rows)
    log.to_csv(CROP_CSV, index=False)
    print(f"Crop check: {int(log.crop_valid.sum())} valid, {int((~log.crop_valid).sum())} flagged "
          f"({log[~log.crop_valid].crop_reason.str.split(' ').str[0].value_counts().to_dict()})")


# ============================================================ 2. classifier predictions
def stage_predict():
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from ensemble_eval import build_resnet50, build_vit_b, get_eval_transform, load_model

    log = pd.read_csv(CROP_CSV)
    log = log[log.readable].reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class DS(Dataset):
        def __init__(self, names, tf): self.names, self.tf = names, tf
        def __len__(self): return len(self.names)
        def __getitem__(self, i):
            return self.tf(Image.open(PREP_DIR / (self.names[i] + ".png")).convert("RGB"))

    @torch.no_grad()
    def probs(model, size):
        dl = DataLoader(DS(list(log.image), get_eval_transform(size)), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)
        return np.concatenate([F.softmax(model(x.to(device)), 1).cpu().numpy() for x in dl])

    rn = load_model(build_resnet50(5), RESNET_WEIGHTS, device)
    p_rn = probs(rn, 224)
    del rn
    vit = load_model(build_vit_b(5), VIT_WEIGHTS, device)
    p_vit = probs(vit, 224)
    p_ens = RESNET_W * p_rn + VIT_W * p_vit

    out = log[["image", "true_label", "source_split", "crop_valid"]].copy()
    for name, P in [("resnet", p_rn), ("vit", p_vit), ("ensemble", p_ens)]:
        for k in range(5):
            out[f"{name}_prob_class{k}"] = P[:, k]
    out.to_csv(PRED_CSV, index=False)
    print(f"Saved {len(out)} predictions to {PRED_CSV}")


# ============================================================ 3. lesion features
def stage_lesions():
    import segmentation_models_pytorch as smp
    import torch
    from eyepacs_lesion_analysis import MODEL_PATH, THRESHOLDS_PATH, analyse_image, retina_mask, run_model
    from train_lesion_unet import ENCODER, LESIONS, preprocess

    thresholds = json.load(open(THRESHOLDS_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(ENCODER, encoder_weights=None, in_channels=3, classes=len(LESIONS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    log = pd.read_csv(CROP_CSV)
    log = log[log.readable].reset_index(drop=True)
    rows = []
    for start in range(0, len(log), 8):
        chunk = log.iloc[start:start + 8]
        imgs = [cv2.imread(str(PREP_DIR / (n + ".png"))) for n in chunk.image]
        P = run_model(model, [preprocess(im) for im in imgs], device)
        for k, (name, im) in enumerate(zip(chunk.image, imgs)):
            feat, _ = analyse_image(P[k], retina_mask(im), thresholds)
            rows.append({"image": name, **feat})
        if (start // 8) % 50 == 0:
            print(f"  lesions {min(start + 8, len(log))}/{len(log)}")
    pd.DataFrame(rows).to_csv(LES_CSV, index=False)
    print(f"Saved lesion features to {LES_CSV}")


# ============================================================ 4. analysis
def calibrate(P, t=TEMPERATURE):
    z = np.log(P + 1e-12) / t
    c = np.exp(z - z.max(1, keepdims=True))
    return c / c.sum(1, keepdims=True)


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(conf[s].mean() - correct[s].mean())
    return e


def boot_ci(n, stat, rng):
    vals = [stat(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    return np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5)


def triage(df, t, lesion=False):
    """Returns AUTO_CLEAR mask and REFER_VERIFIED mask (one-way override)."""
    cand = (df.pred < MANDATORY) & (df.p_nr >= t)
    over = cand & (df[LESION_RULE[0]] > LESION_RULE[1]) if lesion else pd.Series(False, index=df.index)
    return cand & ~over, over


def matched_threshold(df, n_clears):
    """Threshold-only gate that clears (at most) n_clears images."""
    s = df.loc[df.pred < MANDATORY, "p_nr"].sort_values(ascending=False).values
    if n_clears <= 0:
        return 1.0
    return float(np.nextafter(s[n_clears - 1], 1.0)) if n_clears < len(s) else 0.0


def policy_row(df, name, clear, over=None):
    ref = df.true_label >= MANDATORY
    miss = clear & ref
    return {"policy": name, "auto_cleared": int(clear.sum()), "missed_referable": int(miss.sum()),
            "miss_rate_%": round(100 * miss.sum() / max(clear.sum(), 1), 2),
            "referral_recall_%": round(100 * (ref & ~clear).sum() / ref.sum(), 2),
            "clear_rate_%": round(100 * clear.mean(), 1),
            "overrides": int(over.sum()) if over is not None else 0,
            "overrides_referable": int((over & ref).sum()) if over is not None else 0,
            "severe_or_PDR_cleared": int((clear & (df.true_label >= 3)).sum())}


def analyse_subset(df, tag, rng):
    y = df.true_label.values
    out = {}

    # --- A. grading
    rows = []
    for m in ["resnet", "vit", "ensemble"]:
        P = calibrate(df[[f"{m}_prob_class{k}" for k in range(5)]].values)
        g = P.argmax(1)
        lo, hi = boot_ci(len(y), lambda i: cohen_kappa_score(y[i], g[i], weights="quadratic"), rng)
        rows.append({"model": m, "accuracy": round(accuracy_score(y, g), 4),
                     "QWK": round(cohen_kappa_score(y, g, weights="quadratic"), 4),
                     "QWK_CI": f"[{lo:.3f}, {hi:.3f}]"})
    out["A_grading"] = pd.DataFrame(rows)
    out["A_confusion"] = pd.DataFrame(confusion_matrix(y, df.pred, labels=range(5)),
                                      index=[f"true {k}" for k in range(5)],
                                      columns=[f"pred {k}" for k in range(5)])

    # --- B. calibration (ensemble)
    raw = df[[f"ensemble_prob_class{k}" for k in range(5)]].values
    rows = []
    for name, P in [("uncalibrated", raw), (f"T={TEMPERATURE:.4f}", calibrate(raw))]:
        rows.append({"probabilities": name,
                     "ECE_5class": round(ece(P.max(1), (P.argmax(1) == y).astype(float)), 4),
                     "ECE_binary_P(nonref)": round(ece(P[:, :2].sum(1), (y < 2).astype(float)), 4)})
    out["B_calibration"] = pd.DataFrame(rows)

    # --- C. binary referable DR (argmax >= 2 operating point, AUC of P(grade >= 2))
    pos, predpos, score = y >= 2, df.pred.values >= 2, 1 - df.p_nr.values
    sens, spec = (predpos & pos).sum() / pos.sum(), (~predpos & ~pos).sum() / (~pos).sum()
    auc = roc_auc_score(pos, score)
    alo, ahi = boot_ci(len(y), lambda i: roc_auc_score(pos[i], score[i]) if 0 < pos[i].sum() < len(i) else np.nan, rng)
    out["C_binary"] = pd.DataFrame([{"prevalence_referable_%": round(100 * pos.mean(), 1),
                                     "sensitivity_%": round(100 * sens, 1), "specificity_%": round(100 * spec, 1),
                                     "AUC": round(auc, 4), "AUC_CI": f"[{alo:.3f}, {ahi:.3f}]"}])

    # --- D. triage at the frozen gate
    sev = df.pred >= MANDATORY
    c0, _ = triage(df, GATE_T)
    c1, o1 = triage(df, GATE_T, lesion=True)
    tm = matched_threshold(df, int(c1.sum()))
    cm, _ = triage(df, tm)
    rows = [{"route": "severity referral (pred >= 2)", "images": int(sev.sum())},
            {"route": "low-confidence referral", "images": int((~sev & ~c0).sum())},
            {"route": "auto-clear candidate", "images": int(c0.sum())}]
    pol = pd.DataFrame([policy_row(df, f"binary gate P_nr >= {GATE_T}", c0),
                        policy_row(df, "binary gate + lesion check", c1, o1),
                        policy_row(df, f"threshold only, matched clears (t={tm:.4f})", cm)])
    out["D_triage_counts"] = pd.DataFrame(rows)
    out["D_policies"] = pol
    ref = df.true_label >= MANDATORY
    a, b = (c0 & ref), (c1 & ref)                              # gate vs gate+lesion: misses
    m_a, m_b = (cm & ref), (c1 & ref)                          # matched threshold vs gate+lesion
    def mcn(x, z):
        n01, n10 = int((x & ~z).sum()), int((~x & z).sum())
        p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        return n01, n10, round(p, 4)
    rows = []
    for name, (x, z) in [("gate vs gate+lesion", (a, b)), ("matched threshold vs gate+lesion", (m_a, m_b))]:
        n01, n10, p = mcn(x, z)
        rows.append({"comparison": name, "missed_only_by_first": n01, "missed_only_by_second": n10,
                     "exact_McNemar_p": p})
    out["D_paired"] = pd.DataFrame(rows)
    out["D_missed_grades"] = pd.DataFrame({"gate": df[a].true_label.value_counts().sort_index(),
                                           "gate+lesion": df[b].true_label.value_counts().sort_index()}).fillna(0).astype(int)

    # --- E. threshold sweep (threshold-only vs gate+lesion at matched clears)
    rows = []
    for t in SWEEP:
        cv, ov = triage(df, t, lesion=True)
        tmt = matched_threshold(df, int(cv.sum()))
        cmt, _ = triage(df, tmt)
        ct, _ = triage(df, t)
        rows.append({"t": t, "threshold_only_clears": int(ct.sum()), "threshold_only_missed": int((ct & ref).sum()),
                     "verify_clears": int(cv.sum()), "verify_missed": int((cv & ref).sum()),
                     "matched_threshold_clears": int(cmt.sum()), "matched_threshold_missed": int((cmt & ref).sum())})
    out["E_threshold_sweep"] = pd.DataFrame(rows)

    # --- F. lesion features vs grade
    rows = []
    for f in ["MA_count", "HE_count", "EX_count", "SE_count"]:
        v = df[f].values
        rows.append({"feature": f, "spearman_vs_grade": round(spearmanr(v, y)[0], 3),
                     "AUC_grade>=2": round(roc_auc_score(y >= 2, v), 3),
                     "AUC_grade>=3": round(roc_auc_score(y >= 3, v), 3),
                     "median_grade0": float(np.median(v[y == 0])), "median_grade4": float(np.median(v[y == 4]))})
    out["F_lesion_vs_grade"] = pd.DataFrame(rows)

    lines = [f"# APTOS 2019 external evaluation — {tag} (n = {len(df)})",
             f"Grade counts: { {int(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()} }", ""]
    for k, v in out.items():
        lines += [f"## {k}", v.to_string(), ""]
        v.to_csv(OUT_DIR / f"{tag}_{k}.csv")
    return "\n".join(lines)


def stage_analyse():
    pred = pd.read_csv(PRED_CSV)
    les = pd.read_csv(LES_CSV)
    df = pred.merge(les, on="image", how="inner", validate="one_to_one")
    cal = calibrate(df[[f"ensemble_prob_class{k}" for k in range(5)]].values)
    df["pred"] = cal.argmax(1)
    df["p_nr"] = cal[:, :2].sum(1)
    rng = np.random.default_rng(SEED)
    report = [analyse_subset(df, "all", rng)]
    if (~df.crop_valid).any():
        report.append(analyse_subset(df[df.crop_valid].reset_index(drop=True), "crop_valid", rng))
    text = "\n\n".join(report)
    (OUT_DIR / "APTOS_SUMMARY.md").write_text(text, encoding="utf-8")
    print(text)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    stages = [("preprocess", CROP_CSV, stage_preprocess), ("predict", PRED_CSV, stage_predict),
              ("lesions", LES_CSV, stage_lesions)]
    if only != "analyse":
        for name, out, fn in stages:
            if only == name or (only is None and not out.exists()):
                print(f"=== {name}")
                fn()
    if only in (None, "analyse"):
        print("=== analyse")
        stage_analyse()


if __name__ == "__main__":
    main()
