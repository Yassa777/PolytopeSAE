# analysis/pipeline_geometry_to_hsae.py
# ----------------------------------------------------------------------
# Scaffold for post-hoc analysis of Phase-1/2/3 with one synthesis plot.
# ----------------------------------------------------------------------

import os, json, math, glob, warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- Utilities --------------------------------

def _safe_read_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r") as f:
        return json.load(f)

def _find_first(pattern: str) -> Optional[str]:
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None

def _cosine_angle_degrees(x: np.ndarray, y: np.ndarray) -> float:
    # Euclidean fallback; will be replaced by causal angle when Σ^{-1/2} is provided
    num = float(np.dot(x, y))
    den = float(np.linalg.norm(x) * np.linalg.norm(y)) + 1e-12
    v = max(-1.0, min(1.0, num / den))
    return math.degrees(math.acos(v))

def _whitening_from_unembedding(U: np.ndarray, shrinkage: float = 0.05) -> np.ndarray:
    # Compute W = Σ^{-1/2} from rows of U with shrinkage
    # Σ = cov_rows(U) = (U^T U) / (V-1); shrinkage towards I
    V = U.shape[0]
    S = (U.T @ U) / max(1, V - 1)
    S = (1 - shrinkage) * S + shrinkage * np.eye(S.shape[0], dtype=S.dtype)
    # symmetric PD -> eigh
    evals, evecs = np.linalg.eigh(S)
    evals = np.clip(evals, 1e-8, None)
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return W

def _causal_angle_degrees(x: np.ndarray, y: np.ndarray, W: np.ndarray) -> float:
    xw, yw = W @ x, W @ y
    num = float(np.dot(xw, yw))
    den = float(np.linalg.norm(xw) * np.linalg.norm(yw)) + 1e-12
    v = max(-1.0, min(1.0, num / den))
    return math.degrees(math.acos(v))

def _bootstrap_ci(values: np.ndarray, q: Tuple[float, float] = (2.5, 97.5), n=2000, seed=0):
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed=seed)
    stats = []
    for _ in range(n):
        sample = rng.choice(values, size=len(values), replace=True)
        stats.append(np.nanmedian(sample))
    lo, hi = np.percentile(stats, q)
    return lo, hi

# ----------------------------- Data classes ------------------------------

@dataclass
class Paths:
    runs_root: str = "runs/v2_focused"
    phase1_dir: str = "teacher_extraction"
    phase2_dir: str = "baseline_hsae"
    phase3_dir: str = "teacher_init_hsae"

    def p1(self) -> str: return os.path.join(self.runs_root, self.phase1_dir)
    def p2(self) -> str: return os.path.join(self.runs_root, self.phase2_dir)
    def p3(self) -> str: return os.path.join(self.runs_root, self.phase3_dir)

# ----------------------------- Loaders -----------------------------------

def load_phase1_geometry(paths: Paths, model_unembedding: Optional[np.ndarray] = None,
                         shrinkage: float = 0.05) -> Dict:
    """
    Returns:
      dict with keys:
        - per_parent_angles: DataFrame[ parent_id, child_id, angle_deg ]
        - angle_summary: dict(median, q25, q75, frac_ge_80)
        - ratio_invariance_kl: optional per-parent stats if present
    """
    # Required artifacts (naming can vary slightly across runs)
    teacher_json = _find_first(os.path.join(paths.p1(), "teacher_vectors.json"))
    valid_json   = _find_first(os.path.join(paths.p1(), "validation_results.json"))

    if teacher_json is None:
        raise FileNotFoundError("teacher_vectors.json not found under Phase-1 dir")

    tdata = _safe_read_json(teacher_json)

    parent_vecs = {k: np.array(v, dtype=np.float32) for k, v in tdata.get("parent_vectors", {}).items()}
    child_deltas = {
        pid: {cid: np.array(delta, dtype=np.float32) for cid, delta in d.items()}
        for pid, d in tdata.get("child_deltas", {}).items()
    }

    # Try to compute causal angles. If we don't have U, fall back to Euclidean.
    W = None
    if model_unembedding is not None:
        try:
            W = _whitening_from_unembedding(model_unembedding, shrinkage=shrinkage)
        except Exception as e:
            warnings.warn(f"Whitening failed ({e}); falling back to Euclidean angles")

    rows = []
    for pid, pvec in parent_vecs.items():
        for cid, delt in child_deltas.get(pid, {}).items():
            angle = (_causal_angle_degrees(pvec, delt, W) if W is not None
                     else _cosine_angle_degrees(pvec, delt))
            rows.append((pid, cid, angle))
    df = pd.DataFrame(rows, columns=["parent_id", "child_id", "angle_deg"])

    # Summary & thresholding
    med = float(np.nanmedian(df["angle_deg"])) if len(df) else np.nan
    q25 = float(np.nanpercentile(df["angle_deg"], 25)) if len(df) else np.nan
    q75 = float(np.nanpercentile(df["angle_deg"], 75)) if len(df) else np.nan
    frac_ge_80 = float(np.mean(df["angle_deg"] >= 80.0)) if len(df) else np.nan

    angle_summary = dict(median=med, q25=q25, q75=q75, frac_ge_80=frac_ge_80)

    # Optional: ratio-invariance KL if Phase-1 script saved it
    invariance_kl = None
    if valid_json is not None:
        vdata = _safe_read_json(valid_json)
        invariance_kl = vdata.get("ratio_invariance", None)

    return dict(per_parent_angles=df, angle_summary=angle_summary, ratio_invariance_kl=invariance_kl)

def _load_phase_results_generic(dir_path: str) -> dict:
    # Prefer *results*.json; otherwise pick the latest *.json with 'metrics'
    candidates = sorted(glob.glob(os.path.join(dir_path, "*results*.json")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(dir_path, "*.json")))
    for c in candidates[::-1]:
        try:
            return _safe_read_json(c)
        except Exception:
            continue
    raise FileNotFoundError(f"No JSON results found in {dir_path}")

def load_phase2_baseline(paths: Paths) -> dict:
    return _load_phase_results_generic(paths.p2())

def load_phase3_teacher(paths: Paths) -> dict:
    return _load_phase_results_generic(paths.p3())

# ----------------------------- Metric prep --------------------------------

def extract_global_metrics(results_json: dict) -> dict:
    fm = results_json.get("final_metrics", results_json.get("metrics", {})) or {}
    return {
        "one_minus_EV": fm.get("1-EV", np.nan),
        "one_minus_CE": fm.get("1-CE", np.nan),
        "purity": fm.get("purity", np.nan),
        "leakage": fm.get("leakage", np.nan),
        "steering_leakage": fm.get("steering_leakage", np.nan),
    }

def extract_parentwise_metrics(results_json: dict) -> Optional[pd.DataFrame]:
    # Many trainers store per-parent aggregates under a nested key; if absent, return None
    # Expected shape: {parent_id: {purity: float, leakage: float, steering_leakage: float, ...}, ...}
    # Adjust keys to your logger names if needed.
    per_parent = results_json.get("per_parent_metrics", None)
    if per_parent is None:
        return None
    rows = []
    for pid, mdict in per_parent.items():
        rows.append({
            "parent_id": pid,
            "purity": mdict.get("purity", np.nan),
            "leakage": mdict.get("leakage", np.nan),
            "steering_leakage": mdict.get("steering_leakage", np.nan),
            "one_minus_EV": mdict.get("1-EV", np.nan),
            "one_minus_CE": mdict.get("1-CE", np.nan),
        })
    return pd.DataFrame(rows)

def unify_parentwise(baseline_df: Optional[pd.DataFrame],
                     teacher_df: Optional[pd.DataFrame],
                     geometry_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with columns:
      parent_id, angle_median_deg, purity_baseline, purity_teacher, leakage_baseline,
      leakage_teacher, steering_leakage_baseline, steering_leakage_teacher,
      d_purity, d_leakage, d_steering_leakage
    """
    # Collapse child-level angles to per-parent median
    g = (geometry_df.groupby("parent_id")["angle_deg"].median()
         .rename("angle_median_deg").reset_index())

    # If per-parent metrics are missing, propagate globals (not ideal, but pragmatic for early runs)
    def _inflate(df: Optional[pd.DataFrame], tag: str) -> pd.DataFrame:
        if df is not None and len(df):
            out = df.copy()
            out.columns = [c + f"_{tag}" if c not in ["parent_id"] else c for c in out.columns]
            return out
        else:
            warnings.warn(f"No per-parent metrics for {tag}; using global metrics as fallbacks.")
            return pd.DataFrame()

    B = _inflate(baseline_df, "baseline")
    T = _inflate(teacher_df, "teacher")

    # Merge
    merged = g.copy()
    if len(B): merged = merged.merge(B, on="parent_id", how="left")
    if len(T): merged = merged.merge(T, on="parent_id", how="left")

    # If still empty (no per-parent), just keep geometry; downstream plot will show dots only along x.
    # Compute deltas where both sides exist
    def _delta(col):
        b = col + "_baseline"; t = col + "_teacher"
        if b in merged and t in merged:
            merged["d_" + col] = merged[t] - merged[b]
    for col in ["purity", "leakage", "steering_leakage", "one_minus_EV", "one_minus_CE"]:
        _delta(col)

    return merged

# ----------------------------- Synthesis plot -----------------------------

def plot_geometry_to_cleanliness(df: pd.DataFrame,
                                 outfile: Optional[str] = None,
                                 title: str = "Geometry → Cleanliness Map"):
    """
    NEW PLOT (not in spec): For each parent, x = median causal angle (Phase-1),
    y = Δ leakage (teacher - baseline), marker size ~ Δ purity (teacher - baseline),
    marker edge alpha ~ |Δ steering_leakage| (larger edge alpha = larger reduction).
    Interpretation: Points above 0 indicate higher leakage under teacher-init (bad),
    below 0 indicate leakage reduction (good). Expect negative trend => geometry predicts cleanliness.
    """
    if "angle_median_deg" not in df or "d_leakage" not in df:
        warnings.warn("Insufficient columns to draw synthesis plot.")
        return

    x = df["angle_median_deg"].to_numpy()
    y = df["d_leakage"].to_numpy()

    # Bubble size: Δ purity (absolute)
    if "d_purity" in df:
        s = 200.0 * np.clip(np.abs(df["d_purity"].to_numpy()), 0.0, None) + 10.0
    else:
        s = np.full_like(x, 30.0)

    # Edge alpha encodes steering leakage improvement (cap at 0..1)
    edge_alpha = None
    if "d_steering_leakage" in df:
        sl = df["d_steering_leakage"].to_numpy()
        # bigger reduction => larger alpha
        edge_alpha = np.clip(-sl, 0.0, 1.0)
    else:
        edge_alpha = np.full_like(x, 0.3)

    plt.figure(figsize=(7.5, 5.0))
    # Scatter without specifying colors (tooling constraint)
    for i in range(len(x)):
        plt.scatter([x[i]], [y[i]], s=s[i], linewidths=1.0, edgecolors=(0, 0, 0, float(edge_alpha[i])))

    # Trend line (ordinary least squares on available points)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() >= 2:
        X = np.vstack([np.ones(good.sum()), x[good]]).T
        beta, *_ = np.linalg.lstsq(X, y[good], rcond=None)
        x_line = np.linspace(np.nanmin(x[good]), np.nanmax(x[good]), 100)
        y_line = beta[0] + beta[1] * x_line
        plt.plot(x_line, y_line)

    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Median causal angle: ∠c(ℓ_p, δ_{c|p})  (degrees)")
    plt.ylabel("Δ leakage (teacher − baseline)   ↓ better if negative")
    plt.title(title)
    plt.tight_layout()
    if outfile:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile, dpi=200)
    plt.show()

# ----------------------------- Orchestration ------------------------------

def run_all(runs_root: str = "runs/v2_focused",
            model_unembedding: Optional[np.ndarray] = None,
            shrinkage: float = 0.05,
            save_dir: str = "analysis"):
    paths = Paths(runs_root=runs_root)
    os.makedirs(save_dir, exist_ok=True)

    # Phase-1 geometry
    p1 = load_phase1_geometry(paths, model_unembedding=model_unembedding, shrinkage=shrinkage)
    df_angles = p1["per_parent_angles"]
    angle_summary = p1["angle_summary"]

    # Phase-2 / Phase-3
    p2 = load_phase2_baseline(paths)
    p3 = load_phase3_teacher(paths)

    # Global metrics (for reporting / sanity)
    g2 = extract_global_metrics(p2)
    g3 = extract_global_metrics(p3)
    with open(os.path.join(save_dir, "global_metrics.json"), "w") as f:
        json.dump({"baseline": g2, "teacher": g3, "angle_summary": angle_summary}, f, indent=2)

    # Per-parent metrics if available
    df2 = extract_parentwise_metrics(p2)
    df3 = extract_parentwise_metrics(p3)

    merged = unify_parentwise(df2, df3, df_angles)
    merged.to_csv(os.path.join(save_dir, "geometry_to_performance.csv"), index=False)

    # Bootstrap CI over parent's median angle (useful for Fig.1 narrative)
    ci_lo, ci_hi = _bootstrap_ci(df_angles["angle_deg"].to_numpy())
    with open(os.path.join(save_dir, "angle_bootstrap_ci.json"), "w") as f:
        json.dump({"median_ci_2.5": ci_lo, "median_ci_97.5": ci_hi}, f, indent=2)

    # New synthesis plot
    plot_geometry_to_cleanliness(
        merged,
        outfile=os.path.join(save_dir, "fig_geometry_to_cleanliness.png"),
        title="Geometry → Cleanliness Map (Phase‑1 vs Phase‑3)"
    )

if __name__ == "__main__":
    # Example usage: python analysis/pipeline_geometry_to_hsae.py
    # If you want causal angles, pass your unembedding matrix (np.ndarray) to run_all().
    run_all()

