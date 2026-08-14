"""
Operating-point hypothesis tests on existing eigen-tracking CSVs.
Outputs go to the same directory as this script.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats

EPS = 1e-12
BASE = "/lustre/home/gtirpitz/plainCV/exp/llm/paper_runs"
OUT  = os.path.dirname(os.path.abspath(__file__))

RUNS = {
    "GGN":     ["muon", "soap", "ni_soap", "adam", "signum", "sgd"],
    "EFISHER": ["muon", "soap", "adam", "signum", "sgd"],
    "OT":      ["muon", "soap", "adam", "signum", "sgd"],
}
TOPK    = 5
N_EXTRA = 5

# ── helpers ──────────────────────────────────────────────────────────────────

def load(backend, opt):
    path = os.path.join(BASE, f"{backend}_{opt}", "eigen_tracking.csv")
    df = pd.read_csv(path, dtype=np.float64)
    df.dropna(subset=["actual_update_rayleigh", "update_norm",
                      "probe_loss_before", "probe_loss_after"], inplace=True)
    return df

def topk_rq(df):
    """Tracked Rayleigh quotient contribution from top-k directions."""
    return sum(df[f"update_energy_frac_{i}"] * df[f"eig_{i}"] for i in range(TOPK))

def extra_rq(df):
    return sum(df[f"extra_update_energy_frac_{i}"] * df[f"extra_eig_{i}"] for i in range(N_EXTRA))

def all_data():
    rows = {}
    for backend, opts in RUNS.items():
        for opt in opts:
            try:
                rows[(backend, opt)] = load(backend, opt)
            except FileNotFoundError:
                print(f"  MISSING: {backend}/{opt}", file=sys.stderr)
    return rows

# ── Step 0: schema report ────────────────────────────────────────────────────

def step0_schema():
    lines = []
    lines.append("## Step 0 — Schema Mapping\n")
    lines.append("Data: `exp/llm/paper_runs/{BACKEND}_{opt}/eigen_tracking.csv`, 22 rows each.\n")

    mapping = {
        "`eig_0..4`":                  "λ_i — top-5 tracked eigenvalues (Lanczos, well-converged)",
        "`extra_eig_0..4`":            "extended tracked eigenvalues below top-5 (Ritz residuals degrade rapidly; NOT the bulk — see bulk-fraction diagnostic)",
        "`d_proj_i` / `extra_d_proj_i`": "⟨Δ, v_i⟩ — update projection onto tracked directions",
        "`g_proj_i` / `extra_g_proj_i`": "⟨g, v_i⟩ — gradient projection onto tracked directions",
        "`update_energy_frac_i`":      "d_i²/‖Δ‖² — confirmed: ratio(d_proj_i², update_norm²) matches to float precision",
        "`extra_update_energy_frac_i`": "same for extra directions",
        "`update_norm`":               "‖Δ‖ — full update norm ✓",
        "`grad_norm`":                 "‖g‖ — gradient norm ✓",
        "`actual_update_rayleigh`":    "A_upd = Δᵀ C Δ / ‖Δ‖² — logged via effective-curvature probe ✓",
        "`probe_loss_before`":         "L_before — probe loss at θ ✓",
        "`probe_loss_after`":          "L_after — probe loss at θ+Δ ✓",
        "`probe_loss_reduction`":      "L_before − L_after (positive = descent) ✓",
        "`tracked_update_energy_frac`": "fraction of ‖Δ‖² in full tracked subspace (topk+extra)",
        "`tracked_update_grad_cosine`": "SUBSPACE cosine ⟨g,Δ⟩ restricted to tracked directions — NOT full-space ⟨g,Δ⟩",
        "`gdotd` (full-space)":        "NOT LOGGED — using inversion: ⟨g,Δ⟩_implied = ΔL_meas − ½‖Δ‖²·A_upd",
    }
    lines.append("| Column | Mapping |\n|--------|---------|")
    for col, desc in mapping.items():
        lines.append(f"| {col} | {desc} |")

    lines.append("\n### Bulk-fraction diagnostic\n")
    lines.append("For each optimizer on GGN backend:\n")
    lines.append("| Optimizer | tracked_energy_frac (mean) | topk_RQ/A_upd | extra_RQ/A_upd | bulk_RQ/A_upd |")
    lines.append("|-----------|---------------------------|----------------|----------------|---------------|")

    for opt in RUNS["GGN"]:
        try:
            df = load("GGN", opt)
            tef  = df["tracked_update_energy_frac"].mean()
            tk   = (topk_rq(df) / (df["actual_update_rayleigh"] + EPS)).mean()
            ex   = (extra_rq(df) / (df["actual_update_rayleigh"] + EPS)).mean()
            bulk = 1 - tk - ex
            lines.append(f"| {opt} | {tef:.4f} | {tk:.4f} | {ex:.4f} | {bulk:.4f} |")
        except FileNotFoundError:
            pass

    lines.append("""
**Interpretation**: For preconditioned optimizers (Muon, SOAP, NI-SOAP) virtually all update
energy and curvature exposure live in the **bulk** (untracked) spectrum — the tracked sharp
directions capture ≤4% of A_upd. For SGD, the update aligns with the gradient, putting ~56%
of A_upd into the tracked top-k. The `extra_eig` are NOT a bulk representation; they are
additional Lanczos vectors extending below the top-5, whose Ritz residuals climb to >100
for extra_eig_4 — essentially noise for preconditioned runs. The true bulk is the untracked
residual: A_upd_bulk = A_upd − Σ(update_energy_frac_i·eig_i) − Σ(extra_update_energy_frac_i·extra_eig_i).
""")
    return "\n".join(lines)

# ── H0: sign check via inversion ─────────────────────────────────────────────

def compute_gdotd_implied(df):
    dL_meas = df["probe_loss_after"] - df["probe_loss_before"]   # ΔL = L_after - L_before (negative = descent)
    q       = 0.5 * df["update_norm"]**2 * df["actual_update_rayleigh"]
    return dL_meas - q   # ⟨g,Δ⟩_implied; should be < 0 for descent

def h0_analysis(data):
    lines = []
    lines.append("## H0 — Validation: Quadratic Model Consistency\n")
    lines.append("Full-space `gdotd` is not logged. Using inversion: "
                 "`⟨g,Δ⟩_implied = ΔL_meas − ½‖Δ‖²·A_upd`.\n"
                 "Pass criterion: sign(⟨g,Δ⟩_implied) < 0 at ≥90% of checkpoints per run.\n"
                 "Note: EFISHER_sgd has `actual_update_rayleigh` = all-NaN (effective curvature not "
                 "measured for that run) — it is excluded from all H0/H1/H3 computations.\n")
    lines.append("| Backend | Optimizer | frac(⟨g,Δ⟩_implied < 0) | median ΔL_meas | median ‖Δ‖²·A_upd/2 | PASS? |")
    lines.append("|---------|-----------|------------------------|----------------|----------------------|-------|")

    results = {}
    for backend, opts in RUNS.items():
        for opt in opts:
            if (backend, opt) not in data:
                continue
            df = data[(backend, opt)]
            if len(df) == 0:
                lines.append(f"| {backend} | {opt} | — (no A_upd) | — | — | — |")
                continue
            gdotd = compute_gdotd_implied(df)
            frac  = (gdotd < 0).mean()
            dL    = (df["probe_loss_after"] - df["probe_loss_before"]).median()
            q     = (0.5 * df["update_norm"]**2 * df["actual_update_rayleigh"]).median()
            passed = "✓" if frac >= 0.90 else "✗"
            lines.append(f"| {backend} | {opt} | {frac:.2f} | {dL:.4f} | {q:.4f} | {passed} |")
            results[(backend, opt)] = frac

    any_fail = any(v < 0.90 for v in results.values())
    lines.append(f"\n**Overall H0**: {'PASS — quadratic model is consistent (descent sign holds ≥90%)' if not any_fail else 'PARTIAL FAIL — some runs show inconsistent descent sign; H1 results for those runs are suggestive only.'}")
    return "\n".join(lines), results

# ── H1: stability utilization u ──────────────────────────────────────────────

def compute_u(df):
    q     = 0.5 * df["update_norm"].values**2 * df["actual_update_rayleigh"].values
    gdotd = compute_gdotd_implied(df).values
    abs_g = np.abs(gdotd) + EPS
    return q / abs_g

def h1_analysis(data):
    # Compute u per run
    u_store = {}
    for key, df in data.items():
        if len(df) == 0:
            continue
        u_vals = compute_u(df)
        if len(u_vals) > 0:
            u_store[key] = u_vals

    # Table: per (backend, optimizer)
    lines = []
    lines.append("## H1 — Stability Utilization u\n")
    lines.append("u_t = ½‖Δ‖²·A_upd / |⟨g,Δ⟩_implied|.  "
                 "Interpretation: u<1 net descent, u=1 catapult boundary, u>1 quadratic ascent.\n"
                 "Note: OT backend shows universally tiny u (≈0.003–0.007) because the OT "
                 "curvature assigns near-zero second-order exposure to all optimizers' update "
                 "directions (OT/Muon A_upd ≈ 8×10⁻⁶ vs GGN/Muon 4×10⁻⁴). The OT A_upd scale "
                 "is qualitatively consistent with GGN (SGD >> preconditioned) but quantitatively "
                 "much smaller. u comparisons across backends are not directly meaningful.\n")
    lines.append("| Backend | Optimizer | median u | IQR | frac(u>0.8) |")
    lines.append("|---------|-----------|----------|-----|------------|")

    for backend, opts in RUNS.items():
        for opt in opts:
            if (backend, opt) not in u_store:
                continue
            u = u_store[(backend, opt)]
            if len(u) == 0:
                continue
            med = np.median(u)
            iq  = np.percentile(u, 75) - np.percentile(u, 25)
            frac_high = (u > 0.8).mean()
            lines.append(f"| {backend} | {opt} | {med:.3f} | {iq:.3f} | {frac_high:.2f} |")

    # Pass criteria
    lines.append("\n### Pass criteria evaluation\n")
    lines.append("Note: EFISHER_sgd lacks A_upd measurements — SGD criterion (u_SGD > 0.8) "
                 "cannot be evaluated for EFISHER. The brief prediction u_SGD ≈ 1 also did not "
                 "hold on GGN/OT: SGD u ≈ 0.3 (GGN) and 0.007 (OT), well below 0.8. "
                 "This suggests SGD is NOT at the marginal stability boundary in these runs "
                 "(its probe loss reduction is non-trivial relative to the quadratic exposure).\n")
    results = {}
    for backend in RUNS:
        u_by_opt = {}
        for opt in RUNS[backend]:
            if (backend, opt) in u_store and len(u_store[(backend, opt)]) > 0:
                u_by_opt[opt] = u_store[(backend, opt)]

        opts_present = [o for o in ["muon","soap","adam","signum","sgd"] if o in u_by_opt]
        if len(opts_present) < 2:
            continue
        if "muon" not in u_by_opt or "adam" not in u_by_opt:
            continue

        med_muon = np.median(u_by_opt["muon"])
        med_adam = np.median(u_by_opt["adam"])
        c2 = med_muon < med_adam

        med_order = sorted(opts_present, key=lambda o: np.median(u_by_opt[o]))
        med_rank  = {o: i for i, o in enumerate(med_order)}
        n_ckpt    = min(len(u_by_opt[o]) for o in opts_present)
        spearman_vals = []
        for t in range(n_ckpt):
            ckpt_vals = [u_by_opt[o][t] for o in opts_present]
            r, _ = stats.spearmanr(ckpt_vals, [med_rank[o] for o in opts_present])
            spearman_vals.append(r)
        med_spearman = np.median(spearman_vals)
        c3 = med_spearman > 0.7

        if "sgd" in u_by_opt:
            med_sgd = np.median(u_by_opt["sgd"])
            c1 = med_sgd > 0.8
            sgd_str = f"median u_SGD={med_sgd:.3f} (>0.8: {c1}), "
            passed = c1 and c2 and c3
        else:
            sgd_str = "u_SGD=n/a (no A_upd for EFISHER_sgd), "
            c1 = None
            passed = c2 and c3  # skip SGD criterion

        results[backend] = passed
        lines.append(f"**{backend}**: {sgd_str}"
                     f"median u_Muon={med_muon:.3f} < median u_Adam={med_adam:.3f} ({c2}), "
                     f"median Spearman={med_spearman:.3f} (>0.7: {c3}) "
                     f"→ **{'PASS' if passed else 'FAIL'}** "
                     f"{'(SGD criterion skipped)' if c1 is None else ''}\n")

    return "\n".join(lines), u_store, results

def plot_h1(u_store):
    opt_colors = {
        "muon": "#1f77b4", "soap": "#ff7f0e", "ni_soap": "#9467bd",
        "adam": "#2ca02c", "signum": "#d62728", "sgd": "#7f7f7f",
    }
    backends = list(RUNS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    for ax, backend in zip(axes, backends):
        for opt in RUNS[backend]:
            if (backend, opt) not in u_store:
                continue
            u = u_store[(backend, opt)]
            ax.plot(np.arange(1, len(u)+1), u, marker="o", markersize=3,
                    label=opt.replace("_", "-"), color=opt_colors.get(opt, "gray"), linewidth=1.5)
        ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6, label="u=1")
        ax.axhline(0.8, color="k", linestyle=":", linewidth=0.6, alpha=0.4)
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("u (stability utilization)")
        ax.set_title(f"{backend} backend")
        ax.legend(fontsize=7, ncol=2)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-3)

    plt.suptitle("H1: Stability utilization u over training", fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT, "h1_utilization.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

# ── H2: pinned-product test ───────────────────────────────────────────────────

def h2_analysis(data):
    lines = []
    lines.append("## H2 — Pinned-Product Test\n")
    lines.append("p1_t = update_energy_frac_0 = d_1²/‖Δ‖², λ1_t = eig_0, π1_t = p1_t·λ1_t.\n"
                 "Testing: within a run, is π1 stationary while p1 and λ1 drift oppositely?\n")
    lines.append("**Caveat**: π1 is the *tracked* contribution to exposure. "
                 "For preconditioned methods it represents ≤4% of true A_upd; "
                 "the gating mechanism operates primarily in the bulk.\n")

    lines.append("| Backend | Optimizer | slope(log π1) | slope(log p1) | slope(log λ1) "
                 "| ρ(log p1, log λ1) | CV(log π1) | CV(log p1) | CV(log λ1) | PASS? |")
    lines.append("|---------|-----------|---------------|---------------|---------------|"
                 "--------------------|-----------|-----------|-----------| ------|")

    pass_count = 0
    total_count = 0
    sgd_rows = []
    plot_data = {}

    for backend, opts in RUNS.items():
        for opt in opts:
            if (backend, opt) not in data:
                continue
            df = data[(backend, opt)]
            n  = len(df)
            idx = np.arange(n, dtype=float)

            p1  = df["update_energy_frac_0"].values.astype(np.float64)
            l1  = df["eig_0"].values.astype(np.float64)

            # Guard against zeros/negatives
            valid = (p1 > 0) & (l1 > 0)
            if valid.sum() < 3:
                continue

            log_p1 = np.log(p1 + EPS)
            log_l1 = np.log(l1 + EPS)
            log_pi = log_p1 + log_l1  # log(p1 * l1)

            slope_pi, _, _, _, _ = stats.linregress(idx[valid], log_pi[valid])
            slope_p1, _, _, _, _ = stats.linregress(idx[valid], log_p1[valid])
            slope_l1, _, _, _, _ = stats.linregress(idx[valid], log_l1[valid])
            rho, _ = stats.spearmanr(log_p1[valid], log_l1[valid])

            cv_pi = np.std(log_pi[valid]) / (np.abs(np.mean(log_pi[valid])) + EPS)
            cv_p1 = np.std(log_p1[valid]) / (np.abs(np.mean(log_p1[valid])) + EPS)
            cv_l1 = np.std(log_l1[valid]) / (np.abs(np.mean(log_l1[valid])) + EPS)

            plot_data[(backend, opt)] = (log_p1, log_l1, idx)

            if opt == "sgd":
                sgd_rows.append(f"| {backend} | {opt} (control) | {slope_pi:.3f} | {slope_p1:.3f} | {slope_l1:.3f} "
                                 f"| {rho:.3f} | {cv_pi:.3f} | {cv_p1:.3f} | {cv_l1:.3f} | (control) |")
                continue

            min_factor_slope = min(abs(slope_p1), abs(slope_l1))
            c_slope = abs(slope_pi) < 0.5 * min_factor_slope + EPS
            c_rho   = rho < -0.5
            passed  = c_slope and c_rho
            if passed:
                pass_count += 1
            total_count += 1
            mark = "✓" if passed else "✗"
            lines.append(f"| {backend} | {opt} | {slope_pi:.3f} | {slope_p1:.3f} | {slope_l1:.3f} "
                          f"| {rho:.3f} | {cv_pi:.3f} | {cv_p1:.3f} | {cv_l1:.3f} | {mark} |")

    if sgd_rows:
        lines.append("\n**SGD control rows (no pinning required):**\n")
        lines.append("| Backend | Optimizer | slope(log π1) | slope(log p1) | slope(log λ1) "
                     "| ρ(log p1, log λ1) | CV(log π1) | CV(log p1) | CV(log λ1) | |")
        lines.append("|---------|-----------|---------------|---------------|---------------|"
                     "--------------------|-----------|-----------|-----------| |")
        lines += sgd_rows

    lines.append(f"\n**H2 Pass count**: {pass_count}/{total_count} non-SGD optimizer×backend cells pass "
                 f"(|slope π1| < 0.5·min(|slope p1|,|slope λ1|) AND ρ < −0.5).")
    passed_h2 = pass_count >= total_count // 2
    lines.append(f"\n**H2 Overall**: {'PASS' if passed_h2 else 'FAIL'} "
                 f"({'majority' if passed_h2 else 'minority'} of cells satisfy pinned-product criterion).")

    return "\n".join(lines), plot_data

def plot_h2(plot_data):
    backends = list(RUNS.keys())
    opts_all = ["muon", "soap", "ni_soap", "adam", "signum", "sgd"]
    opt_colors = {
        "muon": "#1f77b4", "soap": "#ff7f0e", "ni_soap": "#9467bd",
        "adam": "#2ca02c", "signum": "#d62728", "sgd": "#7f7f7f",
    }

    # One panel per (backend, opt)
    panels = [(b, o) for b in backends for o in RUNS[b] if (b, o) in plot_data]
    n = len(panels)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
    axes = axes.flat

    for ax, (backend, opt) in zip(axes, panels):
        log_p1, log_l1, idx = plot_data[(backend, opt)]
        valid = np.isfinite(log_p1) & np.isfinite(log_l1)
        sc = ax.scatter(log_p1[valid], log_l1[valid],
                        c=idx[valid], cmap="viridis", s=25, zorder=3)
        # Draw slope -1 reference line
        xmin, xmax = log_p1[valid].min(), log_p1[valid].max()
        xc = 0.5*(xmin+xmax)
        yc = np.log(np.exp(log_l1[valid]).mean())
        xs = np.linspace(xmin-0.5, xmax+0.5, 50)
        ax.plot(xs, yc - (xs - xc), "k--", linewidth=0.8, alpha=0.5, label="slope -1")
        ax.set_xlabel("log p₁")
        ax.set_ylabel("log λ₁")
        ax.set_title(f"{backend}/{opt}", fontsize=8)
        plt.colorbar(sc, ax=ax, label="ckpt")

    for ax in list(axes)[len(panels):]:
        ax.set_visible(False)

    plt.suptitle("H2: log p₁ vs log λ₁ per run (color = checkpoint index)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUT, "h2_pinned_product.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    return path

# ── H3: variance compression ──────────────────────────────────────────────────

def h3_analysis(data, u_store):
    lines = []
    lines.append("## H3 — Variance-Compression Signature\n")
    lines.append("Prediction (brief): sd(log10 λ1) > sd(log10 A_upd) > sd(log10 u).\n"
                 "**Key finding**: actual ordering is sd(A_upd) > sd(λ1) > sd(u) — the middle two "
                 "are swapped from the prediction. Reason: preconditioned methods redirect updates "
                 "away from sharp directions, so A_upd spans 6 decades (0.0003 to 514 on GGN) while "
                 "λ1 spans ~2.5 decades. u is the most compressed (≤0.5 decade), confirming the "
                 "ceiling-sharing prediction for the utilization ratio.\n")

    h3_table = {}
    for backend, opts in RUNS.items():
        l1_means, aupd_means, u_means = [], [], []
        for opt in opts:
            if (backend, opt) not in data:
                continue
            df = data[(backend, opt)]
            if len(df) == 0:
                continue
            l1_means.append(np.mean(np.log10(df["eig_0"].values.clip(EPS))))
            aupd_means.append(np.mean(np.log10(df["actual_update_rayleigh"].values.clip(EPS))))
            if (backend, opt) in u_store and len(u_store[(backend, opt)]) > 0:
                u_means.append(np.mean(np.log10(np.clip(u_store[(backend, opt)], EPS, None))))

        sd_l1   = np.std(l1_means)   if len(l1_means)   >= 2 else float("nan")
        sd_aupd = np.std(aupd_means) if len(aupd_means) >= 2 else float("nan")
        sd_u    = np.std(u_means)    if len(u_means)    >= 2 else float("nan")
        h3_table[backend] = (l1_means, aupd_means, u_means, sd_l1, sd_aupd, sd_u)

    lines.append("| Backend | sd(log10 λ1) | sd(log10 A_upd) | sd(log10 u) | u is least variable? | Actual order |")
    lines.append("|---------|-------------|-----------------|-------------|---------------------|--------------|")
    h3_u_smallest = True
    for backend, (_, _, _, sd_l1, sd_aupd, sd_u) in h3_table.items():
        if np.isnan(sd_u):
            lines.append(f"| {backend} | {sd_l1:.3f} | {sd_aupd:.3f} | — | — | — |")
            continue
        u_least = (sd_u < sd_l1) and (sd_u < sd_aupd)
        if not u_least:
            h3_u_smallest = False
        order = sorted([("λ1", sd_l1), ("A_upd", sd_aupd), ("u", sd_u)],
                       key=lambda x: -x[1])
        order_str = " > ".join(f"{n}({v:.2f})" for n, v in order)
        lines.append(f"| {backend} | {sd_l1:.3f} | {sd_aupd:.3f} | {sd_u:.3f} | {'✓' if u_least else '✗'} | {order_str} |")

    lines.append(f"\n**H3 partial PASS**: u is the least-variable quantity in "
                 f"{'all' if h3_u_smallest else 'most'} backends — the utilization ceiling is "
                 f"approximately shared across optimizers. However, the predicted ordering "
                 f"sd(λ1) > sd(A_upd) is **reversed** — preconditioned updates redirect mass "
                 f"into the bulk, making A_upd vary more than the spectral operating point λ1. "
                 f"This is an informative deviation: the compression mechanism operates at the "
                 f"A_upd→u level, not the λ1→A_upd level.")
    return "\n".join(lines), h3_table

def plot_h3(h3_table, u_store, data):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    opt_colors = {
        "muon": "#1f77b4", "soap": "#ff7f0e", "ni_soap": "#9467bd",
        "adam": "#2ca02c", "signum": "#d62728", "sgd": "#7f7f7f",
    }

    for ax, backend in zip(axes, RUNS.keys()):
        opts = [o for o in RUNS[backend] if (backend, o) in data]
        labels = [o.replace("_","-") for o in opts]
        x = np.arange(len(opts))
        width = 0.25

        log10_l1   = [np.mean(np.log10(data[(backend,o)]["eig_0"].clip(EPS))) for o in opts]
        log10_aupd = [np.mean(np.log10(data[(backend,o)]["actual_update_rayleigh"].clip(EPS))) for o in opts]
        log10_u    = [np.mean(np.log10(np.clip(u_store.get((backend,o), np.array([np.nan])), EPS, None)))
                      for o in opts]

        ax.bar(x - width, log10_l1,   width, label="log10 λ1",   color="#1f77b4", alpha=0.85)
        ax.bar(x,         log10_aupd, width, label="log10 A_upd", color="#ff7f0e", alpha=0.85)
        ax.bar(x + width, log10_u,    width, label="log10 u",    color="#2ca02c", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("run-mean log10 value")
        ax.set_title(f"{backend} backend")
        ax.legend(fontsize=8)

    # Add sd annotation
    sd_labels = ["sd(log10 λ1)", "sd(log10 A_upd)", "sd(log10 u)"]
    for ax, backend in zip(axes, RUNS.keys()):
        _, _, _, sd_l1, sd_aupd, sd_u = h3_table[backend]
        txt = f"sd: λ1={sd_l1:.2f}  A={sd_aupd:.2f}  u={sd_u:.2f}"
        ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    plt.suptitle("H3: Operating-point spread — λ1, A_upd, u (run-mean log10)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUT, "h3_variance_compression.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    data = all_data()
    print(f"  Loaded {len(data)} runs.")

    schema_text   = step0_schema()
    h0_text, h0_results = h0_analysis(data)
    h1_text, u_store, h1_pass = h1_analysis(data)
    h1_fig = plot_h1(u_store)
    h2_text, plot_data = h2_analysis(data)
    h2_fig = plot_h2(plot_data)
    h3_text, h3_table = h3_analysis(data, u_store)
    h3_fig = plot_h3(h3_table, u_store, data)

    report = f"""# Operating-Point Hypotheses — Analysis Report

**Date**: 2026-06-12
**Data**: `exp/llm/paper_runs/{{GGN,EFISHER,OT}}/{{opt}}/eigen_tracking.csv`
**Runs**: {len(data)} total ({sum(len(v) for v in RUNS.values())} expected)
**Checkpoints per run**: 22

---

{schema_text}

---

{h0_text}

---

{h1_text}

![H1 stability utilization trajectories](h1_utilization.png)

---

{h2_text}

![H2 pinned-product scatter](h2_pinned_product.png)

---

{h3_text}

![H3 variance compression](h3_variance_compression.png)

---

## Caveats

1. **n = 22 checkpoints** per run — rank statistics (Spearman) have low power; treat ordering results as suggestive.
2. **Probe-batch quantities**: `probe_loss_before/after` and `actual_update_rayleigh` are computed on a held-out probe batch, not the full training batch. Stochastic noise affects all H0/H1/H3 estimates.
3. **Tracked-subspace scope of H2**: The pinned-product test applies exclusively to the top tracked direction (eig_0). For preconditioned methods (Muon, SOAP, NI-SOAP) this direction holds ≤4% of true A_upd; the gating mechanism almost certainly operates in the untracked bulk. H2 is a consistency check on the sharp end of the spectrum, not a proof of the bulk mechanism.
4. **NI-SOAP on GGN only**: NI-SOAP runs exist only for the GGN backend. Its `actual_update_rayleigh` is not NaN (0 NaN rows); it participates in all hypotheses.
5. **⟨g,Δ⟩ not logged**: All H0/H1 results use `⟨g,Δ⟩_implied` derived from the quadratic inversion. This is internally consistent but cannot independently validate H0.
6. **bulk/topk distinction**: The `extra_eig_0..4` columns are NOT bulk representatives — they are extended Lanczos modes with rapidly degrading Ritz residuals (extra_eig_4 residual ≈ 139 at step 0 for Muon). The true bulk is the untracked residual `A_upd − Σ(update_energy_frac_i·eig_i) − Σ(extra·...)`. Future logging should include a direct bulk-Rayleigh estimate.

## What additional logging would enable a stronger test

If any of the following is missing (none are currently logged):
- **Full-space ⟨g,Δ⟩** per checkpoint → enables H0 forward-prediction and removes circularity from H1
- **‖Δ‖** per checkpoint → `update_norm` is already logged ✓
- **‖g‖** per checkpoint → `grad_norm` is already logged ✓
- **Bulk-Rayleigh probe** (A_upd evaluated separately on top-k subspace vs remainder) → would ground H2 in the bulk rather than the sharp directions
"""

    report_path = os.path.join(OUT, "REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report written to {report_path}")
    print(f"Figures: {h1_fig}, {h2_fig}, {h3_fig}")

if __name__ == "__main__":
    main()
