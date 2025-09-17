
import io
import math
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="IMEREC & PNS–IMEREC", layout="wide")

# ----------------------------
# Utilities
# ----------------------------

def df_from_upload(uploaded):
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded)
        else:
            uploaded.seek(0)
            return pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def bytes_png_from_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    buf.seek(0)
    return buf

# ----------------------------
# Normalization (B, H, T)
# ----------------------------

def normalize_matrix(X, crit_types, target_dict=None):
    N = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for c in X.columns:
        col = X[c].astype(float)
        ctype = crit_types.get(c, 'B')
        if ctype == 'B':  # beneficial
            maxv = float(col.max())
            N[c] = (col / maxv) if maxv != 0 else 0.0
        elif ctype == 'H':  # non-beneficial
            minv = float(col.min())
            N[c] = (minv / col.replace(0, np.nan)).fillna(0.0).clip(lower=0.0)
        elif ctype == 'T':  # target-based
            t = float(target_dict.get(c, col.median())) if target_dict else float(col.median())
            D = float(np.max(np.abs(col - t)))
            if D == 0:
                N[c] = 1.0
            else:
                N[c] = 1.0 - (np.abs(col - t) / D)
        else:
            raise ValueError(f"Unknown criterion type for {c}: {ctype}")
        N[c] = N[c].clip(0.0, 1.0)
    return N

# ----------------------------
# Stable transform & IMEREC
# ----------------------------

def f_transform(n, method="log1p", eps=1e-6):
    n = np.asarray(n, dtype=float)
    if method == "log1p":
        return np.abs(np.log1p(n))
    elif method == "epslog":
        return np.abs(np.log(n + eps))
    else:
        raise ValueError("method must be 'log1p' or 'epslog'")

def imerec_scores(N, transform="log1p", eps=1e-6):
    F = pd.DataFrame(f_transform(N.values, method=transform, eps=eps),
                     index=N.index, columns=N.columns)
    S_full = np.log1p(F.mean(axis=1).values)   # ln(1 + avg f)
    S_full = pd.Series(S_full, index=N.index, name="S_full")

    S_minus = {}
    for c in N.columns:
        cols = [x for x in N.columns if x != c]
        S_part = np.log1p(F[cols].mean(axis=1).values)
        S_minus[c] = pd.Series(S_part, index=N.index, name=f"S_minus_{c}")

    E_vals = []
    for c in N.columns:
        E_c = (S_minus[c] - S_full).abs().sum()
        E_vals.append(E_c)
    E = pd.Series(E_vals, index=N.columns, name="E")
    w = (E / E.sum()).rename("w") if E.sum() != 0 else E.copy()
    return S_full, S_minus, E, w, F

# ----------------------------
# PNS (7 linguistic variables)
# ----------------------------

LINGUISTIC_PNS = {
    "Very Low":    (0.15, 0.75, 0.20),
    "Low":         (0.30, 0.50, 0.20),
    "Medium Low":  (0.45, 0.35, 0.20),
    "Medium":      (0.60, 0.20, 0.15),
    "Medium High": (0.70, 0.15, 0.15),
    "High":        (0.80, 0.12, 0.10),
    "Very High":   (0.90, 0.08, 0.10),
}
# ensure mu^2+nu^2 <= 1
for k, (mu, nu, pi) in list(LINGUISTIC_PNS.items()):
    mu = clamp01(mu); nu = clamp01(nu); pi = clamp01(pi)
    if mu**2 + nu**2 > 1.0:
        max_nu = math.sqrt(max(1e-9, 1.0 - mu**2))
        nu = min(nu, max_nu)
    LINGUISTIC_PNS[k] = (mu, nu, pi)

def pns_score(mu, nu, pi):
    return mu**2 - nu**2, mu**2 + nu**2  # (score, accuracy)

def pns_from_term(term):
    return LINGUISTIC_PNS.get(term, LINGUISTIC_PNS["Medium"])

def pns_weighted_aggregate(terms_row, weights):
    mus, nus, pis, ws = [], [], [], []
    for term, w in zip(terms_row, weights):
        mu, nu, pi = pns_from_term(term)
        mus.append(mu * w); nus.append(nu * w); pis.append(pi * w); ws.append(w)
    W = sum(ws) if sum(ws) > 0 else 1.0
    mu_agg = sum(mus) / W; nu_agg = sum(nus) / W; pi_agg = sum(pis) / W
    if mu_agg**2 + nu_agg**2 > 1.0:
        max_nu = math.sqrt(max(1e-9, 1.0 - mu_agg**2))
        nu_agg = min(nu_agg, max_nu)
    score, acc = pns_score(mu_agg, nu_agg, pi_agg)
    return mu_agg, nu_agg, pi_agg, score, acc

# ----------------------------
# Demo data (small)
# ----------------------------

EXAMPLE_OBJ = pd.DataFrame({
    "Alternative": ["A1", "A2", "A3"],
    "Quality (B)": [80, 92, 85],
    "Cost (H)": [12.0, 10.5, 11.0],
    "Delivery (T, t=5)": [6, 5, 7],
})
EXAMPLE_OBJ = EXAMPLE_OBJ.set_index("Alternative")

EXAMPLE_PNS = pd.DataFrame({
    "Alternative": ["A1", "A2", "A3"],
    "Quality (B)": ["Medium", "High", "Medium High"],
    "Cost (H)": ["Medium High", "Very High", "High"],
    "Delivery (T, t=5)": ["Medium", "Very High", "Medium Low"],
}).set_index("Alternative")

# ----------------------------
# UI
# ----------------------------

st.title("IMEREC & PNS–IMEREC (7 Linguistic Variables)")

with st.sidebar:
    st.header("Module")
    mode = st.radio("Choose:", ["IMEREC", "PNS–IMEREC"], index=0)
    st.caption("B = Beneficial, H = Non-beneficial (cost), T = Target-based.")

# ----------------------------
# IMEREC Module
# ----------------------------

if mode == "IMEREC":
    st.subheader("Upload Objective Data (or use example)")
    up = st.file_uploader("CSV/XLSX (alternatives as rows, criteria as columns)", type=["csv","xlsx"])
    if up:
        X = df_from_upload(up)
        if X is not None and "Alternative" in X.columns:
            X = X.set_index("Alternative")
    else:
        st.info("Using small example dataset.")
        X = EXAMPLE_OBJ.copy()

    st.dataframe(X, use_container_width=True)

    st.subheader("Criterion Types & Targets")
    cols = list(X.columns)
    c_types = {}
    targets = {}
    tcols = st.columns(4)
    for i, c in enumerate(cols):
        with tcols[i % 4]:
            default_idx = 0
            if "(H" in c or "H)" in c: default_idx = 1
            if "(T" in c or "T)" in c: default_idx = 2
            ctype = st.selectbox(f"Type — {c}", ["B","H","T"], index=default_idx, key=f"type_{i}")
            c_types[c] = ctype
            if ctype == "T":
                t = st.number_input(f"Target for {c}", value=float(X[c].median()), key=f"tar_{i}")
                targets[c] = t

    st.subheader("Transform Settings")
    tr = st.selectbox("f(n)", ["log1p", "epslog"], index=0)
    eps = st.number_input("ε (for epslog)", value=1e-6, format="%.6f")

    if st.button("Compute IMEREC", type="primary"):
        Xnum = X.apply(pd.to_numeric, errors="coerce")
        if Xnum.isna().any().any():
            st.error("Non-numeric values found. Clean your data or use PNS–IMEREC for linguistic inputs.")
        else:
            N = normalize_matrix(Xnum, c_types, target_dict=targets)
            S_full, S_minus, E, w, F = imerec_scores(N, transform=tr, eps=eps)

            st.markdown("### Normalized Matrix (N)")
            st.dataframe(N.style.format("{:.6f}"), use_container_width=True)

            st.markdown("### IMEREC Weights")
            st.dataframe(w.to_frame().style.format("{:.6f}"), use_container_width=True)

            # Chart (matplotlib, downloadable)
            fig, ax = plt.subplots(figsize=(6,3))
            w.plot(kind="bar", ax=ax)
            ax.set_ylabel("Weight")
            ax.set_title("IMEREC Weights")
            buf = bytes_png_from_fig(fig)
            st.pyplot(fig)
            st.download_button("Download Weights Chart (PNG)", data=buf, file_name="imerec_weights.png", mime="image/png")

            st.markdown("### Full-Model Scores $S_i$")
            st.dataframe(S_full.to_frame().style.format("{:.6f}"), use_container_width=True)

            st.markdown("### Removal Effects $E_j$")
            st.dataframe(E.to_frame().style.format("{:.6f}"), use_container_width=True)

            # JSON download
            results = {
                "normalized": N.round(6).to_dict(),
                "S_full": S_full.round(6).to_dict(),
                "E": E.round(6).to_dict(),
                "w": w.round(6).to_dict(),
                "types": c_types,
                "targets": targets,
                "transform": tr,
                "eps": eps,
                "timestamp": datetime.utcnow().isoformat()+"Z",
            }
            st.download_button("Download IMEREC Results (JSON)",
                               data=json.dumps(results, indent=2),
                               file_name="imerec_results.json",
                               mime="application/json")

# ----------------------------
# PNS–IMEREC Module
# ----------------------------

if mode == "PNS–IMEREC":
    st.subheader("Step 1 — Objective Data for IMEREC (Weights)")
    up1 = st.file_uploader("Objective CSV/XLSX", type=["csv","xlsx"], key="obj")
    if up1:
        X = df_from_upload(up1)
        if X is not None and "Alternative" in X.columns:
            X = X.set_index("Alternative")
    else:
        st.info("Using small example objective data.")
        X = EXAMPLE_OBJ.copy()
    st.dataframe(X, use_container_width=True)

    st.markdown("**Criterion types & targets**")
    cols = list(X.columns)
    c_types = {}
    targets = {}
    tcols = st.columns(4)
    for i, c in enumerate(cols):
        with tcols[i % 4]:
            default_idx = 0
            if "(H" in c or "H)" in c: default_idx = 1
            if "(T" in c or "T)" in c: default_idx = 2
            ctype = st.selectbox(f"Type — {c}", ["B","H","T"], index=default_idx, key=f"ptype_{i}")
            c_types[c] = ctype
            if ctype == "T":
                t = st.number_input(f"Target for {c}", value=float(X[c].median()), key=f"ptar_{i}")
                targets[c] = t

    st.markdown("**Transform**")
    tr = st.selectbox("f(n)", ["log1p","epslog"], index=0, key="pns_tr")
    eps = st.number_input("ε (for epslog)", value=1e-6, format="%.6f", key="pns_eps")

    st.subheader("Step 2 — Linguistic Evaluations (7 terms)")
    st.caption("Allowed terms: " + ", ".join(LINGUISTIC_PNS.keys()))
    up2 = st.file_uploader("Linguistic CSV/XLSX", type=["csv","xlsx"], key="ling")
    if up2:
        L = df_from_upload(up2)
        if L is not None and "Alternative" in L.columns:
            L = L.set_index("Alternative")
    else:
        st.info("Using small example linguistic data.")
        L = EXAMPLE_PNS.copy()
    # align
    L = L.reindex(index=X.index, columns=X.columns)
    st.dataframe(L, use_container_width=True)

    if st.button("Run PNS–IMEREC", type="primary"):
        Xnum = X.apply(pd.to_numeric, errors="coerce")
        if Xnum.isna().any().any():
            st.error("Objective matrix has non-numeric values.")
        else:
            N = normalize_matrix(Xnum, c_types, target_dict=targets)
            S_full, S_minus, E, w, F = imerec_scores(N, transform=tr, eps=eps)

            st.markdown("### IMEREC Weights")
            st.dataframe(w.to_frame().style.format("{:.6f}"), use_container_width=True)
            fig, ax = plt.subplots(figsize=(6,3))
            w.plot(kind="bar", ax=ax)
            ax.set_ylabel("Weight"); ax.set_title("IMEREC Weights")
            buf_w = bytes_png_from_fig(fig)
            st.pyplot(fig)
            st.download_button("Download Weights Chart (PNG)",
                               data=buf_w, file_name="pns_imerec_weights.png", mime="image/png")

            # PNS aggregation
            rows = []
            for alt in L.index:
                terms = [str(L.loc[alt, c]).strip() for c in L.columns]
                mu, nu, pi, score, acc = pns_weighted_aggregate(terms, w.values)
                rows.append({"Alternative": alt, "mu": round(mu,6), "nu": round(nu,6), "pi": round(pi,6),
                             "Score": round(score,6), "Accuracy": round(acc,6)})
            AGG = pd.DataFrame(rows).set_index("Alternative")
            RANK = AGG.sort_values(by=["Score","Accuracy"], ascending=[False, False]).copy()
            RANK["Rank"] = range(1, len(RANK)+1)

            st.markdown("### Aggregated PNS")
            st.dataframe(AGG, use_container_width=True)

            st.markdown("### Final Ranking")
            st.dataframe(RANK[["Score","Accuracy","mu","nu","pi","Rank"]], use_container_width=True)

            # Ranking chart
            fig2, ax2 = plt.subplots(figsize=(7,3))
            RANK["Score"].plot(kind="bar", ax=ax2)
            ax2.set_ylabel("Score"); ax2.set_title("PNS–IMEREC Scores by Alternative")
            buf_r = bytes_png_from_fig(fig2)
            st.pyplot(fig2)
            st.download_button("Download Ranking Chart (PNG)",
                               data=buf_r, file_name="pns_imerec_scores.png", mime="image/png")

            # JSON download
            out = {
                "weights": w.round(6).to_dict(),
                "aggregated_pns": AGG.to_dict(),
                "ranking": RANK.to_dict(),
                "linguistic_map": {k: {"mu":v[0], "nu":v[1], "pi":v[2]} for k,v in LINGUISTIC_PNS.items()},
                "types": c_types, "targets": targets,
                "transform": tr, "eps": eps,
                "timestamp": datetime.utcnow().isoformat()+"Z"
            }
            st.download_button("Download PNS–IMEREC Results (JSON)",
                               data=json.dumps(out, indent=2),
                               file_name="pns_imerec_results.json",
                               mime="application/json")
