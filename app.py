# app.py
"""
 Agricultural Supply Chain — Stochastic Routing & Allocation Lab
==================================================================
A single-file Streamlit application that:

  1. Generates synthetic shipment + network data (numpy / pandas)
  2. Trains & compares 5 regression models (scikit-learn) predicting
     a Produce Deterioration Factor in [0, 1]
  3. Solves a multi-period, multi-depot MILP (PuLP) for routing &
     allocation
  4. Runs a full stochastic-programming analysis:
        - EV   (Expected Value / deterministic mean problem)
        - EEV  (Expected result of using the EV solution)
        - RP   (Recourse Problem / here-and-now stochastic optimum)
        - WS   (Wait-and-See / perfect information)
        - VSS = EEV - RP   (Value of the Stochastic Solution)
        - EVPI = RP - WS   (Expected Value of Perfect Information)
  5. Shows a Sample Average Approximation (SAA) convergence sweep
     -> answers "does adding stochastic scenarios increase capacity?"

Run:  streamlit run app.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pulp
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ══════════════════════════════════════════════════════════════════════
# 0. Page config & theme
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Agri Supply Chain Lab",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

GREEN = "#2E7D32"
AMBER = "#EF6C00"
BLUE = "#1565C0"
RED = "#C62828"

st.markdown(
    """
    <style>
      .big-title {font-size: 2.1rem; font-weight: 800; color: #1B5E20; margin-bottom: 0;}
      .subtitle  {font-size: 1.0rem; color: #555; margin-top: 0;}
      .metric-box {padding: 0.8rem 1rem; border-radius: 10px; background: #F1F8E9;
                   border-left: 6px solid #2E7D32;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="big-title">🌾 Agricultural Supply Chain Lab</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Machine-learning deterioration forecasting '
    '→ multi-period MILP routing → stochastic-programming value analysis.</p>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════
# 1. Synthetic data generators
# ══════════════════════════════════════════════════════════════════════

CARGO_TYPES = ["tomatoes", "lettuce", "berries", "potatoes", "grains", "chemicals"]
CARGO_EFFECT = {
    "tomatoes": 0.050,
    "lettuce": 0.060,
    "berries": 0.090,
    "potatoes": 0.020,
    "grains": 0.015,
    "chemicals": 0.004,   # near non-perishable
}


@st.cache_data(show_spinner=False)
def generate_shipments(n_rows: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """10k historical shipments with logically consistent spoilage physics."""
    rng = np.random.default_rng(seed)

    cargo = rng.choice(CARGO_TYPES, size=n_rows, p=[0.22, 0.16, 0.14, 0.20, 0.18, 0.10])
    hours = np.clip(rng.gamma(shape=2.5, scale=14.0, size=n_rows), 1, 200)
    temp = np.clip(rng.normal(12, 9, size=n_rows), -10, 45)
    hum = np.clip(rng.normal(72, 14, size=n_rows), 20, 100)

    ce = np.array([CARGO_EFFECT[c] for c in cargo])

    # Logical structure: spoilage ↑ with transit time, heat, humidity,
    # and cargo perishability.
    spoilage = (
        0.006
        + 0.0030 * hours
        + 0.0100 * np.maximum(0, temp - 4.0)
        + 0.0040 * np.maximum(0, hum - 70.0)
        + ce
        + rng.normal(0, 0.020, n_rows)
    )
    spoilage = np.clip(spoilage, 0.0, 1.0)

    return pd.DataFrame(
        {
            "shipment_id": [f"SHP{idx:06d}" for idx in range(n_rows)],
            "cargo_type": cargo,
            "hours_in_transit": hours.round(2),
            "avg_temperature": temp.round(2),
            "avg_humidity": hum.round(2),
            "measured_spoilage_rate": spoilage.round(4),
        }
    )


@dataclass
class Network:
    M: int
    N: int
    T: int
    depot_names: list
    hub_names: list
    supply_cap: np.ndarray          # (M, T)
    demand: np.ndarray              # (N, T)
    distance: np.ndarray            # (M, N)
    transport_cost: np.ndarray      # (M, N)  $/unit
    fixed_cost: np.ndarray          # (M, N)  $ per activation
    link_cap: np.ndarray            # (M, N)  max units
    depot_xy: np.ndarray
    hub_xy: np.ndarray


@st.cache_data(show_spinner=False)
def generate_network(M: int = 5, N: int = 8, T: int = 2, seed: int = 42) -> Network:
    rng = np.random.default_rng(seed + 1)

    depot_names = [f"Farm-{chr(65+i)}" for i in range(M)]
    hub_names = [f"Hub-{j+1:02d}" for j in range(N)]

    depot_xy = rng.uniform(0, 100, size=(M, 2))
    hub_xy = rng.uniform(0, 100, size=(N, 2))

    # Distance matrix (Euclidean, scaled to km)
    dist = np.linalg.norm(depot_xy[:, None, :] - hub_xy[None, :, :], axis=-1) * 3.5 + 5.0

    transport_cost = dist * rng.uniform(1.2, 2.4, size=(M, N))       # $ / unit
    fixed_cost = rng.uniform(40, 160, size=(M, N))                    # $ / activation
    link_cap = np.full((M, N), 320.0)

    # Supply & demand across T periods
    supply_cap = rng.uniform(260, 460, size=(M, T)).round(1)
    demand = rng.uniform(70, 190, size=(N, T)).round(1)

    return Network(
        M=M, N=N, T=T,
        depot_names=depot_names, hub_names=hub_names,
        supply_cap=supply_cap, demand=demand,
        distance=dist, transport_cost=transport_cost,
        fixed_cost=fixed_cost, link_cap=link_cap,
        depot_xy=depot_xy, hub_xy=hub_xy,
    )


# ══════════════════════════════════════════════════════════════════════
# 2. Deterioration scenarios (stochastic inputs to the MILP)
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def build_scenarios(
    M: int, N: int, T: int, K: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        D_scen : (K, M, N, T) deterioration factors per scenario
        temp_k : (K,) mean temperature shock per scenario (for display)
        hum_k  : (K,) mean humidity per scenario
        hours  : (M, N) baseline transit hours between depot i and hub j
    """
    rng = np.random.default_rng(seed + 7)
    hours = rng.uniform(8, 72, size=(M, N))            # hours depot→hub
    cargo_idx = rng.integers(0, len(CARGO_TYPES), size=(M, N))

    D_scen = np.zeros((K, M, N, T))
    temp_k, hum_k = [], []

    for k in range(K):
        # Scenario-level weather shock
        temp_shift = rng.normal(0, 6.0)
        hum_shift = rng.normal(0, 8.0)
        temp_k.append(12 + temp_shift)
        hum_k.append(72 + hum_shift)

        for i in range(M):
            for j in range(N):
                ct = CARGO_TYPES[cargo_idx[i, j]]
                ce = CARGO_EFFECT[ct]
                for t in range(T):
                    # Seasonal drift across periods
                    seasonal = 2.5 * np.sin(2 * np.pi * t / max(T, 1))
                    temp = np.clip(12 + temp_shift + seasonal + rng.normal(0, 2), -10, 45)
                    hum = np.clip(72 + hum_shift + rng.normal(0, 4), 20, 100)
                    h = np.clip(hours[i, j] * (1 + 0.10 * rng.normal()), 1, 200)

                    s = (
                        0.006
                        + 0.0030 * h
                        + 0.0100 * max(0, temp - 4)
                        + 0.0040 * max(0, hum - 70)
                        + ce
                        + rng.normal(0, 0.02)
                    )
                    D_scen[k, i, j, t] = float(np.clip(s, 0, 1))

    return D_scen, np.array(temp_k), np.array(hum_k), hours


# ══════════════════════════════════════════════════════════════════════
# 3. ML model lab
# ══════════════════════════════════════════════════════════════════════

def _make_pipeline(model) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), ["cargo_type"])],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("model", model)])


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame, seed: int = 42):
    features = ["cargo_type", "hours_in_transit", "avg_temperature", "avg_humidity"]
    X = df[features]
    y = df["measured_spoilage_rate"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge (α=1.0)": Ridge(alpha=1.0, random_state=seed),
        "Random Forest": RandomForestRegressor(
            n_estimators=120, max_depth=12, n_jobs=-1, random_state=seed
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.06, max_depth=4, random_state=seed
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.07, random_state=seed
        ),
    }

    results, fitted = [], {}
    for name, mdl in models.items():
        pipe = _make_pipeline(mdl)
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        fit_time = time.perf_counter() - t0
        pred = pipe.predict(X_te)
        results.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_te, pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
                "R²": r2_score(y_te, pred),
                "Fit (s)": fit_time,
            }
        )
        fitted[name] = pipe

    res_df = pd.DataFrame(results).sort_values("R²", ascending=False).reset_index(drop=True)
    return res_df, fitted, X_te, y_te


# ══════════════════════════════════════════════════════════════════════
# 4. MILP core
# ══════════════════════════════════════════════════════════════════════

@dataclass
class OptResult:
    status: str
    objective: float
    flow: np.ndarray       # (M, N, T)
    x: np.ndarray          # (M, N, T) 0/1
    unmet: np.ndarray      # (N, T)
    solve_time: float
    total_shipped: float
    active_routes: int
    total_unmet: float


def _solve_milp(
    det: np.ndarray,           # (M, N, T)
    net: Network,
    penalty_price: float,
    unmet_penalty: float,
    fixed_x: np.ndarray | None = None,
    time_limit: int = 25,
) -> OptResult:
    """Single-scenario MILP with continuous flows + binary route activations."""
    M, N, T = det.shape
    prob = pulp.LpProblem("agri_milp", pulp.LpMinimize)

    flow = {
        (i, j, t): pulp.LpVariable(f"f_{i}_{j}_{t}", lowBound=0)
        for i in range(M) for j in range(N) for t in range(T)
    }
    x = {
        (i, j, t): pulp.LpVariable(f"x_{i}_{j}_{t}", cat="Binary")
        for i in range(M) for j in range(N) for t in range(T)
    }
    unmet = {
        (j, t): pulp.LpVariable(f"u_{j}_{t}", lowBound=0)
        for j in range(N) for t in range(T)
    }

    prob += (
        pulp.lpSum(
            net.transport_cost[i, j] * flow[(i, j, t)]
            + penalty_price * det[i, j, t] * flow[(i, j, t)]
            + net.fixed_cost[i, j] * x[(i, j, t)]
            for i in range(M) for j in range(N) for t in range(T)
        )
        + pulp.lpSum(unmet_penalty * unmet[(j, t)] for j in range(N) for t in range(T))
    )

    for i in range(M):
        for t in range(T):
            prob += pulp.lpSum(flow[(i, j, t)] for j in range(N)) <= net.supply_cap[i, t]

    for j in range(N):
        for t in range(T):
            prob += (
                pulp.lpSum(flow[(i, j, t)] for i in range(M)) + unmet[(j, t)]
                >= net.demand[j, t]
            )

    for i in range(M):
        for j in range(N):
            for t in range(T):
                prob += flow[(i, j, t)] <= net.link_cap[i, j] * x[(i, j, t)]

    if fixed_x is not None:
        for i in range(M):
            for j in range(N):
                for t in range(T):
                    prob += x[(i, j, t)] == int(round(fixed_x[i, j, t]))

    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    dt = time.perf_counter() - t0

    status = pulp.LpStatus[prob.status]
    F = np.zeros((M, N, T))
    X = np.zeros((M, N, T))
    U = np.zeros((N, T))
    if status in ("Optimal", "Not Solved"):  # CBC sometimes returns "Not Solved" on time limit
        for i in range(M):
            for j in range(N):
                for t in range(T):
                    F[i, j, t] = flow[(i, j, t)].value() or 0.0
                    X[i, j, t] = x[(i, j, t)].value() or 0.0
        for j in range(N):
            for t in range(T):
                U[j, t] = unmet[(j, t)].value() or 0.0

    return OptResult(
        status=status,
        objective=float(pulp.value(prob.objective) or 0.0),
        flow=F, x=X, unmet=U,
        solve_time=dt,
        total_shipped=float(F.sum()),
        active_routes=int((X > 0.5).sum()),
        total_unmet=float(U.sum()),
    )


def _recourse_cost(
    det_k: np.ndarray,
    net: Network,
    x_fixed: np.ndarray,
    penalty_price: float,
    unmet_penalty: float,
    time_limit: int = 15,
) -> tuple[float, float, float]:
    """Flow-only recourse given fixed route activations. Returns (cost, shipped, unmet)."""
    M, N, T = det_k.shape
    prob = pulp.LpProblem("recourse", pulp.LpMinimize)

    flow = {
        (i, j, t): pulp.LpVariable(f"f_{i}_{j}_{t}", lowBound=0)
        for i in range(M) for j in range(N) for t in range(T)
    }
    unmet = {
        (j, t): pulp.LpVariable(f"u_{j}_{t}", lowBound=0)
        for j in range(N) for t in range(T)
    }

    prob += (
        pulp.lpSum(
            net.transport_cost[i, j] * flow[(i, j, t)]
            + penalty_price * det_k[i, j, t] * flow[(i, j, t)]
            for i in range(M) for j in range(N) for t in range(T)
        )
        + pulp.lpSum(unmet_penalty * unmet[(j, t)] for j in range(N) for t in range(T))
    )

    for i in range(M):
        for t in range(T):
            prob += pulp.lpSum(flow[(i, j, t)] for j in range(N)) <= net.supply_cap[i, t]
    for j in range(N):
        for t in range(T):
            prob += (
                pulp.lpSum(flow[(i, j, t)] for i in range(M)) + unmet[(j, t)]
                >= net.demand[j, t]
            )
    for i in range(M):
        for j in range(N):
            for t in range(T):
                cap = net.link_cap[i, j] if x_fixed[i, j, t] > 0.5 else 0.0
                prob += flow[(i, j, t)] <= cap

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))

    F = np.zeros((M, N, T))
    U = np.zeros((N, T))
    for i in range(M):
        for j in range(N):
            for t in range(T):
                F[i, j, t] = flow[(i, j, t)].value() or 0.0
    for j in range(N):
        for t in range(T):
            U[j, t] = unmet[(j, t)].value() or 0.0

    return float(pulp.value(prob.objective) or 0.0), float(F.sum()), float(U.sum())


def _solve_stochastic(
    D_scen: np.ndarray,       # (K, M, N, T)
    net: Network,
    penalty_price: float,
    unmet_penalty: float,
    time_limit: int = 40,
) -> OptResult:
    """Two-stage SAA: shared here-and-now x, scenario-dependent recourse flows."""
    K, M, N, T = D_scen.shape
    prob = pulp.LpProblem("stoch_milp", pulp.LpMinimize)

    f = {
        (k, i, j, t): pulp.LpVariable(f"f_{k}_{i}_{j}_{t}", lowBound=0)
        for k in range(K) for i in range(M) for j in range(N) for t in range(T)
    }
    x = {
        (i, j, t): pulp.LpVariable(f"x_{i}_{j}_{t}", cat="Binary")
        for i in range(M) for j in range(N) for t in range(T)
    }
    u = {
        (k, j, t): pulp.LpVariable(f"u_{k}_{j}_{t}", lowBound=0)
        for k in range(K) for j in range(N) for t in range(T)
    }

    w = 1.0 / K
    prob += (
        pulp.lpSum(
            w * (net.transport_cost[i, j] * f[(k, i, j, t)]
                 + penalty_price * D_scen[k, i, j, t] * f[(k, i, j, t)])
            for k in range(K) for i in range(M) for j in range(N) for t in range(T)
        )
        + pulp.lpSum(w * unmet_penalty * u[(k, j, t)]
                     for k in range(K) for j in range(N) for t in range(T))
        + pulp.lpSum(net.fixed_cost[i, j] * x[(i, j, t)]
                     for i in range(M) for j in range(N) for t in range(T))
    )

    for i in range(M):
        for t in range(T):
            for k in range(K):
                prob += pulp.lpSum(f[(k, i, j, t)] for j in range(N)) <= net.supply_cap[i, t]

    for j in range(N):
        for t in range(T):
            for k in range(K):
                prob += (
                    pulp.lpSum(f[(k, i, j, t)] for i in range(M)) + u[(k, j, t)]
                    >= net.demand[j, t]
                )

    for i in range(M):
        for j in range(N):
            for t in range(T):
                for k in range(K):
                    prob += f[(k, i, j, t)] <= net.link_cap[i, j] * x[(i, j, t)]

    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    dt = time.perf_counter() - t0

    status = pulp.LpStatus[prob.status]
    X = np.zeros((M, N, T))
    F_avg = np.zeros((M, N, T))
    U_avg = np.zeros((N, T))
    for i in range(M):
        for j in range(N):
            for t in range(T):
                X[i, j, t] = x[(i, j, t)].value() or 0.0
                F_avg[i, j, t] = np.mean([f[(k, i, j, t)].value() or 0.0 for k in range(K)])
    for j in range(N):
        for t in range(T):
            U_avg[j, t] = np.mean([u[(k, j, t)].value() or 0.0 for k in range(K)])

    return OptResult(
        status=status,
        objective=float(pulp.value(prob.objective) or 0.0),
        flow=F_avg, x=X, unmet=U_avg,
        solve_time=dt,
        total_shipped=float(F_avg.sum()),
        active_routes=int((X > 0.5).sum()),
        total_unmet=float(U_avg.sum()),
    )


# ══════════════════════════════════════════════════════════════════════
# 5. Sidebar controls
# ══════════════════════════════════════════════════════════════════════

st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("Network size")
M = st.sidebar.slider("Supply depots (M)", 3, 6, 5)
N = st.sidebar.slider("Demand hubs (N)", 4, 10, 8)
T = st.sidebar.slider("Time periods (T)", 1, 4, 2)
K = st.sidebar.slider("Stochastic scenarios (K)", 1, 6, 3)

st.sidebar.subheader("Cost parameters")
penalty_price = st.sidebar.slider(
    "Deterioration penalty ($/unit × D)",
    0.0, 60.0, 20.0, step=1.0,
    help="Cost charged per unit shipped, scaled by deterioration factor D ∈ [0,1].",
)
unmet_penalty = st.sidebar.slider(
    "Unmet-demand penalty ($/unit)",
    20.0, 300.0, 120.0, step=5.0,
)

st.sidebar.subheader("Solver")
time_limit = st.sidebar.slider("CBC time limit per MILP (s)", 5, 60, 20)

seed = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: The stochastic lab runs ~2K+2 MILPs. Keep M, N, T, K modest "
    "for fast exploration."
)

# ══════════════════════════════════════════════════════════════════════
# 6. Build shared artefacts
# ══════════════════════════════════════════════════════════════════════

df_ship = generate_shipments(10_000, seed=int(seed))
net = generate_network(M=M, N=N, T=T, seed=int(seed))
D_scen, temp_k, hum_k, hours_ij = build_scenarios(M, N, T, K, seed=int(seed))

res_df, fitted_models, X_te, y_te = train_models(df_ship, seed=int(seed))
best_model_name = res_df.iloc[0]["Model"]
best_model = fitted_models[best_model_name]

# ══════════════════════════════════════════════════════════════════════
# 7. Tabs
# ══════════════════════════════════════════════════════════════════════

tab_data, tab_ml, tab_opt, tab_stoch, tab_math, tab_deploy = st.tabs(
    ["🌾 Network & Data", "🤖 ML Model Lab", "⚙️ MILP Optimizer",
     "🎲 Stochastic Lab", "📐 Mathematics", "🚀 Deploy"]
)


# ──────────────────────────────────────────────────────────────────────
# TAB 1 — Network & Data
# ──────────────────────────────────────────────────────────────────────
with tab_data:
    st.header("Synthetic Enterprise Dataset")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historical shipments", f"{len(df_ship):,}")
    c2.metric("Depots (M)", M)
    c3.metric("Hubs (N)", N)
    c4.metric("Periods (T)", T)

    st.markdown("#### 📦 historical_shipments.csv")
    st.dataframe(df_ship.head(200), use_container_width=True, height=280)

    st.markdown("#### 🔬 Spoilage physics — correlation sanity check")
    num_cols = ["hours_in_transit", "avg_temperature", "avg_humidity", "measured_spoilage_rate"]
    corr = df_ship[num_cols].corr()

    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdYlGn_r",
        title="Correlation matrix — spoilage responds to transit & weather",
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🗺️ Network topology")
    fig = go.Figure()
    for i in range(M):
        for j in range(N):
            fig.add_trace(go.Scatter(
                x=[net.depot_xy[i, 0], net.hub_xy[j, 0]],
                y=[net.depot_xy[i, 1], net.hub_xy[j, 1]],
                mode="lines", line=dict(color="rgba(150,150,150,0.25)", width=1),
                showlegend=False, hoverinfo="skip",
            ))
    fig.add_trace(go.Scatter(
        x=net.depot_xy[:, 0], y=net.depot_xy[:, 1], mode="markers+text",
        text=net.depot_names, textposition="top center",
        marker=dict(size=20, color=GREEN, symbol="square"),
        name="Depots (farms)",
    ))
    fig.add_trace(go.Scatter(
        x=net.hub_xy[:, 0], y=net.hub_xy[:, 1], mode="markers+text",
        text=net.hub_names, textposition="top center",
        marker=dict(size=18, color=BLUE, symbol="circle"),
        name="Hubs (demand)",
    ))
    fig.update_layout(
        height=520, xaxis_title="km (x)", yaxis_title="km (y)",
        title=f"Depot → Hub network  ·  {M}×{N} = {M*N} possible corridors",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 network_nodes.csv — supply / demand tables"):
        cA, cB = st.columns(2)
        with cA:
            st.write("**Supply capacity (depots × periods)**")
            st.dataframe(
                pd.DataFrame(net.supply_cap, index=net.depot_names,
                             columns=[f"t{t}" for t in range(T)]).round(1),
                use_container_width=True,
            )
        with cB:
            st.write("**Demand (hubs × periods)**")
            st.dataframe(
                pd.DataFrame(net.demand, index=net.hub_names,
                             columns=[f"t{t}" for t in range(T)]).round(1),
                use_container_width=True,
            )


# ──────────────────────────────────────────────────────────────────────
# TAB 2 — ML Model Lab
# ──────────────────────────────────────────────────────────────────────
with tab_ml:
    st.header("Deterioration Model — Model Comparison")
    st.caption(
        "Five regressors predict the **Produce Deterioration Factor** "
        "∈ [0,1] from transit time, temperature, humidity and cargo type."
    )

    st.dataframe(
        res_df.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}", "Fit (s)": "{:.2f}"})
        .background_gradient(subset=["R²"], cmap="Greens"),
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("🏆 Best model", best_model_name)
    c2.metric("R² (hold-out)", f"{res_df.iloc[0]['R²']:.4f}")
    c3.metric("MAE (hold-out)", f"{res_df.iloc[0]['MAE']:.4f}")

    fig = px.bar(
        res_df.melt(id_vars="Model", value_vars=["MAE", "RMSE"]),
        x="Model", y="value", color="variable", barmode="group",
        title="Error metrics (lower is better)",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        res_df, x="Model", y="R²", color="R²", color_continuous_scale="Greens",
        title="Explained variance R² (higher is better)",
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🎯 Predicted vs. actual (best model)")
    y_pred = best_model.predict(X_te)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_te, y=y_pred, mode="markers",
        marker=dict(size=4, color=BLUE, opacity=0.4), name="predictions",
    ))
    lims = [0, 1]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                             line=dict(color=RED, dash="dash"), name="y = x"))
    fig.update_layout(
        height=480, xaxis_title="Actual spoilage", yaxis_title="Predicted spoilage",
        xaxis=dict(range=lims), yaxis=dict(range=lims),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📈 Feature importance (tree models)"):
        for name in ["Random Forest", "Gradient Boosting"]:
            pipe = fitted_models[name]
            ohe = pipe.named_steps["pre"].named_transformers_["cat"]
            cat_feats = list(ohe.get_feature_names_out(["cargo_type"]))
            num_feats = ["hours_in_transit", "avg_temperature", "avg_humidity"]
            names = cat_feats + num_feats
            imp = pipe.named_steps["model"].feature_importances_
            imp_df = pd.DataFrame({"feature": names, "importance": imp}).sort_values(
                "importance", ascending=False
            )
            fig = px.bar(imp_df, x="importance", y="feature", orientation="h",
                         title=f"{name} — feature importance")
            fig.update_layout(height=380, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────
# TAB 3 — MILP Optimizer (deterministic mean scenario)
# ──────────────────────────────────────────────────────────────────────
with tab_opt:
    st.header("Deterministic MILP — Mean-Scenario Routing")
    st.caption(
        "Uses the *mean* deterioration field D̄ (over K scenarios). "
        "This is the **Expected Value (EV)** problem."
    )

    D_mean = D_scen.mean(axis=0)

    if st.button("▶️ Solve deterministic MILP", type="primary", key="solve_det"):
        with st.spinner("Solving MILP with CBC…"):
            res = _solve_milp(D_mean, net, penalty_price, unmet_penalty, time_limit=time_limit)
        st.session_state["det_res"] = res

    if "det_res" in st.session_state:
        res: OptResult = st.session_state["det_res"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Objective ($)", f"{res.objective:,.0f}")
        c2.metric("Total shipped (units)", f"{res.total_shipped:,.0f}")
        c3.metric("Active routes", f"{res.active_routes} / {M*N*T}")
        c4.metric("Unmet demand", f"{res.total_unmet:,.0f}")

        st.success(f"Status: **{res.status}** · solve time {res.solve_time:.2f}s")

        # Heatmap of flows
        for t in range(T):
            mat = res.flow[:, :, t]
            fig = px.imshow(
                mat, text_auto=".0f", aspect="auto",
                labels=dict(x="Hub", y="Depot", color="Units"),
                x=net.hub_names, y=net.depot_names,
                color_continuous_scale="YlGn",
                title=f"Flow allocation — period t = {t}",
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        # Active route map
        st.markdown("#### 🚚 Active corridors")
        fig = go.Figure()
        for i in range(M):
            for j in range(N):
                if res.x[i, j].sum() > 0.5:
                    fig.add_trace(go.Scatter(
                        x=[net.depot_xy[i, 0], net.hub_xy[j, 0]],
                        y=[net.depot_xy[i, 1], net.hub_xy[j, 1]],
                        mode="lines", line=dict(color=AMBER, width=3),
                        showlegend=False, hoverinfo="skip",
                    ))
        fig.add_trace(go.Scatter(
            x=net.depot_xy[:, 0], y=net.depot_xy[:, 1], mode="markers+text",
            text=net.depot_names, textposition="top center",
            marker=dict(size=18, color=GREEN, symbol="square"), name="Depots",
        ))
        fig.add_trace(go.Scatter(
            x=net.hub_xy[:, 0], y=net.hub_xy[:, 1], mode="markers+text",
            text=net.hub_names, textposition="top center",
            marker=dict(size=16, color=BLUE), name="Hubs",
        ))
        fig.update_layout(height=500, title="Activated depot→hub corridors (any period)")
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────
# TAB 4 — Stochastic Lab
# ──────────────────────────────────────────────────────────────────────
with tab_stoch:
    st.header("Stochastic Value Analysis")
    st.caption(
        "Two-stage stochastic program (Sample Average Approximation). "
        "Route activations `x` are here-and-now; flows are recourse."
    )

    st.markdown(
        """
        <div class="metric-box">
        <b>Does randomness change capacity?</b><br>
        Yes — modelling uncertainty forces the optimizer to <i>hedge</i>.
        Because the here-and-now route set <code>x</code> must be feasible
        across <i>all</i> K scenarios, the solver typically activates
        <b>more corridors</b> and ships a different volume than the
        mean-scenario (EV) solution. The gap between these worlds is the
        <b>Value of the Stochastic Solution (VSS)</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    run = st.button("🎲 Run full stochastic analysis", type="primary", key="run_stoch")

    if run:
        progress = st.progress(0.0, text="Starting…")
        t_start = time.perf_counter()

        # --- 1. EV / deterministic ---
        progress.progress(0.05, text="Solving EV (mean-scenario) MILP…")
        D_mean = D_scen.mean(axis=0)
        ev = _solve_milp(D_mean, net, penalty_price, unmet_penalty, time_limit=time_limit)

        # --- 2. EEV: fix x from EV, evaluate on each scenario ---
        progress.progress(0.20, text="Evaluating EV solution under uncertainty (EEV)…")
        fixed_cost_ev = float((net.fixed_cost * ev.x.sum(axis=2)).sum())
        recourse_ev = []
        for k in range(K):
            r, _, _ = _recourse_cost(D_scen[k], net, ev.x, penalty_price, unmet_penalty,
                                     time_limit=time_limit)
            recourse_ev.append(r)
        EEV = fixed_cost_ev + float(np.mean(recourse_ev))

        # --- 3. RP: stochastic here-and-now ---
        progress.progress(0.45, text="Solving two-stage stochastic MILP (RP)…")
        rp = _solve_stochastic(D_scen, net, penalty_price, unmet_penalty, time_limit=time_limit)
        RP = rp.objective

        # --- 4. WS: wait-and-see (perfect information) ---
        progress.progress(0.75, text="Solving wait-and-see problems (WS)…")
        ws_costs = []
        for k in range(K):
            r = _solve_milp(D_scen[k], net, penalty_price, unmet_penalty, time_limit=time_limit)
            ws_costs.append(r.objective)
        WS = float(np.mean(ws_costs))

        progress.progress(1.0, text="Done.")
        elapsed = time.perf_counter() - t_start

        VSS = EEV - RP
        EVPI = RP - WS

        st.session_state["stoch"] = dict(
            ev=ev, rp=rp, EEV=EEV, RP=RP, WS=WS, VSS=VSS, EVPI=EVPI,
            ws_costs=ws_costs, recourse_ev=recourse_ev, elapsed=elapsed,
        )

    if "stoch" in st.session_state:
        S = st.session_state["stoch"]
        ev, rp = S["ev"], S["rp"]

        st.success(f"Completed in {S['elapsed']:.1f}s")

        # KPI row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EV cost (deterministic)", f"${ev.objective:,.0f}",
                  help="Optimal cost under mean deterioration.")
        c2.metric("EEV (EV solution, real uncertainty)", f"${S['EEV']:,.0f}",
                  help="Cost of deploying the EV route set under all scenarios.")
        c3.metric("RP (stochastic optimum)", f"${S['RP']:,.0f}",
                  help="Optimal here-and-now expected cost.")
        c4.metric("WS (perfect information)", f"${S['WS']:,.0f}",
                  help="Expected cost if we knew which scenario would occur.")

        c1, c2, c3 = st.columns(3)
        c1.metric("💰 VSS = EEV − RP", f"${S['VSS']:,.0f}",
                  help="Value of modelling uncertainty (≥ 0).")
        c2.metric("🔮 EVPI = RP − WS", f"${S['EVPI']:,.0f}",
                  help="Max price worth paying for perfect forecasts.")
        c3.metric("📈 VSS / RP", f"{100*S['VSS']/max(S['RP'],1e-9):.2f}%")

        st.markdown("---")
        st.markdown("#### 📊 Cost comparison")

        cost_df = pd.DataFrame(
            {
                "Model": ["EV (mean)", "EEV (EV × uncertainty)", "RP (stochastic)", "WS (perfect info)"],
                "Cost ($)": [ev.objective, S["EEV"], S["RP"], S["WS"]],
            }
        )
        fig = px.bar(cost_df, x="Model", y="Cost ($)", color="Cost ($)",
                     color_continuous_scale="RdYlGn_r", text_auto=".3s",
                     title="Deterministic → Stochastic → Perfect information")
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # ---- Capacity comparison ----
        st.markdown("#### 🏗️ Capacity & hedging: does randomness add capacity?")

        cap_df = pd.DataFrame(
            {
                "Solution": ["EV (deterministic)", "RP (stochastic)"],
                "Active routes": [ev.active_routes, rp.active_routes],
                "Total shipped (units)": [ev.total_shipped, rp.total_shipped],
                "Unmet demand (units)": [ev.total_unmet, rp.total_unmet],
            }
        )
        st.dataframe(cap_df.style.format({
            "Total shipped (units)": "{:,.0f}",
            "Unmet demand (units)": "{:,.0f}",
        }), use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Active routes", x=["EV", "RP"],
                             y=[ev.active_routes, rp.active_routes],
                             marker_color=[BLUE, GREEN], yaxis="y"))
        fig.add_trace(go.Bar(name="Total shipped", x=["EV", "RP"],
                             y=[ev.total_shipped, rp.total_shipped],
                             marker_color=["#90CAF9", "#A5D6A7"], yaxis="y2"))
        fig.update_layout(
            height=420, barmode="group",
            yaxis=dict(title="Active routes", side="left"),
            yaxis2=dict(title="Units shipped", overlaying="y", side="right"),
            title="Route activations & volume: EV vs stochastic",
        )
        st.plotly_chart(fig, use_container_width=True)

        delta_routes = rp.active_routes - ev.active_routes
        delta_vol = rp.total_shipped - ev.total_shipped
        delta_unmet = ev.total_unmet - rp.total_unmet

        st.info(
            f"**Interpretation.** Switching from the mean-scenario plan to the "
            f"stochastic plan changes the deployed capacity by "
            f"**{delta_routes:+d} active corridors**, ships "
            f"**{delta_vol:+,.0f} units**, and reduces unmet demand by "
            f"**{delta_unmet:,.0f} units**. "
            f"VSS = **${S['VSS']:,.0f}** is the money the firm would lose by "
            f"pretending the future is deterministic."
        )

        # ---- Scenario breakdown ----
        st.markdown("#### 🌡️ Scenario-level breakdown")
        scen_df = pd.DataFrame({
            "Scenario": [f"S{k+1}" for k in range(K)],
            "Mean temp (°C)": temp_k,
            "Mean humidity (%)": hum_k,
            "Recourse cost under EV routes ($)": S["recourse_ev"],
            "Wait-and-see cost ($)": S["ws_costs"],
        })
        st.dataframe(
            scen_df.style.format({
                "Mean temp (°C)": "{:.1f}",
                "Mean humidity (%)": "{:.1f}",
                "Recourse cost under EV routes ($)": "{:,.0f}",
                "Wait-and-see cost ($)": "{:,.0f}",
            }),
            use_container_width=True,
        )

        # ---- SAA convergence sweep ----
        st.markdown("---")
        st.markdown("#### 🔁 SAA convergence sweep — cost & capacity vs K")
        st.caption(
            "As the number of scenarios K grows, the SAA objective converges. "
            "Watch how **active routes** (deployed capacity) evolve."
        )

        if st.button("Run convergence sweep (K = 1…K)", key="sweep"):
            ks = list(range(1, K + 1))
            rp_costs, rp_routes, rp_volumes = [], [], []
            prog = st.progress(0.0, text="Sweeping K…")
            for idx, kk in enumerate(ks):
                sub = D_scen[:kk]
                r = _solve_stochastic(sub, net, penalty_price, unmet_penalty,
                                      time_limit=time_limit)
                rp_costs.append(r.objective)
                rp_routes.append(r.active_routes)
                rp_volumes.append(r.total_shipped)
                prog.progress((idx + 1) / len(ks), text=f"K = {kk}")
            prog.empty()

            sweep_df = pd.DataFrame({
                "K (scenarios)": ks,
                "RP cost ($)": rp_costs,
                "Active routes": rp_routes,
                "Total shipped": rp_volumes,
            })
            st.dataframe(sweep_df.style.format({
                "RP cost ($)": "{:,.0f}",
                "Total shipped": "{:,.0f}",
            }), use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ks, y=rp_costs, mode="lines+markers",
                                     name="RP cost ($)", line=dict(color=RED, width=3)))
            fig.update_layout(height=380, xaxis_title="K (scenarios)",
                              yaxis_title="RP cost ($)",
                              title="Objective convergence")
            st.plotly_chart(fig, use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=ks, y=rp_routes, name="Active routes",
                                 marker_color=GREEN))
            fig.update_layout(height=380, xaxis_title="K (scenarios)",
                              yaxis_title="Active routes",
                              title="Deployed capacity vs scenario count")
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────
# TAB 5 — Mathematics
# ──────────────────────────────────────────────────────────────────────
with tab_math:
    st.header("Mathematical Formulation")

    st.markdown("### 1. Predictive deterioration model")
    st.markdown(
        "A supervised regressor $f_\\theta$ maps shipment features to a "
        "**deterioration factor** $D \\in [0,1]$:"
    )
    st.latex(
        r"\hat{D}_{ijt} = f_\theta\!\left(h_{ij},\; T_{ijt},\; H_{ijt},\; c_{ij}\right)"
        r"\quad\text{s.t.}\quad \hat{D}_{ijt}\in[0,1]"
    )
    st.markdown(
        "where $h_{ij}$ = transit hours, $T$ = temperature, $H$ = humidity, "
        "$c$ = cargo type. For the synthetic ground truth we used the "
        "physics-inspired generator:"
    )
    st.latex(
        r"D = \mathrm{clip}\!\Big(0.006 + 0.003\,h + 0.010\,\max(0,T-4) "
        r"+ 0.004\,\max(0,H-70) + \kappa_c + \varepsilon,\; 0,\; 1\Big)"
    )

    st.markdown("### 2. Deterministic MILP (EV problem)")
    st.markdown("**Decision variables**")
    st.latex(r"q_{ijt} \ge 0 \quad\text{(continuous flow, units)}")
    st.latex(r"x_{ijt} \in \{0,1\} \quad\text{(route activation)}")
    st.latex(r"u_{jt} \ge 0 \quad\text{(unmet demand)}")

    st.markdown("**Objective**")
    st.latex(
        r"\min\;\; \sum_{t}\sum_{i}\sum_{j}\Big["
        r"\underbrace{c_{ij}\,q_{ijt}}_{\text{transport}}"
        r"+ \underbrace{\pi\, \hat{D}_{ijt}\, q_{ijt}}_{\text{deterioration penalty}}"
        r"+ \underbrace{F_{ij}\, x_{ijt}}_{\text{fixed activation}}\Big]"
        r"+ \sum_{t}\sum_{j} \rho\, u_{jt}"
    )

    st.markdown("**Constraints**")
    st.latex(r"\text{(supply)}\quad \sum_{j} q_{ijt} \le S_{it} \quad \forall i,t")
    st.latex(
        r"\text{(demand)}\quad \sum_{i} q_{ijt} + u_{jt} \ge d_{jt} \quad \forall j,t"
    )
    st.latex(
        r"\text{(link capacity)}\quad q_{ijt} \le \bar{Q}_{ij}\, x_{ijt} \quad \forall i,j,t"
    )
    st.latex(r"q_{ijt}\ge 0,\;\; u_{jt}\ge 0,\;\; x_{ijt}\in\{0,1\}")

    st.markdown("### 3. Two-stage stochastic program (RP)")
    st.markdown(
        "Route activations $x$ are **here-and-now** (decided before uncertainty "
        "resolves); flows $q^{(k)}$ are **recourse** (scenario-dependent). "
        "With K equally-weighted scenarios $\\omega_k$:"
    )
    st.latex(
        r"\min_{x}\; \sum_{i,j,t} F_{ij} x_{ijt} "
        r"+ \frac{1}{K}\sum_{k=1}^{K} \mathcal{Q}(x,\omega_k)"
    )
    st.latex(
        r"\mathcal{Q}(x,\omega_k) = \min_{q^{(k)},u^{(k)}} "
        r"\sum_{t,i,j}\Big(c_{ij} + \pi \hat{D}^{(k)}_{ijt}\Big) q^{(k)}_{ijt} "
        r"+ \sum_{t,j}\rho\, u^{(k)}_{jt}"
    )

    st.markdown("### 4. Value-of-information metrics")
    st.markdown("Let $\\bar{D} = \\frac{1}{K}\\sum_k D^{(k)}$ be the mean field.")
    st.latex(
        r"\text{EV}  = \min_{x,q}\; \text{cost}(x,q;\bar{D}) "
        r"\qquad\text{(deterministic mean problem)}"
    )
    st.latex(
        r"\text{EEV} = \sum_{i,j,t} F_{ij}\bar{x}_{ijt} "
        r"+ \frac{1}{K}\sum_{k=1}^{K} \mathcal{Q}(\bar{x},\omega_k)"
        r"\quad\text{where }\bar{x} = \arg\min \text{EV}"
    )
    st.latex(r"\text{RP} = \min_{x}\; F^\top x + \frac{1}{K}\sum_{k=1}^{K}\mathcal{Q}(x,\omega_k)")
    st.latex(
        r"\text{WS} = \frac{1}{K}\sum_{k=1}^{K} \min_{x,q}\; "
        r"\text{cost}(x,q;\omega_k)"
    )

    st.markdown("**Value of the Stochastic Solution**")
    st.latex(r"\text{VSS} = \text{EEV} - \text{RP} \;\ge\; 0")
    st.markdown("**Expected Value of Perfect Information**")
    st.latex(r"\text{EVPI} = \text{RP} - \text{WS} \;\ge\; 0")

    st.info(
        "**Reading the numbers.** If `VSS ≈ 0`, uncertainty barely matters — "
        "the mean-scenario plan is fine. A large `VSS` means ignoring "
        "stochasticity is expensive. `EVPI` caps how much you should ever pay "
        "for a *perfect* weather/demand forecasting system."
    )

    st.markdown("### 5. Why does capacity change with randomness?")
    st.markdown(
        "In the deterministic EV model the binary vector $\\bar{x}$ only needs "
        "to cover the mean scenario. Under uncertainty, feasibility of the "
        "here-and-now plan must hold for **every** $\\omega_k$:"
    )
    st.latex(
        r"\forall k:\quad \sum_i q^{(k)}_{ijt} + u^{(k)}_{jt} \ge d_{jt}, "
        r"\qquad q^{(k)}_{ijt}\le \bar{Q}_{ij}x_{ijt}"
    )
    st.markdown(
        "The intersection of feasible sets is tighter than any single "
        "scenario's, so the optimizer tends to **add corridors** and shift "
        "volume toward cooler, faster routes — that is the hedging effect you "
        "see in the *Stochastic Lab* tab."
    )


# ──────────────────────────────────────────────────────────────────────
# TAB 6 — Deploy
# ──────────────────────────────────────────────────────────────────────
with tab_deploy:
    st.header("🚀 Deploy to GitHub + Streamlit Cloud")

    st.markdown(
        """
        ### 1. Repo layout
