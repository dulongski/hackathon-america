
# app.py — Liga MX "XI Optimizer" (xG Impact + Atributos)
# Run locally with:  streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------
# Config & Constants
# ------------------------------
st.set_page_config(page_title="XI Optimizer — Liga MX", layout="wide")

FORMATIONS = {
    "4-2-3-1": ["GK","RB","RCB","LCB","LB","DM","CM","AM","RW","LW","ST"],
    "4-3-3":   ["GK","RB","RCB","LCB","LB","DM","RCM","LCM","RW","LW","ST"],
    "5-3-2":   ["GK","RWB","RCB","CB","LCB","LWB","DM","RCM","LCM","RST","LST"]
}

# Map slots to canonical base positions used by the data
POSITION_MAP = {
    "GK": "GK",
    "RB": "RB", "RWB": "RB",
    "LB": "LB", "LWB": "LB",
    "RCB": "CB", "LCB": "CB", "CB": "CB",
    "DM": "DM",
    "CM": "CM", "RCM": "CM", "LCM": "CM",
    "AM": "AM",
    "RW": "RW",
    "LW": "LW",
    "ST": "ST", "RST": "ST", "LST": "ST"
}

ATTRIBUTES = ["pace","passing","ball_progression","pressing","aerial","finishing","def_1v1","tackling","vision","stamina"]

# ------------------------------
# Dummy Data
# ------------------------------
def load_dummy_data(seed: int = 7) -> pd.DataFrame:
    """
    Generate a reproducible dummy player dataset with realistic-ish distributions.
    """
    rng = np.random.default_rng(seed)

    # Position pools and rough counts
    pos_counts = {
        "GK": 12,
        "RB": 14,
        "LB": 14,
        "CB": 28,
        "DM": 16,
        "CM": 28,
        "AM": 16,
        "RW": 18,
        "LW": 18,
        "ST": 20,
    }
    clubs = [f"Club {c}" for c in ["A","B","C","D","E","F","G","H","I","J"]]
    footed_choices = ["left","right","both"]

    rows = []
    pid = 1

    def clip(x, lo, hi):
        return float(np.clip(x, lo, hi))

    for pos, count in pos_counts.items():
        for i in range(count):
            club = rng.choice(clubs)
            age = int(rng.normal(25, 3.5))
            age = int(np.clip(age, 18, 36))
            footed = rng.choice(footed_choices, p=[0.28, 0.58, 0.14])

            # Baseline attributes by position family
            base_attr = {
                "GK": dict(pace=45, passing=55, ball_progression=52, pressing=30, aerial=78, finishing=18, def_1v1=70, tackling=55, vision=60, stamina=58),
                "RB": dict(pace=76, passing=66, ball_progression=68, pressing=65, aerial=58, finishing=40, def_1v1=64, tackling=66, vision=60, stamina=74),
                "LB": dict(pace=76, passing=66, ball_progression=68, pressing=65, aerial=58, finishing=40, def_1v1=64, tackling=66, vision=60, stamina=74),
                "CB": dict(pace=60, passing=62, ball_progression=58, pressing=55, aerial=78, finishing=28, def_1v1=76, tackling=78, vision=56, stamina=66),
                "DM": dict(pace=64, passing=70, ball_progression=66, pressing=70, aerial=66, finishing=34, def_1v1=70, tackling=74, vision=68, stamina=72),
                "CM": dict(pace=68, passing=72, ball_progression=70, pressing=68, aerial=62, finishing=48, def_1v1=60, tackling=64, vision=72, stamina=72),
                "AM": dict(pace=70, passing=74, ball_progression=74, pressing=60, aerial=56, finishing=60, def_1v1=48, tackling=52, vision=76, stamina=68),
                "RW": dict(pace=82, passing=68, ball_progression=72, pressing=62, aerial=54, finishing=66, def_1v1=48, tackling=48, vision=66, stamina=72),
                "LW": dict(pace=82, passing=68, ball_progression=72, pressing=62, aerial=54, finishing=66, def_1v1=48, tackling=48, vision=66, stamina=72),
                "ST": dict(pace=76, passing=62, ball_progression=60, pressing=58, aerial=70, finishing=78, def_1v1=46, tackling=44, vision=62, stamina=70),
            }[pos]

            attrs = {}
            for a, mu in base_attr.items():
                attrs[a] = clip(rng.normal(mu, 7.5), 30, 95)

            # RAPM-like impacts by family
            if pos in ["ST","RW","LW","AM"]:
                xgf = rng.normal(0.06 if pos=="ST" else 0.05, 0.03)
                xga = rng.normal(0.00 if pos=="ST" else 0.01, 0.03)
            elif pos in ["CM","DM"]:
                xgf = rng.normal(0.03 if pos=="CM" else 0.02, 0.02)
                xga = rng.normal(-0.01 if pos=="CM" else -0.02, 0.03)
            elif pos in ["RB","LB"]:
                xgf = rng.normal(0.02, 0.02)
                xga = rng.normal(-0.02, 0.03)
            elif pos == "CB":
                xgf = rng.normal(0.01, 0.01)
                xga = rng.normal(-0.03, 0.03)
            else:  # GK
                xgf = rng.normal(0.00, 0.005)
                xga = rng.normal(-0.04, 0.03)

            # Secondary positions (simple rules)
            secondary = []
            if pos == "RB": secondary = ["RWB"]
            if pos == "LB": secondary = ["LWB"]
            if pos == "CB": secondary = ["RCB","LCB"]
            if pos == "CM": secondary = ["RCM","LCM","DM","AM"]
            if pos == "DM": secondary = ["CM","CB"]
            if pos == "AM": secondary = ["CM","RW","LW"]
            if pos == "RW": secondary = ["AM","ST"]
            if pos == "LW": secondary = ["AM","ST"]
            if pos == "ST": secondary = ["RW","LW","RST","LST"]

            # Player row
            row = {
                "player_id": pid,
                "player_name": f"{pos}-{pid:03d}",
                "primary_pos": pos,
                "secondary_pos": secondary,
                "club": club,
                "age": age,
                "footed": footed,
                "rapm_xgf90": float(xgf),
                "rapm_xga90": float(xga),
            }
            row.update(attrs)
            rows.append(row)
            pid += 1

    df = pd.DataFrame(rows)
    return df

# ------------------------------
# Helper functions
# ------------------------------
def canonical_base(slot: str) -> str:
    return POSITION_MAP.get(slot, slot)

def score_player_for_slot(p: pd.Series) -> float:
    # Composite score to pick baseline (offense - defense)
    return float(p["rapm_xgf90"] - p["rapm_xga90"])

def filter_candidates(df: pd.DataFrame, slot: str) -> pd.DataFrame:
    base = canonical_base(slot)
    mask = (df["primary_pos"] == base) | df["secondary_pos"].apply(lambda lst: base in (lst or []))
    return df[mask].copy()

def init_baseline(formation_key: str, df: pd.DataFrame) -> dict:
    xi = {}
    used_ids = set()
    for slot in FORMATIONS[formation_key]:
        cand = filter_candidates(df, slot)
        cand = cand[~cand["player_id"].isin(used_ids)]
        if cand.empty:
            # Fallback: any not-used player
            cand = df[~df["player_id"].isin(used_ids)]
        cand = cand.assign(score=cand.apply(score_player_for_slot, axis=1))
        pick = cand.sort_values(by="score", ascending=False).head(1)
        pid = int(pick["player_id"].iloc[0])
        xi[slot] = pid
        used_ids.add(pid)
    return xi

def compute_team_metrics(xi: dict, df: pd.DataFrame, synergy: bool=False, intensity: float=0.05):
    ids = list(xi.values())
    team = df[df["player_id"].isin(ids)].copy()

    xgf = float(team["rapm_xgf90"].sum())
    xga = float(team["rapm_xga90"].sum())

    # Simple synergy model (optional): small adjustments based on passing link quality and def link
    if synergy:
        # Adjacency pairs by common football logic
        adj_pairs = [
            ("RB","RW"), ("LB","LW"), ("RCB","DM"), ("LCB","DM"), ("CB","DM"),
            ("DM","CM"), ("CM","AM"), ("RW","ST"), ("LW","ST"), ("RCM","RW"), ("LCM","LW")
        ]
        # Build lookup slot -> player row
        slot_player = {slot: team.loc[team["player_id"] == pid] for slot, pid in xi.items()}
        for (a,b) in adj_pairs:
            if a in xi and b in xi:
                A = slot_player[a]
                B = slot_player[b]
                if not A.empty and not B.empty:
                    pa = float(A["passing"].iloc[0])
                    pb = float(B["passing"].iloc[0])
                    link = ((pa + pb)/200.0) - 0.5  # -0.5..+0.5
                    xgf += intensity * link  # tiny bump if strong link

                    da = float(A["def_1v1"].iloc[0])
                    tb = float(B["tackling"].iloc[0])
                    def_link = ((da + tb)/200.0) - 0.5
                    xga += -intensity * def_link  # reduce GA if defensive link strong

    attr_means = {a: float(team[a].mean()) for a in ATTRIBUTES}
    return dict(xGF90=xgf, xGA90=xga, attr_means=attr_means, team_df=team)

def pct_change(curr: float, base: float) -> float:
    denom = base if abs(base) > 1e-6 else 1e-6
    return 100.0 * (curr - base) / denom

def player_row(df, pid):
    r = df.loc[df["player_id"]==pid]
    return r.iloc[0] if not r.empty else None

# ------------------------------
# UI
# ------------------------------
@st.cache_data
def get_df():
    return load_dummy_data(seed=7)

df = get_df()

# State init
if "formation" not in st.session_state:
    st.session_state.formation = "4-2-3-1"
if "xi_baseline" not in st.session_state:
    st.session_state.xi_baseline = init_baseline(st.session_state.formation, df)
if "xi_current" not in st.session_state:
    st.session_state.xi_current = dict(st.session_state.xi_baseline)
if "synergy_on" not in st.session_state:
    st.session_state.synergy_on = False
if "synergy_intensity" not in st.session_state:
    st.session_state.synergy_intensity = 0.05

# Sidebar controls
st.sidebar.header("⚙️ Configuración")
formation = st.sidebar.selectbox("Formación", list(FORMATIONS.keys()), index=list(FORMATIONS.keys()).index(st.session_state.formation))
if formation != st.session_state.formation:
    st.session_state.formation = formation
    st.session_state.xi_baseline = init_baseline(formation, df)
    st.session_state.xi_current = dict(st.session_state.xi_baseline)

st.session_state.synergy_on = st.sidebar.toggle("Activar sinergia (simple)", value=st.session_state.synergy_on)
st.session_state.synergy_intensity = st.sidebar.slider("Intensidad sinergia", 0.0, 0.20, st.session_state.synergy_intensity, 0.01)

if st.sidebar.button("🔄 Reset XI a Baseline"):
    st.session_state.xi_current = dict(st.session_state.xi_baseline)

# Main layout
st.title("XI Optimizer — Liga MX (Demo)")
st.caption("Selecciona formación, sustituye jugadores por slot y observa el impacto en xG For/Against y atributos del equipo (vs baseline).")

col_left, col_right = st.columns([0.55, 0.45], gap="large")

with col_left:
    st.subheader("Alineación Actual")
    # To avoid duplicates: track already picked IDs
    picked_ids = set()
    for slot in FORMATIONS[st.session_state.formation]:
        base_pos = canonical_base(slot)
        cands = filter_candidates(df, slot)
        # Filter out already picked in other slots except this slot's current
        current_pid = st.session_state.xi_current.get(slot, None)
        cands = cands[ (~cands["player_id"].isin(picked_ids)) | (cands["player_id"]==current_pid) ]
        # Build display labels
        cands = cands.sort_values(by=["primary_pos","player_name"])
        options = cands["player_id"].tolist()
        labels = [f'{row.player_name}  · {row.primary_pos}  · xGF90:{row.rapm_xgf90:.3f}  xGA90:{row.rapm_xga90:.3f}'
                  for _,row in cands.iterrows()]

        # Current selection index
        try:
            curr_index = options.index(current_pid)
        except ValueError:
            curr_index = 0
            current_pid = options[0] if options else None

        new_pid = st.selectbox(f"{slot} ({base_pos})", options=options, index=curr_index, format_func=lambda pid: labels[options.index(pid)] if pid in options else str(pid), key=f"slot_{slot}")
        st.session_state.xi_current[slot] = new_pid
        picked_ids.add(new_pid)

    # Per-slot reset buttons
    st.markdown("---")
    cols = st.columns(4)
    i = 0
    for slot in FORMATIONS[st.session_state.formation]:
        if cols[i].button(f"Reset {slot}"):
            st.session_state.xi_current[slot] = st.session_state.xi_baseline[slot]
        i = (i + 1) % 4

with col_right:
    # Compute metrics
    base = compute_team_metrics(st.session_state.xi_baseline, df, synergy=st.session_state.synergy_on, intensity=st.session_state.synergy_intensity)
    curr = compute_team_metrics(st.session_state.xi_current, df, synergy=st.session_state.synergy_on, intensity=st.session_state.synergy_intensity)

    # KPIs
    st.subheader("Impacto vs Baseline")
    k1, k2 = st.columns(2)
    k1.metric("xGF90 (Equipo)", f"{curr['xGF90']:.3f}", delta=f"{curr['xGF90']-base['xGF90']:+.3f}")
    # For xGA: lower is better → invert meaning in delta color
    delta_xga = curr["xGA90"] - base["xGA90"]
    k2.metric("xGA90 (Equipo, ↓ mejor)", f"{curr['xGA90']:.3f}", delta=f"{delta_xga:+.3f}")

    # Bar chart baseline vs current
    st.plotly_chart(
        go.Figure(data=[
            go.Bar(name="Baseline", x=["xGF90","xGA90"], y=[base["xGF90"], base["xGA90"]]),
            go.Bar(name="Actual", x=["xGF90","xGA90"], y=[curr["xGF90"], curr["xGA90"]]),
        ]).update_layout(barmode="group", height=300, title="xG For / Against — Baseline vs Actual"),
        use_container_width=True
    )

    # Team attributes table + % change
    base_attrs = base["attr_means"]
    curr_attrs = curr["attr_means"]
    rows = []
    for a in ATTRIBUTES:
        b = base_attrs[a]
        c = curr_attrs[a]
        rows.append({"attribute": a, "baseline": b, "current": c, "% change": pct_change(c, b)})
    attr_df = pd.DataFrame(rows).sort_values(by="% change", ascending=False)
    st.subheader("Atributos del Equipo (Δ% vs Baseline)")
    st.dataframe(attr_df.style.format({"baseline":"{:.1f}","current":"{:.1f}","% change":"{:+.1f}%"}), use_container_width=True, height=300)

    # Bar chart of % change
    st.plotly_chart(
        px.bar(attr_df, x="% change", y="attribute", orientation="h").update_layout(height=420, title="Cambio porcentual de atributos"),
        use_container_width=True
    )

    # Replacement diff panel — show last changed slot if any
    st.markdown("---")
    st.subheader("Comparador de reemplazo (slot por slot)")
    # Find first slot where current != baseline
    diff_slots = [s for s in FORMATIONS[st.session_state.formation] if st.session_state.xi_current[s] != st.session_state.xi_baseline[s]]
    if diff_slots:
        s = diff_slots[0]
        pid_old = st.session_state.xi_baseline[s]
        pid_new = st.session_state.xi_current[s]
        a = player_row(df, pid_old)
        b = player_row(df, pid_new)
        if a is not None and b is not None:
            st.write(f"**{s}** — {a['player_name']} ➜ {b['player_name']}")
            d1, d2, d3 = st.columns(3)
            d1.metric("Δ xGF90 (jugador)", f"{b['rapm_xgf90'] - a['rapm_xgf90']:+.3f}")
            d2.metric("Δ xGA90 (jugador, ↓ mejor)", f"{b['rapm_xga90'] - a['rapm_xga90']:+.3f}")
            # Top-3 attribute movers by absolute change
            diffs = [(attr, float(b[attr]-a[attr])) for attr in ATTRIBUTES]
            diffs.sort(key=lambda x: abs(x[1]), reverse=True)
            top3 = diffs[:3]
            with d3:
                st.write("**Top-3 atributos (Δ):**")
                for attr, v in top3:
                    st.write(f"- {attr}: {v:+.1f}")
    else:
        st.caption("No hay cambios vs. baseline aún. Cambia un jugador para ver el comparador.")
        
st.markdown("---")
st.caption("Demo con datos dummy. En datos reales, `rapm_xgf90/xga90` provienen de un modelo RAPM-ΔxG y los atributos de tu feature store.")
