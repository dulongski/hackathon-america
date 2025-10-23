# app.py
# Liga MX — Dynamic xG Lineup (artifact-driven)
# UI: vertical pitch (axis-swapped), inline swaps, optional cross-position pool,
# per-swap rankings, baseline save/reset, safe session_state handling.

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

# Optional (pitch). App still runs without it.
import matplotlib.pyplot as plt
try:
    from mplsoccer import VerticalPitch
    MPLSOCCER_OK = True
except Exception:
    MPLSOCCER_OK = False


# =========================
# Paths / Constants
# =========================
ARTIFACTS_DIR = Path("./artifacts")

FILES = {
    "players_meta": ARTIFACTS_DIR / "players_meta.parquet",
    "clusters": ARTIFACTS_DIR / "clusters_team_agnostic.parquet",
    "role_lookup": ARTIFACTS_DIR / "role_lookup.parquet",
    "mapping": ARTIFACTS_DIR / "mapping.json",
    "feature_names": ARTIFACTS_DIR / "feature_names.json",
    "rapm_for": ARTIFACTS_DIR / "rapm_for.pkl",
    "rapm_against": ARTIFACTS_DIR / "rapm_against.pkl",
    "events_min": ARTIFACTS_DIR / "events_min.parquet",
}

LIGA_MX_TEAMS = [
    'Guadalajara', 'Tigres UANL', 'Monterrey', 'León', 'Pachuca',
    'Pumas UNAM', 'Atlas', 'Santos Laguna', 'Cruz Azul', 'Tijuana',
    'América', 'Juárez', 'Puebla', 'Toluca', 'Mazatlán',
    'Atlético San Luis', 'Querétaro', 'Necaxa'
]

TEAM_COLORS = {
    "América": "#fbd000", "Cruz Azul": "#0033a0", "Guadalajara": "#c8102e",
    "Tigres UANL": "#ffb612", "Monterrey": "#0a2342", "León": "#006c3b",
    "Pachuca": "#1f4e79", "Pumas UNAM": "#0d2240", "Atlas": "#e43b3b",
    "Santos Laguna": "#1db954", "Tijuana": "#c8102e", "Juárez": "#5bbf21",
    "Puebla": "#1e90ff", "Toluca": "#c8102e", "Mazatlán": "#6a1b9a",
    "Atlético San Luis": "#0f4c81", "Querétaro": "#1f4e79", "Necaxa": "#ff2a2a",
}
TEAM_BADGE = {
    "América": "🟡", "Cruz Azul": "🔵", "Guadalajara": "🔴", "Tigres UANL": "🟠",
    "Monterrey": "🔵", "León": "🟢", "Pachuca": "🔵", "Pumas UNAM": "🔵",
    "Atlas": "🔴", "Santos Laguna": "🟢", "Tijuana": "🔴", "Juárez": "🟢",
    "Puebla": "🔵", "Toluca": "🔴", "Mazatlán": "🟣", "Atlético San Luis": "🔵",
    "Querétaro": "🔵", "Necaxa": "🔴",
}

GROUPED_POSITIONS = [
    "Goalkeeper",
    "Right Back", "Center Back", "Left Back",
    "Defensive Midfield",
    "Center Midfield",
    "Attacking Midfield",
    "Right Wing", "Left Wing",
    "Center Forward",
]

# Anchors for a 120x80 vertical StatsBomb pitch (we’ll axis-swap when plotting)
POS_ANCHORS = {
    "Goalkeeper":          (40, 10),
    "Right Back":          (10, 24),
    "Center Back":         (40, 24),
    "Left Back":           (70, 24),

    "Defensive Midfield":  (40, 40),
    "Center Midfield":     (40, 58),
    "Attacking Midfield":  (40, 76),

    "Right Wing":          (16, 76),
    "Left Wing":           (64, 76),

    "Center Forward":      (40, 98),
}
DUP_OFFSETS = {
    1: [0],
    2: [-9, +9],
    3: [-18, 0, +18],
    4: [-20, -6, +6, +20],
    5: [-25, -11, 0, +11, +25],
}


# =========================
# Streamlit setup
# =========================
st.set_page_config(page_title="Liga MX — Dynamic xG Lineup", layout="wide")
st.title("Liga MX — Dynamic xG Lineup")

# Validate artifacts
missing = [name for name, p in FILES.items() if not p.exists()]
if missing:
    st.error(f"Missing artifact files: {missing}\nRun your exporter notebook to create them.")
    st.stop()

# =========================
# Cached loaders
# =========================
@st.cache_data(show_spinner=False)
def load_parquet(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p)

@st.cache_data(show_spinner=False)
def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def load_pickle(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)

# Load data
players_meta = load_parquet(FILES["players_meta"])
clusters_df  = load_parquet(FILES["clusters"])
role_lookup_df = load_parquet(FILES["role_lookup"])
role_lookup = role_lookup_df.set_index("player_id")["pos_cluster_id"]
mapping = load_json(FILES["mapping"])
feature_names = load_json(FILES["feature_names"])
rapm_for = load_pickle(FILES["rapm_for"])
rapm_against = load_pickle(FILES["rapm_against"])
events_min = load_parquet(FILES["events_min"])


# =========================
# Helpers (XI + model)
# =========================
def _parse_datetime_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def get_last_starting_xi_ids(events_min: pd.DataFrame, team_name: str):
    df = events_min.copy()
    mt_cols = ["match_id"] + (["match_date"] if "match_date" in df.columns else [])
    team_matches = df.loc[df["team"] == team_name, mt_cols].drop_duplicates("match_id")
    if team_matches.empty:
        raise ValueError(f"No matches found for team='{team_name}'.")

    if "match_date" in team_matches.columns and team_matches["match_date"].notna().any():
        team_matches["match_date"] = _parse_datetime_safe(team_matches["match_date"])
        team_matches = team_matches.dropna(subset=["match_date"])
        latest = team_matches.sort_values("match_date").iloc[-1]
    else:
        latest = team_matches.sort_values("match_id").iloc[-1]

    match_id = int(latest["match_id"])
    emt = df[(df["match_id"] == match_id) & (df["team"] == team_name)].copy()

    xi_rows = emt.loc[emt["type"] == "Starting XI"]
    if not xi_rows.empty and "tactics" in xi_rows.columns:
        tac = xi_rows.iloc[0].get("tactics", {})
        lineup = tac.get("lineup", []) if isinstance(tac, dict) else []
        ids = [p.get("player_id") for p in lineup if isinstance(p, dict) and p.get("player_id") is not None]
        if len(ids) >= 10:
            return [int(x) for x in ids[:11]], match_id

    emt["tmin"] = emt["minute"].fillna(0).astype(float) + emt["second"].fillna(0).astype(float)/60.0
    first_seen = (emt.dropna(subset=["player_id"])
                    .groupby(["player_id","player"], as_index=False)["tmin"]
                    .min().sort_values("tmin").head(11))
    return first_seen["player_id"].astype(int).tolist(), match_id


def vectorize_lineup(player_ids, feature_names, mapping, role_lookup, is_home=1):
    # Dedup & ints
    pids, seen = [], set()
    for p in player_ids:
        try:
            ip = int(p)
        except Exception:
            continue
        if ip not in seen:
            seen.add(ip); pids.append(ip)

    P_cols = [c for c in feature_names if c.startswith("P_")]
    pid_to_col = {int(c.split("_",1)[1]): i for i, c in enumerate(P_cols)}
    row_p = np.zeros(len(P_cols), dtype=float)
    for pid in pids:
        if pid in pid_to_col: row_p[pid_to_col[pid]] = 1.0

    R_cols = [c for c in feature_names if c.startswith("R_")]
    rid_to_col = {int(c.split("_",1)[1]): i for i, c in enumerate(R_cols)}
    row_r = np.zeros(len(R_cols), dtype=float)
    for pid in pids:
        rid = role_lookup.get(pid, None)
        if pd.notna(rid):
            rid = int(rid)
            if rid in rid_to_col: row_r[rid_to_col[rid]] += 1.0

    row_c = np.array([float(is_home)], dtype=float)

    parts = []
    if len(P_cols): parts.append(pd.DataFrame([row_p], columns=P_cols))
    if len(R_cols): parts.append(pd.DataFrame([row_r], columns=R_cols))
    parts.append(pd.DataFrame([row_c], columns=["is_home"]))
    X = pd.concat(parts, axis=1)
    return X.reindex(columns=feature_names, fill_value=0.0)

def score_lineup_components(xi, rapm_for: Pipeline, rapm_against: Pipeline,
                            feature_names, mapping, role_lookup, is_home=1):
    X = vectorize_lineup(xi, feature_names, mapping, role_lookup, is_home=is_home)
    xg_for_hat = float(rapm_for.predict(X.values)[0])
    xg_against_hat = float(rapm_against.predict(X.values)[0])
    return xg_for_hat, xg_against_hat, xg_for_hat - xg_against_hat

def delta_swap_components(current, out_pid, in_pid, rapm_for, rapm_against,
                          feature_names, mapping, role_lookup, is_home=1):
    try:
        out_pid = int(out_pid); in_pid = int(in_pid)
    except Exception:
        return np.nan, np.nan, np.nan
    if out_pid == in_pid:
        return 0.0, 0.0, 0.0
    before_f, before_a, _ = score_lineup_components(current, rapm_for, rapm_against, feature_names, mapping, role_lookup, is_home)
    new_lineup = [pid for pid in current if int(pid) != out_pid] + [in_pid]
    after_f, after_a, _ = score_lineup_components(new_lineup, rapm_for, rapm_against, feature_names, mapping, role_lookup, is_home)
    return after_f - before_f, after_a - before_a, (after_f - after_a) - (before_f - before_a)


# =========================
# Pitch helpers (axis-swapped vertical)
# =========================
def primary_position_for_pid(pid: int, players_meta: pd.DataFrame, clusters_df: pd.DataFrame) -> str:
    row = players_meta.loc[players_meta["player_id"] == pid]
    if not row.empty and pd.notna(row.iloc[0].get("primary_pos")):
        return str(row.iloc[0]["primary_pos"])
    row2 = clusters_df.loc[clusters_df["player_id"] == pid]
    if not row2.empty and pd.notna(row2.iloc[0].get("primary_grouped_pos")):
        return str(row2.iloc[0]["primary_grouped_pos"])
    return "Center Midfield"

def coords_for_xi(xi: list[int], players_meta: pd.DataFrame, clusters_df: pd.DataFrame):
    pos_to_pids: dict[str, list[int]] = {}
    for pid in xi[:11]:
        pos = primary_position_for_pid(pid, players_meta, clusters_df)
        pos = pos if pos in POS_ANCHORS else "Center Midfield"
        pos_to_pids.setdefault(pos, []).append(int(pid))

    entries = []
    for pos, pids in pos_to_pids.items():
        ax, ay = POS_ANCHORS[pos]
        offs = DUP_OFFSETS.get(len(pids))
        if not offs:
            step = 6; k = len(pids); start = -step*(k//2)
            offs = [start + i*step for i in range(k)]
        for i, pid in enumerate(pids):
            name = players_meta.loc[players_meta["player_id"] == pid, "player"]
            name = name.iloc[0] if not name.empty else str(pid)
            entries.append({"player_id": pid, "name": name, "pos": pos, "x": ax + offs[i], "y": ay})
    return entries

def pitch_label_name(full_name: str) -> str:
    """
    Pitch labels:
      - 4+ words  -> first initial + 3rd word
      - 2–3 words -> first initial + 2nd word
      - 1 word    -> the word itself
    """
    if not isinstance(full_name, str):
        return ""
    parts = [p for p in full_name.strip().split() if p]
    if not parts:
        return ""
    first_initial = parts[0][0].upper() + "."
    if len(parts) >= 4:
        return f"{first_initial} {parts[2]}"
    elif len(parts) >= 2:
        return f"{first_initial} {parts[1]}"
    else:
        return parts[0]

def draw_pitch_with_xi(xi: list[int], base_team: str,
                       players_meta: pd.DataFrame, clusters_df: pd.DataFrame):
    """Vertical StatsBomb pitch with axis swap: show formation going up (GK bottom, CF top)."""
    if not MPLSOCCER_OK:
        st.info("Install `mplsoccer` to render the pitch:  pip install mplsoccer")
        return

    color = TEAM_COLORS.get(base_team, "#1f77b4")
    pitch = VerticalPitch(pitch_type='statsbomb', half=False,
                          pad_bottom=2, pad_top=2, pad_left=2, pad_right=2)
    fig, ax = pitch.draw(figsize=(6.2, 10))

    entries = coords_for_xi(xi, players_meta, clusters_df)

    # StatsBomb dims: width=80 (x), length=120 (y)
    xs = [80 - e["x"] for e in entries]  # horizontal flip to put RB on viewer's right
    ys = [e["y"] for e in entries]

    # Axis swap: plot (y, x)
    pitch.scatter(ys, xs, s=290, c=color, edgecolors='black', linewidth=1.1, ax=ax, zorder=3)
    for e, x, y in zip(entries, ys, xs):
        name_short = pitch_label_name(e["name"])
        pitch.annotate(f"{name_short} (id {e['player_id']})", (x-5, y),
                       ax=ax, ha='center', fontsize=9, zorder=4)
        pitch.annotate(e["pos"], (x-8, y),
                       ax=ax, ha='center', fontsize=7, color="#444", zorder=4)

    st.pyplot(fig, use_container_width=True)


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Team & Filters")
    base_team = st.selectbox("Baseline team (latest XI):", options=LIGA_MX_TEAMS, index=LIGA_MX_TEAMS.index("América"))
    latest_team_filter = ["(All)"] + LIGA_MX_TEAMS
    sel_team = st.selectbox("Filter candidate pool by latest team:", latest_team_filter, index=0)
    allow_cross_pos = st.checkbox("Allow cross-position replacements", value=False)
    st.markdown("---")
    is_home = st.selectbox("Match context", options=["Home","Away"], index=0)
    is_home_flag = 1 if is_home == "Home" else 0
    st.markdown("---")
    if st.button("Load latest XI for selected team"):
        try:
            xi_default_btn, _ = get_last_starting_xi_ids(events_min, base_team)
        except Exception:
            xi_default_btn = []
        st.session_state["xi_current"] = xi_default_btn
        st.rerun()


# =========================
# App state: authoritative XI + baseline
# =========================
if "xi_current" not in st.session_state:
    try:
        _xi0, _ = get_last_starting_xi_ids(events_min, team_name=base_team)
        st.session_state["xi_current"] = _xi0
    except Exception:
        st.session_state["xi_current"] = []

if "xi_baseline" not in st.session_state:
    st.session_state["xi_baseline"] = list(st.session_state["xi_current"])

def _parse_xi_text_to_list(xi_text: str) -> list[int]:
    out, seen = [], set()
    for tok in xi_text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            val = int(tok)
            if val not in seen:
                seen.add(val); out.append(val)
        except Exception:
            pass
    return out

def _on_xi_text_change():
    st.session_state["xi_current"] = _parse_xi_text_to_list(st.session_state["xi_text"])


# =========================
# Baseline XI text input (drives xi_current via callback)
# =========================
xi_text_default = ",".join(map(str, st.session_state["xi_current"]))
st.text_input(
    "Edit XI (comma-separated player_ids)",
    value=xi_text_default,
    key="xi_text",
    on_change=_on_xi_text_change
)
xi = list(st.session_state["xi_current"])


# =========================
# XI list + Pitch + Inline swap (with rankings)
# =========================
toprow = st.columns([1.0, 1.0, 2.0])
with toprow[0]:
    if st.button("💾 Save baseline (use current XI)"):
        st.session_state["xi_baseline"] = list(st.session_state["xi_current"])
        st.success("Baseline saved.", icon="✅")
with toprow[1]:
    if st.button("↩️ Reset to baseline"):
        st.session_state["xi_current"] = list(st.session_state["xi_baseline"])
        st.rerun()

st.markdown("### Starting XI")

# Build ordered XI table
xi_df = pd.DataFrame({"player_id": xi}).merge(
    players_meta[["player_id","player","latest_team","primary_pos","secondary_pos","third_pos"]],
    on="player_id", how="left"
)
xi_df = xi_df.merge(clusters_df[["player_id","primary_grouped_pos"]], on="player_id", how="left")
xi_df["main_pos"] = xi_df["primary_pos"].fillna(xi_df["primary_grouped_pos"])

pos_order = {p:i for i,p in enumerate(GROUPED_POSITIONS)}
xi_df["pos_rank"] = xi_df["main_pos"].map(pos_order).fillna(99)
xi_df = xi_df.sort_values(["pos_rank","player"]).reset_index(drop=True)

col_pitch, col_list = st.columns([1.6, 1.4], gap="large")

with col_pitch:
    draw_pitch_with_xi(xi, base_team, players_meta, clusters_df)

with col_list:
    st.markdown("**Swap any player directly from this list**")

    # Build base candidate pool once
    base_cands = (players_meta[["player_id","player","latest_team","primary_pos","secondary_pos","third_pos"]]
        .merge(clusters_df[["player_id","primary_grouped_pos","pos_cluster_id","pos_cluster_label"]],
               on="player_id", how="left"))
    known_players = set(map(int, mapping.get("players", [])))
    base_cands = base_cands[base_cands["player_id"].isin(known_players)].copy()
    if sel_team != "(All)":
        base_cands = base_cands[base_cands["latest_team"] == sel_team]

    def badge(team): return TEAM_BADGE.get(team, "⚪")

    for _, r in xi_df.iterrows():
        pid = int(r["player_id"])
        pname = r["player"] if pd.notna(r["player"]) else str(pid)
        mpos = r["main_pos"] if pd.notna(r["main_pos"]) else "Unknown"

        st.write(f"**{pname} (id {pid})** — {mpos}")

        # Filter candidates by position unless cross-position is allowed
        if (not allow_cross_pos) and (mpos in GROUPED_POSITIONS):
            cand = base_cands[base_cands["primary_grouped_pos"] == mpos].copy()
        else:
            cand = base_cands.copy()

        # Remove the same player from candidates
        cand = cand[cand["player_id"] != pid].copy()

        if cand.empty:
            st.caption("_No candidates match current filters._")
            st.markdown("---")
            continue

        cand["label"] = cand.apply(
            lambda x: f"{badge(x['latest_team'])}  {x['player']} (id {int(x['player_id'])})  "
                      f"[{x['latest_team']}] — main: {x['primary_pos']} | alt: {x['secondary_pos']}/{x['third_pos']}",
            axis=1
        )
        sel = st.selectbox("Replacement", options=["(choose)"] + cand["label"].tolist(),
                           key=f"swap_sel_{pid}", label_visibility="collapsed")

        if sel != "(choose)":
            in_pid = int(sel.split("(id")[1].split(")")[0])

            # Show deltas for the chosen player
            dF, dA, dD = delta_swap_components(
                current=xi, out_pid=pid, in_pid=in_pid,
                rapm_for=rapm_for, rapm_against=rapm_against,
                feature_names=feature_names, mapping=mapping, role_lookup=role_lookup,
                is_home=is_home_flag
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Δ xG For", f"{dF:+.2f}")
            c2.metric("Δ xG Against", f"{dA:+.2f}")
            c3.metric("Δ ΔxG", f"{dD:+.2f}")

            # Button to apply the swap -> update xi_current only, then rerun
            if st.button("Apply swap", key=f"apply_{pid}_{in_pid}"):
                new_xi = [int(x) for x in xi if int(x) != pid] + [in_pid]
                st.session_state["xi_current"] = new_xi
                st.rerun()

            # Rankings for this outgoing player (pool honors current filters)
            with st.expander("Show candidate rankings for this swap"):
                pool = cand.copy()
                results = []
                for p in pool["player_id"].tolist():
                    DF, DA, DD = delta_swap_components(
                        current=xi, out_pid=pid, in_pid=int(p),
                        rapm_for=rapm_for, rapm_against=rapm_against,
                        feature_names=feature_names, mapping=mapping, role_lookup=role_lookup,
                        is_home=is_home_flag
                    )
                    results.append({"player_id": int(p), "delta_xg_for": DF, "delta_xg_against": DA, "delta_deltaxg": DD})
                res_df = pd.DataFrame(results)
                ranked = (pool.merge(res_df, on="player_id", how="left")
                              .sort_values(["delta_deltaxg","delta_xg_for"], ascending=[False, False])
                              .reset_index(drop=True))
                st.dataframe(
                    ranked[["player_id","player","latest_team","primary_pos","secondary_pos","third_pos",
                            "primary_grouped_pos","pos_cluster_label","delta_xg_for","delta_xg_against","delta_deltaxg"]],
                    use_container_width=True, height=360
                )

        st.markdown("---")
