# app.py
# Liga MX — Dynamic xG Lineup (team-agnostic, artifact-driven)

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

# =========================
# Config / Paths
# =========================
ARTIFACTS_DIR = Path("./artifacts")  # change if needed

FILES = {
    "players_meta": ARTIFACTS_DIR / "players_meta.parquet",         # player_id, player, latest_team, primary_pos, secondary_pos, third_pos
    "clusters": ARTIFACTS_DIR / "clusters_team_agnostic.parquet",   # player_id, player, primary_grouped_pos, games_in_pos, pos_cluster_id, pos_cluster_label
    "role_lookup": ARTIFACTS_DIR / "role_lookup.parquet",           # player_id, pos_cluster_id
    "mapping": ARTIFACTS_DIR / "mapping.json",                      # {"players":[...], "roles":[...]}
    "feature_names": ARTIFACTS_DIR / "feature_names.json",          # ["P_123", "R_1", ..., "is_home"]
    "rapm_for": ARTIFACTS_DIR / "rapm_for.pkl",                     # sklearn Pipeline (StandardScaler+RidgeCV)
    "rapm_against": ARTIFACTS_DIR / "rapm_against.pkl",             # sklearn Pipeline
    "events_min": ARTIFACTS_DIR / "events_min.parquet",             # minimal events for latest XI extraction
}

LIGA_MX_TEAMS = [
    'Guadalajara', 'Tigres UANL', 'Monterrey', 'León', 'Pachuca',
    'Pumas UNAM', 'Atlas', 'Santos Laguna', 'Cruz Azul', 'Tijuana',
    'América', 'Juárez', 'Puebla', 'Toluca', 'Mazatlán',
    'Atlético San Luis', 'Querétaro', 'Necaxa'
]

# =========================
# Caching loaders
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

# =========================
# Helpers
# =========================
def _parse_datetime_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def get_last_starting_xi_ids(events_min: pd.DataFrame, team_name: str):
    """
    Returns ([player_id,... up to 11], match_id) for the team's most recent match.
    Prefers 'Starting XI' with tactics.lineup; falls back to first 11 to appear.
    """
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

    # Prefer Starting XI
    xi_rows = emt.loc[emt["type"] == "Starting XI"]
    if not xi_rows.empty and "tactics" in xi_rows.columns:
        tac = xi_rows.iloc[0].get("tactics", {})
        lineup = tac.get("lineup", []) if isinstance(tac, dict) else []
        ids = [p.get("player_id") for p in lineup if isinstance(p, dict) and p.get("player_id") is not None]
        if len(ids) >= 10:
            return [int(x) for x in ids[:11]], match_id

    # Fallback: earliest 11 to appear
    emt["tmin"] = emt["minute"].fillna(0).astype(float) + emt["second"].fillna(0).astype(float)/60.0
    first_seen = (emt.dropna(subset=["player_id"])
                    .groupby(["player_id","player"], as_index=False)["tmin"]
                    .min().sort_values("tmin").head(11))
    return first_seen["player_id"].astype(int).tolist(), match_id

def vectorize_lineup(player_ids, feature_names, mapping, role_lookup, is_home=1):
    """Builds a one-row DataFrame matching training feature order (players + role-mix + control)."""
    import numpy as np
    import pandas as pd

    # Dedup & sanitize
    pids = []
    seen = set()
    for p in player_ids:
        try:
            ip = int(p)
        except Exception:
            continue
        if ip not in seen:
            seen.add(ip)
            pids.append(ip)

    # Player dummies
    P_cols = [c for c in feature_names if c.startswith("P_")]
    pid_to_col = {int(c.split("_",1)[1]): i for i, c in enumerate(P_cols)}
    row_p = np.zeros(len(P_cols), dtype=float)
    for pid in pids:
        if pid in pid_to_col:
            row_p[pid_to_col[pid]] = 1.0

    # Role-mix counts
    R_cols = [c for c in feature_names if c.startswith("R_")]
    rid_to_col = {int(c.split("_",1)[1]): i for i, c in enumerate(R_cols)}
    row_r = np.zeros(len(R_cols), dtype=float)
    for pid in pids:
        rid = role_lookup.get(pid, None)
        if pd.notna(rid):
            rid = int(rid)
            if rid in rid_to_col:
                row_r[rid_to_col[rid]] += 1.0

    # Controls
    row_c = np.array([float(is_home)], dtype=float)

    # Assemble (NOTE the '=' below)
    parts = []
    if len(P_cols): parts.append(pd.DataFrame([row_p], columns=P_cols))
    if len(R_cols): parts.append(pd.DataFrame([row_r], columns=R_cols))  # <-- fixed here
    parts.append(pd.DataFrame([row_c], columns=["is_home"]))
    X = pd.concat(parts, axis=1)

    # Ensure exact training order
    return X.reindex(columns=feature_names, fill_value=0.0)


def score_lineup_components(xi, rapm_for: Pipeline, rapm_against: Pipeline, feature_names, mapping, role_lookup, is_home=1):
    X = vectorize_lineup(xi, feature_names, mapping, role_lookup, is_home=is_home)
    xg_for_hat = float(rapm_for.predict(X.values)[0])
    xg_against_hat = float(rapm_against.predict(X.values)[0])
    return xg_for_hat, xg_against_hat, xg_for_hat - xg_against_hat

def delta_swap_components(current, out_pid, in_pid, rapm_for, rapm_against, feature_names, mapping, role_lookup, is_home=1):
    """Returns (ΔxG_for, ΔxG_against, ΔΔxG). Exact self-swap → zeros."""
    try:
        out_pid = int(out_pid); in_pid = int(in_pid)
    except Exception:
        return np.nan, np.nan, np.nan
    if out_pid == in_pid:
        return 0.0, 0.0, 0.0

    before_f, before_a, before_d = score_lineup_components(current, rapm_for, rapm_against, feature_names, mapping, role_lookup, is_home)
    new_lineup = [pid for pid in current if int(pid) != out_pid] + [in_pid]
    after_f, after_a, after_d = score_lineup_components(new_lineup, rapm_for, rapm_against, feature_names, mapping, role_lookup, is_home)
    return after_f - before_f, after_a - before_a, after_d - before_d

# =========================
# App
# =========================
st.set_page_config(page_title="Liga MX — Dynamic xG Lineup", layout="wide")
st.title("Liga MX — Dynamic xG Lineup (team-agnostic, artifact-driven)")

# ---- Validate artifacts exist
missing = [name for name, p in FILES.items() if not p.exists()]
if missing:
    st.error(f"Missing artifact files: {missing}\n\nRun your exporter to generate them.")
    st.stop()

# ---- Load artifacts (cached)
players_meta = load_parquet(FILES["players_meta"])
clusters_df  = load_parquet(FILES["clusters"])
role_lookup_df = load_parquet(FILES["role_lookup"])
role_lookup = role_lookup_df.set_index("player_id")["pos_cluster_id"]
mapping = load_json(FILES["mapping"])
feature_names = load_json(FILES["feature_names"])
rapm_for = load_pickle(FILES["rapm_for"])
rapm_against = load_pickle(FILES["rapm_against"])
events_min = load_parquet(FILES["events_min"])

# ---- Sidebar: choose baseline team & candidate filter
with st.sidebar:
    st.header("Team & Filters")

    base_team = st.selectbox("Baseline team (latest XI):", options=LIGA_MX_TEAMS, index=LIGA_MX_TEAMS.index("América"))
    st.caption("Baseline XI will auto-load from the most recent match (Starting XI if available).")

    latest_team_filter = ["(All)"] + LIGA_MX_TEAMS
    sel_team = st.selectbox("Filter replacement candidates by latest team:", latest_team_filter, index=0)

    st.markdown("---")
    st.header("Options")
    show_ids = st.checkbox("Show XI IDs", value=False)
    is_home = st.selectbox("Match context", options=["Home","Away"], index=0)
    is_home_flag = 1 if is_home == "Home" else 0

# ---- Baseline XI for chosen team
@st.cache_data(show_spinner=False)
def _baseline_xi(events_df: pd.DataFrame, team_name: str):
    return get_last_starting_xi_ids(events_df, team_name=team_name)

try:
    xi_default, baseline_match_id = _baseline_xi(events_min, base_team)
except Exception as e:
    st.error(f"Failed to build baseline XI for {base_team}: {e}")
    st.stop()

# ---- Top banner with baseline scores
st.subheader(f"Baseline XI — {base_team}")
if show_ids:
    st.code(xi_default)

# Allow user edit XI (comma-separated)
xi_text = st.text_area("Edit XI (comma-separated player_ids)", value=",".join(map(str, xi_default)), height=80)
try:
    xi = [int(x.strip()) for x in xi_text.split(",") if x.strip()]
except Exception:
    st.warning("Invalid XI format — using baseline XI.")
    xi = xi_default

base_for, base_against, base_delta = score_lineup_components(
    xi, rapm_for, rapm_against, feature_names, mapping, role_lookup, is_home=is_home_flag
)

m1, m2, m3 = st.columns(3)
m1.metric("Baseline xG For", f"{base_for:.2f}")
m2.metric("Baseline xG Against", f"{base_against:.2f}")
m3.metric("Baseline ΔxG", f"{base_delta:.2f}")

st.markdown("---")

# ---- Candidate pool (filterable)
meta_cols = ["player_id","player","latest_team","primary_pos","secondary_pos","third_pos"]
candidates = players_meta[meta_cols].merge(
    clusters_df[["player_id","primary_grouped_pos","pos_cluster_id","pos_cluster_label"]],
    on="player_id", how="left"
)

if sel_team != "(All)":
    candidates = candidates[candidates["latest_team"] == sel_team]

# Only players the models know (columns exist in trained design)
known_players = set(map(int, mapping.get("players", [])))
candidates = candidates[candidates["player_id"].isin(known_players)].reset_index(drop=True)

# ---- OUT selector (from current XI)
st.subheader("Swap a player")

c1, c2, c3 = st.columns([2,2,3], gap="large")

with c1:
    st.markdown("**Current XI — pick OUT**")
    xi_df = pd.DataFrame({"player_id": xi}).merge(
        players_meta[["player_id","player","latest_team","primary_pos","secondary_pos","third_pos"]],
        on="player_id", how="left"
    )
    if xi_df.empty:
        st.error("Current XI is empty.")
        st.stop()
    out_labels = xi_df.apply(lambda r: f'{int(r.player_id)} — {r.player or "unknown"} ({r.primary_pos or "?"})', axis=1).tolist()
    out_choice = st.selectbox("Player to remove:", options=out_labels, index=0)
    try:
        out_pid = int(out_choice.split("—")[0].strip())
    except Exception:
        out_pid = xi[0]

with c2:
    st.markdown("**Replacement filter**")
    pos_opts = ["(Any)"] + sorted(candidates["primary_grouped_pos"].dropna().unique().tolist())
    pos_sel = st.selectbox("Grouped position:", options=pos_opts, index=0)

with c3:
    st.markdown("**Pick replacement**")
    cand_view = candidates.copy()
    if pos_sel != "(Any)":
        cand_view = cand_view[cand_view["primary_grouped_pos"] == pos_sel]

    if cand_view.empty:
        st.warning("No candidates match the current filters.")
        in_pid = None
    else:
        cand_view["label"] = cand_view.apply(
            lambda r: f'{int(r.player_id)} — {r.player} [{r.latest_team}] | main: {r.primary_pos} | alt: {r.secondary_pos}/{r.third_pos}', axis=1
        )
        in_label = st.selectbox("Replacement player:", options=cand_view["label"].tolist(), index=0)
        try:
            in_pid = int(in_label.split("—")[0].strip())
        except Exception:
            in_pid = None

st.markdown("---")

# ---- Run swap & show results
if out_pid is not None and in_pid is not None:
    d_for, d_against, d_delta = delta_swap_components(
        current=xi, out_pid=out_pid, in_pid=in_pid,
        rapm_for=rapm_for, rapm_against=rapm_against,
        feature_names=feature_names, mapping=mapping, role_lookup=role_lookup, is_home=is_home_flag
    )

    r1, r2, r3 = st.columns(3)
    r1.metric("Δ xG For", f"{d_for:+.2f}")
    r2.metric("Δ xG Against", f"{d_against:+.2f}")
    r3.metric("Δ ΔxG", f"{d_delta:+.2f}")

    st.markdown("### Player cards")
    lcol, rcol = st.columns(2)

    out_row = players_meta.loc[players_meta["player_id"] == out_pid].merge(
        clusters_df[["player_id","primary_grouped_pos","pos_cluster_label"]],
        on="player_id", how="left"
    )
    in_row = players_meta.loc[players_meta["player_id"] == in_pid].merge(
        clusters_df[["player_id","primary_grouped_pos","pos_cluster_label"]],
        on="player_id", how="left"
    )

    with lcol:
        st.markdown("**Out**")
        if not out_row.empty:
            r = out_row.iloc[0]
            st.write(
                f"**{int(r.player_id)} — {r.player}**  \n"
                f"Team: {r.latest_team}  \n"
                f"Positions: {r.primary_pos} | {r.secondary_pos} | {r.third_pos}  \n"
                f"Grouped: {r.primary_grouped_pos}  \n"
                f"Cluster: {r.pos_cluster_label}"
            )
        else:
            st.write("Not found in meta.")

    with rcol:
        st.markdown("**In**")
        if not in_row.empty:
            r = in_row.iloc[0]
            st.write(
                f"**{int(r.player_id)} — {r.player}**  \n"
                f"Team: {r.latest_team}  \n"
                f"Positions: {r.primary_pos} | {r.secondary_pos} | {r.third_pos}  \n"
                f"Grouped: {r.primary_grouped_pos}  \n"
                f"Cluster: {r.pos_cluster_label}"
            )
        else:
            st.write("Not found in meta.")

# ---- Optional: rank all replacements quickly
st.markdown("---")
with st.expander("Batch rank candidates for this XI (fast)"):
    batch_pos = st.selectbox("Candidate position (grouped):", options=["(All)"] + sorted(candidates["primary_grouped_pos"].dropna().unique()))
    pool = candidates.copy()
    if sel_team != "(All)":
        pool = pool[pool["latest_team"] == sel_team]
    if batch_pos != "(All)":
        pool = pool[pool["primary_grouped_pos"] == batch_pos]
    pool = pool[pool["player_id"].isin(known_players)].copy()

    if st.button("Compute Δ for pool"):
        results = []
        for pid in pool["player_id"].tolist():
            dF, dA, dD = delta_swap_components(
                current=xi, out_pid=out_pid, in_pid=int(pid),
                rapm_for=rapm_for, rapm_against=rapm_against,
                feature_names=feature_names, mapping=mapping, role_lookup=role_lookup, is_home=is_home_flag
            )
            results.append({"player_id": int(pid), "delta_xg_for": dF, "delta_xg_against": dA, "delta_deltaxg": dD})
        res_df = pd.DataFrame(results)
        out = (pool.merge(res_df, on="player_id", how="left")
                    .sort_values(["delta_deltaxg","delta_xg_for"], ascending=[False, False])
                    .reset_index(drop=True))
        st.dataframe(out[["player_id","player","latest_team","primary_pos","secondary_pos","third_pos",
                          "primary_grouped_pos","pos_cluster_label","delta_xg_for","delta_xg_against","delta_deltaxg"]],
                     use_container_width=True)

st.caption("All numbers are model estimates. Use comparatively.")
