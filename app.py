# app.py
# Liga MX — Dynamic xG Lineup (artifact-driven)
# Vertical pitch (axis-swapped), inline swaps (two-column), cross-position toggle,
# per-swap rankings, baseline save/reset, Plotly totals bars with auto axis,
# safe session_state syncing (IDs box), team-color dots for swapped-in players,
# BIG centered current-vs-baseline delta banner, NO inline (per-select) deltas.

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

# Pitch (optional)
try:
    from mplsoccer import VerticalPitch
    MPLSOCCER_OK = True
except Exception:
    MPLSOCCER_OK = False

import plotly.graph_objects as go

from typing import List, Dict, Optional
from datetime import datetime

# network effects bundle
try:
    from modules.network_effects import (
        predict_transfer_bundle,   # the bundle we wrote
    )
    NETWORK_OK = True
except Exception:
    NETWORK_OK = False



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
    "rw_profiles": ARTIFACTS_DIR / "rw_per90_team_agnostic.parquet"   # <-- NEW

}

LIGA_MX_TEAMS = [
    'Guadalajara', 'Tigres UANL', 'Monterrey', 'León', 'Pachuca',
    'Pumas UNAM', 'Atlas', 'Santos Laguna', 'Cruz Azul', 'Tijuana',
    'América', 'Juárez', 'Puebla', 'Toluca', 'Mazatlán',
    'Atlético San Luis', 'Querétaro', 'Necaxa'
]

# ---------- COLORS / BADGES ----------
TEAM_COLORS = {
    "América": "#fbd000",
    "Cruz Azul": "#0033a0",
    "Guadalajara": "#c8102e",
    "Tigres UANL": "#ffb612",
    "Monterrey": "#0a2342",
    "León": "#006c3b",
    "Pachuca": "#1f4e79",
    "Pumas UNAM": "#ffffff",     # WHITE
    "Atlas": "#e43b3b",
    "Santos Laguna": "#1db954",
    "Tijuana": "#c8102e",
    "Juárez": "#5bbf21",
    "Puebla": "#ffffff",         # WHITE
    "Toluca": "#c8102e",
    "Mazatlán": "#6a1b9a",
    "Atlético San Luis": "#c8102e",  # RED
    "Querétaro": "#1f4e79",
    "Necaxa": "#ff2a2a",
}
TEAM_BADGE = {
    "América": "🟡", "Cruz Azul": "🔵", "Guadalajara": "🔴", "Tigres UANL": "🟠",
    "Monterrey": "🔵", "León": "🟢", "Pachuca": "🔵",
    "Pumas UNAM": "⚪",   # WHITE
    "Atlas": "🔴", "Santos Laguna": "🟢", "Tijuana": "🔴", "Juárez": "🟢",
    "Puebla": "⚪",       # WHITE
    "Toluca": "🔴", "Mazatlán": "🟣",
    "Atlético San Luis": "🔴",  # RED
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

# Anchors for a 120x80 vertical StatsBomb pitch (we axis-swap when plotting)
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

# ===========================================
# Tabs: Lineup (existing) | Player Comparison (new)
# ===========================================
tab1, tab2, tab3 = st.tabs(["xG Impact", "Player Comparison", "Network Effects"])
st.set_page_config(page_title="Player Scouting Tool", layout="wide")

with tab1:
    # =========================
    # Streamlit setup
    # =========================
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
    rw_profiles = load_parquet(FILES["rw_profiles"])

        # === Age & Country helpers (after loading players_meta) ===
    from datetime import date

    COUNTRY_TO_ALPHA2 = {
        "Mexico":"MX","Argentina":"AR","Brazil":"BR","Uruguay":"UY","Paraguay":"PY",
        "Chile":"CL","Colombia":"CO","Ecuador":"EC","Peru":"PE","United States":"US",
        "Spain":"ES","France":"FR","Portugal":"PT","Italy":"IT","Germany":"DE",
        "Netherlands":"NL","Belgium":"BE","Poland":"PL","Denmark":"DK","Sweden":"SE",
        "Norway":"NO","Finland":"FI","Japan":"JP","Korea Republic":"KR","Canada":"CA",
        "England":"GB","Scotland":"GB","Wales":"GB","Northern Ireland":"GB",
        "Croatia":"HR","Serbia":"RS","Bosnia and Herzegovina":"BA","Slovenia":"SI",
        "Czech Republic":"CZ","Slovakia":"SK","Switzerland":"CH","Austria":"AT",
        "Greece":"GR","Turkey":"TR","Morocco":"MA","Algeria":"DZ","Tunisia":"TN",
        "Nigeria":"NG","Ghana":"GH","Cameroon":"CM","Ivory Coast":"CI","Senegal":"SN",
        "Costa Rica":"CR","Honduras":"HN","Guatemala":"GT","El Salvador":"SV",
        "Panama":"PA","Venezuela":"VE","Bolivia":"BO","Armenia":"AM","Georgia":"GE",
    }

    def country_flag(country: str) -> str:
        if not isinstance(country, str) or not country.strip():
            return "🏳️"
        code = COUNTRY_TO_ALPHA2.get(country.strip(), None)
        if not code:
            return "🏳️"
        # regional indicator letters
        return "".join(chr(127397 + ord(c)) for c in code.upper())

    def compute_age_from_birthdate(s: pd.Series, ref: date = date.today()) -> pd.Series:
        bd = pd.to_datetime(s, errors="coerce", utc=True).dt.date
        days = (pd.Series(ref, index=bd.index) - bd).dt.days
        return np.floor(days / 365.2425).astype("Int64")

    # Ensure players_meta has age and a displayable country flag label
    if "age" not in players_meta.columns or players_meta["age"].isna().all():
        if "birth_date" in players_meta.columns:
            players_meta["age"] = compute_age_from_birthdate(players_meta["birth_date"])
        else:
            players_meta["age"] = pd.Series([pd.NA]*len(players_meta), dtype="Int64")

    # Normalize country string & build "flag + name" label
    if "country" not in players_meta.columns:
        players_meta["country"] = pd.NA
    players_meta["country"] = players_meta["country"].astype("string")
    players_meta["country_flag"] = players_meta["country"].map(country_flag)
    players_meta["country_label"] = players_meta.apply(
        lambda r: f"{r['country_flag']} {r['country']}" if pd.notna(r["country"]) else "🏳️ —",
        axis=1
    )


    # ---- Network Effects artifacts (optional; soft-load) ----
    try:
        rw_profiles = pd.read_parquet(ARTIFACTS_DIR / "rw_per90_team_agnostic.parquet")
        team_style  = pd.read_parquet(ARTIFACTS_DIR / "team_style.parquet")
        team_role_mix = pd.read_parquet(ARTIFACTS_DIR / "team_role_mix.parquet")

        with open(ARTIFACTS_DIR / "network_feature_cols.json", "r", encoding="utf-8") as f:
            network_feature_cols = json.load(f)
        with open(ARTIFACTS_DIR / "network_target_cols.json", "r", encoding="utf-8") as f:
            network_target_cols = json.load(f)
        with open(ARTIFACTS_DIR / "network_models.pkl", "rb") as f:
            network_models = pickle.load(f)

        NETWORK_ARTIFACTS_OK = True
    except Exception:
        NETWORK_ARTIFACTS_OK = False


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
    # Utilities for colors and names
    # =========================
    def get_latest_team(pid: int) -> str:
        row = players_meta.loc[players_meta["player_id"] == pid]
        if not row.empty and pd.notna(row.iloc[0].get("latest_team")):
            return str(row.iloc[0]["latest_team"])
        return ""

    def color_for_player(pid: int, base_team: str, baseline_set: set[int]) -> str:
        if pid not in baseline_set:
            team = get_latest_team(pid)
            return TEAM_COLORS.get(team, "#888888")
        return TEAM_COLORS.get(base_team, "#1f77b4")


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
                        players_meta: pd.DataFrame, clusters_df: pd.DataFrame,
                        baseline_xi: list[int]):
        """Vertical StatsBomb pitch with axis swap: GK bottom, CF top.
        Swapped-in players (not in baseline) are colored by their latest team."""
        if not MPLSOCCER_OK:
            st.info("Install `mplsoccer` to render the pitch:  pip install mplsoccer")
            return

        baseline_set = set(int(p) for p in baseline_xi)

        # Build entries and per-player colors
        entries = coords_for_xi(xi, players_meta, clusters_df)
        colors = [color_for_player(e["player_id"], base_team, baseline_set) for e in entries]

        # Draw pitch
        pitch = VerticalPitch(pitch_type='statsbomb', half=False,
                            pad_bottom=2, pad_top=2, pad_left=2, pad_right=2)
        fig, ax = pitch.draw(figsize=(6.6, 10))

        # StatsBomb dims: width=80 (x), length=120 (y)
        xs = [80 - e["x"] for e in entries]  # horizontal flip to put RB on viewer's right
        ys = [e["y"] for e in entries]

        # Axis swap: plot (y, x)
        pitch.scatter(ys, xs, s=290, c=colors, edgecolors='black', linewidth=1.1, ax=ax, zorder=3)
        for e, x, y in zip(entries, ys, xs):
            name_short = pitch_label_name(e["name"])
            pitch.annotate(f"{name_short} (id {e['player_id']})", (x-5, y),
                        ax=ax, ha='center', fontsize=9, zorder=4)
            pitch.annotate(e["pos"], (x-8, y),
                        ax=ax, ha='center', fontsize=7, color="#444", zorder=4)

        st.pyplot(fig)


    # =========================
    # Position normalization (so filters actually work)
    # =========================
    _POS_MAP = {
        "GK": "Goalkeeper", "GOALKEEPER": "Goalkeeper",

        "RB": "Right Back", "RWB": "Right Back", "RIGHT BACK": "Right Back",

        "CB": "Center Back", "RCB": "Center Back", "LCB": "Center Back",
        "CENTER BACK": "Center Back", "CENTRE BACK": "Center Back",

        "LB": "Left Back", "LWB": "Left Back", "LEFT BACK": "Left Back",

        "CDM": "Defensive Midfield", "DM": "Defensive Midfield",
        "DEFENSIVE MIDFIELD": "Defensive Midfield",

        "CM": "Center Midfield", "MC": "Center Midfield",
        "CENTER MIDFIELD": "Center Midfield", "CENTRE MIDFIELD": "Center Midfield",

        "CAM": "Attacking Midfield", "AM": "Attacking Midfield",
        "ATTACKING MIDFIELD": "Attacking Midfield",

        "RW": "Right Wing", "RM": "Right Wing", "RIGHT WING": "Right Wing",

        "LW": "Left Wing", "LM": "Left Wing", "LEFT WING": "Left Wing",

        "CF": "Center Forward", "ST": "Center Forward", "FW": "Center Forward",
        "STRIKER": "Center Forward", "CENTER FORWARD": "Center Forward",
        "CENTRE FORWARD": "Center Forward",
    }

    def _to_grouped(v: str) -> str | None:
        if pd.isna(v):
            return None
        s = str(v).strip()
        if not s:
            return None
        if s in GROUPED_POSITIONS:
            return s
        return _POS_MAP.get(s.upper(), None)


    # =========================
    # Sidebar (team dropdown with color, alpha-sorted; auto-reset baseline/current)
    # =========================
    _sorted = sorted(LIGA_MX_TEAMS)
    team_labels = [f"{TEAM_BADGE.get(t, '⚪')}  {t}" for t in _sorted]
    label_to_team = {lbl: t for lbl, t in zip(team_labels, _sorted)}

    with st.sidebar:
        st.header("Team & Filters (only for Lineup tool)")

        # Baseline team
        default_idx = _sorted.index("América")
        sel_label = st.selectbox("Baseline team (latest XI):", options=team_labels, index=default_idx)
        base_team = label_to_team[sel_label]

        # Global team filter
        latest_team_filter = ["(All)"] + _sorted
        sel_team = st.selectbox("Filter candidate pool by latest team:", latest_team_filter, index=0)

        # Positions (multi)
        sel_positions = st.multiselect(
            "Filter candidate pool by positions:",
            options=GROUPED_POSITIONS,
            default=[]
        )

        # Only main position?
        only_main_pos = st.checkbox("Only main position equals filter", value=False)

        st.markdown("#### Demographics")
        # Age slider bounds (ignore NaN)
        _ages = players_meta["age"].dropna().astype(int)
        if _ages.empty:
            age_min, age_max = 15, 45
        else:
            age_min, age_max = int(_ages.min()), int(_ages.max())
        sel_age = st.slider("Age range:", min_value=age_min, max_value=age_max,
                            value=(age_min, age_max), step=1)

        # Country multiselect with emoji flag labels (A→Z)
        _countries = (players_meta[["country","country_flag"]]
                        .dropna()
                        .drop_duplicates()
                        .assign(label=lambda d: d["country_flag"] + " " + d["country"])
                        .sort_values("country", kind="stable"))
        country_labels = _countries["label"].tolist()
        label_to_country = {lbl: c for lbl, c in zip(_countries["label"], _countries["country"])}
        sel_countries_lbl = st.multiselect("Countries:", options=country_labels, default=country_labels)
        sel_countries = {label_to_country[lbl] for lbl in sel_countries_lbl}

                # Merge demographics and filter by age & country

        # ---- PREVIEW POOL to build cluster options coherently ----
        _preview_cands = (
            players_meta[["player_id","player","latest_team","primary_pos","secondary_pos","third_pos"]]
            .merge(
                clusters_df[["player_id","primary_grouped_pos","pos_cluster_id",
                            "pos_cluster_label","pos_cluster_name"]],
                on="player_id", how="left"
            )
        )
        _preview_cands = _preview_cands.merge(
            players_meta[["player_id","age","country","country_flag","country_label"]],
            on="player_id", how="left"
        )
        _preview_cands = _preview_cands[
            _preview_cands["age"].between(sel_age[0], sel_age[1], inclusive="both", na=False)
            & (_preview_cands["country"].isin(sel_countries))
        ]

        # normalized grouped versions of pos fields
        _preview_cands["primary_pos_grouped_norm"]   = _preview_cands["primary_pos"].map(_to_grouped)
        _preview_cands["secondary_pos_grouped_norm"] = _preview_cands["secondary_pos"].map(_to_grouped)
        _preview_cands["third_pos_grouped_norm"]     = _preview_cands["third_pos"].map(_to_grouped)

        

        # team filter
        if sel_team != "(All)":
            _preview_cands = _preview_cands[_preview_cands["latest_team"] == sel_team]

        # positions filter
        if sel_positions:
            if only_main_pos:
                mask_pos_prev = _preview_cands["primary_grouped_pos"].isin(sel_positions)
            else:
                mask_pos_prev = (
                    _preview_cands["primary_grouped_pos"].isin(sel_positions)
                    | _preview_cands["primary_pos_grouped_norm"].isin(sel_positions)
                    | _preview_cands["secondary_pos_grouped_norm"].isin(sel_positions)
                    | _preview_cands["third_pos_grouped_norm"].isin(sel_positions)
                )
            _preview_cands = _preview_cands[mask_pos_prev]

        # cluster column resolution
        cluster_col_preview = "pos_cluster_name" if (
            "pos_cluster_name" in _preview_cands.columns and _preview_cands["pos_cluster_name"].notna().any()
        ) else "pos_cluster_label"

        cluster_options = (
            _preview_cands[cluster_col_preview]
            .dropna().astype(str).sort_values().unique().tolist()
        )

        sel_clusters = st.multiselect(
            "Filter candidate pool by cluster:",
            options=cluster_options,
            default=[]
        )

        allow_cross_pos = st.checkbox("Allow cross-position replacements", value=False)
                # === NEW: Age & Country filters (Lineup tab only) ===


        st.markdown("---")
        is_home = st.selectbox("Match context", options=["Home","Away"], index=0)
        is_home_flag = 1 if is_home == "Home" else 0


    # =========================
    # App state: authoritative XI + baseline (+ queued edit-box sync)
    # =========================
    def reset_to_team_latest(team_name: str):
        try:
            xi_latest, _ = get_last_starting_xi_ids(events_min, team_name)
        except Exception:
            xi_latest = []
        st.session_state["xi_current"] = xi_latest
        st.session_state["xi_baseline"] = list(xi_latest)
        st.session_state["_xi_text_pending"] = ",".join(map(str, xi_latest))

    # Initialize or update when team changes
    if "xi_current" not in st.session_state or "xi_baseline" not in st.session_state:
        reset_to_team_latest(base_team)
    else:
        if st.session_state.get("_team_for_state") != base_team:
            reset_to_team_latest(base_team)
    st.session_state["_team_for_state"] = base_team

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
        # Only updates the active XI (not baseline)
        st.session_state["xi_current"] = _parse_xi_text_to_list(st.session_state["xi_text"])

    # Apply any queued xi_text sync BEFORE rendering the input
    if "_xi_text_pending" in st.session_state:
        st.session_state["xi_text"] = st.session_state.pop("_xi_text_pending")

    # Baseline XI text input (drives xi_current via callback)
    xi_text_default = st.session_state.get("xi_text", ",".join(map(str, st.session_state["xi_current"])))
    st.text_input(
        "Edit XI (comma-separated player_ids) — edits active lineup only",
        value=xi_text_default,
        key="xi_text",
        on_change=_on_xi_text_change
    )
    xi = list(st.session_state["xi_current"])
    xi_baseline = list(st.session_state["xi_baseline"])

    # ----------------------------
    # Player Comparison helpers
    # ----------------------------
    def _detect_per90_rw_cols(df: pd.DataFrame) -> list[str]:
        return sorted([c for c in df.columns if c.endswith("_per90_rw")])

    def _merge_roster_for_compare(rw_profiles: pd.DataFrame,
                                players_meta: pd.DataFrame,
                                clusters_df: pd.DataFrame) -> pd.DataFrame:
        cols_keep = ["player_id","player"] + _detect_per90_rw_cols(rw_profiles)
        left = rw_profiles[cols_keep].copy()

        meta = players_meta[["player_id","player","latest_team",
                            "primary_pos","secondary_pos","third_pos"]]
        clu  = clusters_df[["player_id","primary_grouped_pos","pos_cluster_id","pos_cluster_name"]]

        out = (left.merge(meta, on=["player_id","player"], how="left")
                .merge(clu, on="player_id", how="left"))
        # drop players without grouped primary position (insufficient apps)
        out = out.dropna(subset=["primary_grouped_pos"]).reset_index(drop=True)
        return out

    def _apply_compare_filters(pool: pd.DataFrame,
                            team: str | None,
                            pos: str | None,
                            cluster_name: str | None) -> pd.DataFrame:
        df = pool.dropna(subset=["primary_grouped_pos"]).copy()
        if team and team != "(All)":
            df = df[df["latest_team"] == team]
        if pos and pos != "(All)":
            mask = (
                (df["primary_grouped_pos"] == pos) |
                (df["primary_pos"] == pos) |
                (df["secondary_pos"] == pos) |
                (df["third_pos"] == pos)
            )
            df = df[mask]
        if cluster_name and cluster_name != "(All)":
            df = df[df["pos_cluster_name"] == cluster_name]
        return df.sort_values("player").reset_index(drop=True)


    def _percentiles_within_pos(df: pd.DataFrame, pos: str, stat_cols: list[str]) -> pd.DataFrame:
        # compute percentiles per stat within peers of the same grouped position
        peers = df[df["primary_grouped_pos"] == pos].copy()
        out = df[["player_id","player","primary_grouped_pos"]].copy()
        for c in stat_cols:
            x = peers[c].astype(float)
            # rank pct robust to ties/NaN
            ranks = x.rank(pct=True, method="average")
            # map back to full df by player_id
            map_pct = peers[["player_id"]].assign(pct=ranks.fillna(0.0))
            out = out.merge(map_pct, on="player_id", how="left", suffixes=("",""))
            out = out.rename(columns={"pct": c.replace("_per90_rw","_pct")})
        return out
    
    def hex_to_rgba(c: str, alpha: float = 0.35) -> str:
        """'#RRGGBB' -> 'rgba(r,g,b,a)'. Falls back to the input if it's already rgb/rgba."""
        if not isinstance(c, str):
            return f"rgba(31,119,180,{alpha})"
        s = c.strip().lower()
        if s.startswith("rgba(") or s.startswith("rgb(") or s in ("white", "black"):
            return s if s.startswith("rgba(") else s.replace("rgb", "rgba").replace(")", f", {alpha})")
        if s.startswith("#"):
            s = s[1:]
            if len(s) == 3:
                s = "".join(ch*2 for ch in s)  # #abc -> #aabbcc
            if len(s) >= 6:
                r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
                return f"rgba({r},{g},{b},{alpha})"
        # default fallback
        return f"rgba(31,119,180,{alpha})"


    def _radar_for_players(df: pd.DataFrame,
                        pA_id: int, pB_id: int,
                        stat_cols: list[str],
                        title: str = "Radar (position-relative percentiles)"):
        pA = df[df["player_id"] == pA_id].iloc[0]
        pB = df[df["player_id"] == pB_id].iloc[0]
        posA = str(pA.get("primary_grouped_pos", "Center Midfield")) or "Center Midfield"
        posB = str(pB.get("primary_grouped_pos", "Center Midfield")) or "Center Midfield"

        # percentiles within each player's position
        pctA = _percentiles_within_pos(df, posA, stat_cols)
        pctB = _percentiles_within_pos(df, posB, stat_cols)
        rowA = pctA[pctA["player_id"] == pA_id].iloc[0]
        rowB = pctB[pctB["player_id"] == pB_id].iloc[0]

        cats = [c.replace("_per90_rw","") for c in stat_cols]
        valsA = [100.0 * float(rowA[c.replace("_per90_rw","_pct")]) for c in stat_cols]
        valsB = [100.0 * float(rowB[c.replace("_per90_rw","_pct")]) for c in stat_cols]

        cats_loop = cats + [cats[0]]
        valsA_loop = valsA + [valsA[0]]
        valsB_loop = valsB + [valsB[0]]

        # Colors by latest team
        teamA = str(pA.get("latest_team", "")) if pd.notna(pA.get("latest_team")) else ""
        teamB = str(pB.get("latest_team", "")) if pd.notna(pB.get("latest_team")) else ""
        colA = TEAM_COLORS.get(teamA, "#1f77b4")
        colB = TEAM_COLORS.get(teamB, "#ff7f0e")

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=valsA_loop, theta=cats_loop, fill='toself',
            name=f"{pA['player']} ({posA})",
            fillcolor=hex_to_rgba(colA, 0.35),   # <-- use rgba fill
            line=dict(color=colA, width=3)
        ))
        fig.add_trace(go.Scatterpolar(
            r=valsB_loop, theta=cats_loop, fill='toself',
            name=f"{pB['player']} ({posB})",
            fillcolor=hex_to_rgba(colB, 0.35),   # <-- use rgba fill
            line=dict(color=colB, width=3)
        ))
        fig.update_layout(
            title=title,
            polar=dict(radialaxis=dict(range=[0,100], tickvals=[0,25,50,75,100])),
            showlegend=True, margin=dict(l=20,r=20,t=50,b=20), height=520,
        )
        return fig



    def _percentile_bar(df: pd.DataFrame, pid: int, stat_cols: list[str], title: str):
        import math

        row = df[df["player_id"] == pid]
        if row.empty:
            return go.Figure()  # nothing to plot

        row = row.iloc[0]
        pos = str(row.get("primary_grouped_pos", "Center Midfield")) if pd.notna(row.get("primary_grouped_pos")) else "Center Midfield"

        # Percentiles within the player's grouped position
        pct = _percentiles_within_pos(df, pos, stat_cols)
        prow = pct[pct["player_id"] == pid]
        if prow.empty:
            return go.Figure()

        prow = prow.iloc[0]
        cats = [c.replace("_per90_rw","") for c in stat_cols]

        # Build values safely (NaN -> 0)
        vals = []
        for c in stat_cols:
            v = prow.get(c.replace("_per90_rw","_pct"), np.nan)
            try:
                v = float(v)
                if not math.isfinite(v):
                    v = 0.0
            except Exception:
                v = 0.0
            vals.append(100.0 * v)

        # Color ramp: red→green; if value was NaN (now 0), draw gray
        colors = []
        for v in vals:
            # if v came from NaN originally you'd likely want neutral — detect by very small abs
            if not math.isfinite(v):
                colors.append("rgba(160,160,160,0.85)")
            else:
                # clamp
                vv = max(0.0, min(100.0, v))
                r = int(255 * (1.0 - vv/100.0))
                g = int(255 * (vv/100.0))
                colors.append(f"rgba({r},{g},0,0.85)")

        fig = go.Figure(go.Bar(
            x=cats, y=vals,
            marker=dict(color=colors),
            text=[f"{v:.0f}" for v in vals],
            textposition="outside",
            cliponaxis=False,
        ))
        fig.update_layout(
            title=title,
            yaxis=dict(range=[0, 100]),
            height=340,
            margin=dict(l=20,r=20,t=40,b=20),
            bargap=0.35,
        )
        return fig




    # =========================
    # Swap pairing & baseline deltas (for chart)
    # =========================
    def main_grouped_pos(pid: int) -> str:
        row = players_meta.loc[players_meta["player_id"] == pid, "primary_pos"]
        if not row.empty and pd.notna(row.iloc[0]): return str(row.iloc[0])
        row2 = clusters_df.loc[clusters_df["player_id"] == pid, "primary_grouped_pos"]
        if not row2.empty and pd.notna(row2.iloc[0]): return str(row2.iloc[0])
        return "Center Midfield"

    def pair_swaps_vs_baseline(baseline: list[int], current: list[int]) -> list[tuple[int,int]]:
        """Pair added players with removed players, preferring same grouped position."""
        bset, cset = set(baseline), set(current)
        removed = [p for p in baseline if p not in cset]
        added   = [p for p in current  if p not in bset]
        pairs = []
        used_out = set()
        for a in added:
            apos = main_grouped_pos(a)
            match = None
            for r in removed:
                if r in used_out: continue
                if main_grouped_pos(r) == apos:
                    match = r; break
            if match is None:
                for r in removed:
                    if r not in used_out:
                        match = r; break
            if match is not None:
                used_out.add(match)
                pairs.append((match, a))
        return pairs

    def baseline_deltas(baseline_xi: list[int], current_xi: list[int]):
        """Compute deltas (For, Against, Delta) for each (out,in) swap vs baseline."""
        pairs = pair_swaps_vs_baseline(baseline_xi, current_xi)
        rows = []
        for out_pid, in_pid in pairs:
            dF, dA, dD = delta_swap_components(
                current=baseline_xi, out_pid=out_pid, in_pid=in_pid,
                rapm_for=rapm_for, rapm_against=rapm_against,
                feature_names=feature_names, mapping=mapping, role_lookup=role_lookup,
                is_home=is_home_flag
            )
            in_row = players_meta.loc[players_meta["player_id"] == in_pid]
            in_name = in_row["player"].iloc[0] if not in_row.empty else str(in_pid)
            in_team = in_row["latest_team"].iloc[0] if not in_row.empty and pd.notna(in_row["latest_team"].iloc[0]) else ""
            rows.append({
                "out_pid": int(out_pid),
                "in_pid": int(in_pid),
                "in_name": in_name,
                "in_team": in_team,
                "delta_xg_for": float(dF),
                "delta_xg_against": float(dA),
                "delta_deltaxg": float(dD),
            })
        return pd.DataFrame(rows)


    # =========================
    # XI list + Pitch + Inline swap (with rankings)
    # =========================
    toprow = st.columns([1.0, 1.0, 2.0])
    with toprow[0]:
        if st.button("💾 Save baseline (use current XI)"):
            st.session_state["xi_baseline"] = list(st.session_state["xi_current"])
            st.session_state["_xi_text_pending"] = ",".join(map(str, st.session_state["xi_current"]))
            st.success("Baseline saved.", icon="✅")
    with toprow[1]:
        if st.button("↩️ Reset to baseline"):
            st.session_state["xi_current"] = list(st.session_state["xi_baseline"])
            st.session_state["_xi_text_pending"] = ",".join(map(str, st.session_state["xi_baseline"]))
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
        draw_pitch_with_xi(xi, base_team, players_meta, clusters_df, baseline_xi=list(st.session_state["xi_baseline"]))

    with col_list:
        # ======= BIG centered current vs baseline deltas banner =======
        baseF, baseA, baseD = score_lineup_components(
            st.session_state["xi_baseline"], rapm_for, rapm_against, feature_names, mapping, role_lookup, is_home=is_home_flag
        )
        curF, curA, curD = score_lineup_components(
            st.session_state["xi_current"], rapm_for, rapm_against, feature_names, mapping, role_lookup, is_home=is_home_flag
        )
        dF = curF - baseF
        dA = curA - baseA
        dD = curD - baseD

        def colored_delta_html(label: str, value: float, good_when: str) -> str:
            good = (value >= 0) if good_when == "higher" else (value <= 0)
            arrow = "▲" if good else "▼"
            color = "#0a7e22" if good else "#b00020"
            return f"<span style='color:{color}'>{label}: {arrow} {value:+.2f}</span>"

        banner = (
            "<div style='text-align:center; margin-bottom: 10px;'>"
            "<div style='font-size:26px; font-weight:800; margin-bottom: 6px;'>Current vs Baseline</div>"
            f"<div style='font-size:20px;'>"
            f"{colored_delta_html('Δ xG For', dF, 'higher')} &nbsp;&nbsp; "
            f"{colored_delta_html('Δ xG Against', dA, 'lower')} &nbsp;&nbsp; "
            f"{colored_delta_html('Δ ΔxG', dD, 'higher')}"
            "</div></div>"
        )
        st.markdown(banner, unsafe_allow_html=True)

        st.markdown("**Swap any player directly from this list**")

        # ====== Global candidate pool (apply the SAME filters as sidebar) ======
        base_cands = (
            players_meta[["player_id","player","latest_team","primary_pos","secondary_pos","third_pos"]]
            .merge(
                clusters_df[[
                    "player_id","primary_grouped_pos","pos_cluster_id",
                    "pos_cluster_label","pos_cluster_name"
                ]],
                on="player_id", how="left"
            )
        )

                # Add demographics and filter by Age/Country (Lineup tab only)
        base_cands = base_cands.merge(
            players_meta[["player_id","age","country","country_flag","country_label"]],
            on="player_id", how="left"
        )
        base_cands = base_cands[
            base_cands["age"].between(sel_age[0], sel_age[1], inclusive="both", na=False)
            & (base_cands["country"].isin(sel_countries))
        ]


        # Normalized grouped versions
        base_cands["primary_pos_grouped_norm"]   = base_cands["primary_pos"].map(_to_grouped)
        base_cands["secondary_pos_grouped_norm"] = base_cands["secondary_pos"].map(_to_grouped)
        base_cands["third_pos_grouped_norm"]     = base_cands["third_pos"].map(_to_grouped)

        # Team filter
        if sel_team != "(All)":
            base_cands = base_cands[base_cands["latest_team"] == sel_team]

        # Positions filter
        if sel_positions:
            if only_main_pos:
                mask_pos = base_cands["primary_grouped_pos"].isin(sel_positions)
            else:
                mask_pos = (
                    base_cands["primary_grouped_pos"].isin(sel_positions)
                    | base_cands["primary_pos_grouped_norm"].isin(sel_positions)
                    | base_cands["secondary_pos_grouped_norm"].isin(sel_positions)
                    | base_cands["third_pos_grouped_norm"].isin(sel_positions)
                )
            base_cands = base_cands[mask_pos]

        # Cluster filter (use whatever exists)
        cluster_col = "pos_cluster_name" if (
            "pos_cluster_name" in base_cands.columns and base_cands["pos_cluster_name"].notna().any()
        ) else "pos_cluster_label"
        if sel_clusters:
            base_cands[cluster_col] = base_cands[cluster_col].astype(str)
            base_cands = base_cands[base_cands[cluster_col].isin(sel_clusters)]

        # Clean display fallback
        base_cands["pos_cluster_name"] = base_cands["pos_cluster_name"].fillna("—")

        known_players = set(map(int, mapping.get("players", [])))
        base_cands = base_cands[base_cands["player_id"].isin(known_players)].copy()

        def badge(team): return TEAM_BADGE.get(team, "⚪")

        # compact two-column layout for swap controls
        cols = st.columns(2, vertical_alignment="top")
        col_idx = 0

        for _, r in xi_df.iterrows():
            with cols[col_idx]:
                pid = int(r["player_id"])
                pname = r["player"] if pd.notna(r["player"]) else str(pid)
                mpos = r["main_pos"] if pd.notna(r["main_pos"]) else "Unknown"

                st.write(f"**{pname} (id {pid})** — {mpos}")

                # Respect cross-position toggle
                if (not allow_cross_pos) and (mpos in GROUPED_POSITIONS):
                    # Restrict to the outgoing player's grouped slot
                    cand = base_cands[base_cands["primary_grouped_pos"] == mpos].copy()
                else:
                    cand = base_cands.copy()

                # remove self
                cand = cand[cand["player_id"] != pid].copy()

                if cand.empty:
                    st.caption("_No candidates match current filters._")
                else:
                    cand["label"] = cand.apply(
                        lambda x: f"{badge(x['latest_team'])}  {x['player']} (id {int(x['player_id'])})  "
                                f"[{x['primary_grouped_pos']} • {x.get('pos_cluster_name','—')}] — "
                                f"main: {x['primary_pos']} | alt: {x['secondary_pos']}/{x['third_pos']}",
                        axis=1
                    )

                    sel = st.selectbox("Replacement", options=["(choose)"] + cand["label"].tolist(),
                                    key=f"swap_sel_{pid}", label_visibility="collapsed")

                    if sel != "(choose)":
                        in_pid = int(sel.split("(id")[1].split(")")[0])

                        if st.button("Apply swap", key=f"apply_{pid}_{in_pid}"):
                            new_xi = [int(x) for x in xi if int(x) != pid] + [in_pid]
                            st.session_state["xi_current"] = new_xi
                            st.session_state["_xi_text_pending"] = ",".join(map(str, new_xi))
                            st.rerun()

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
                                results.append({
                                    "player_id": int(p),
                                    "delta_xg_for": DF,
                                    "delta_xg_against": DA,
                                    "delta_deltaxg": DD
                                })
                            res_df = pd.DataFrame(results)
                            ranked = (pool.merge(res_df, on="player_id", how="left")
                                        .merge(players_meta[["player_id","age","country_label"]], on="player_id", how="left")
                                        .sort_values(["delta_deltaxg","delta_xg_for"], ascending=[False, False])
                                        .reset_index(drop=True))
                            st.dataframe(
                                ranked[["player_id","player","latest_team","age","country_label",
                                        "primary_pos","secondary_pos","third_pos",
                                        "primary_grouped_pos","pos_cluster_label","pos_cluster_name",
                                        "delta_xg_for","delta_xg_against","delta_deltaxg"]],
                                height=320
                            )


            col_idx = 1 - col_idx  # alternate columns


    # =========================
    # Horizontal totals bars (Baseline, Current, Δ) with auto axis + table of changes
    # =========================
    def build_totals_bars(title: str, base_val: float, cur_val: float, good_when: str):
        delta = cur_val - base_val
        good = (delta >= 0) if good_when == "higher" else (delta <= 0)
        bar_delta_color = "#0a7e22" if good else "#b00020"

        xs = [base_val, cur_val, delta]
        # auto axis padding around data range
        m = max(abs(base_val), abs(cur_val), abs(delta), 0.1)
        pad = 0.15 * m + 0.05
        xmin, xmax = min(0.0, min(xs) - pad), max(0.0, max(xs) + pad)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=["Baseline"], x=[base_val], orientation="h",
            marker=dict(color="rgba(160,165,175,0.9)", line=dict(color="rgba(80,80,80,0.6)", width=1.5)),
            text=[f"{base_val:.2f}"], textposition="outside", cliponaxis=False, name="Baseline"
        ))
        fig.add_trace(go.Bar(
            y=["Current"], x=[cur_val], orientation="h",
            marker=dict(color="rgba(98,122,255,0.85)", line=dict(color="rgba(70,70,150,0.6)", width=1.5)),
            text=[f"{cur_val:.2f}"], textposition="outside", cliponaxis=False, name="Current"
        ))
        fig.add_trace(go.Bar(
            y=["Δ"], x=[delta], orientation="h",
            marker=dict(color=bar_delta_color), text=[f"{delta:+.2f}"],
            textposition="outside", cliponaxis=False, name="Δ"
        ))

        fig.update_layout(
            title=title, barmode="stack", showlegend=False,
            margin=dict(l=10, r=10, t=46, b=10), height=200,
            bargap=0.45
        )
        fig.update_xaxes(range=[xmin, xmax], zeroline=True, zerolinewidth=1, zerolinecolor="rgba(120,120,120,0.6)")
        fig.update_yaxes(showgrid=False)
        return fig

    st.divider()
    st.markdown("### Changes vs Baseline XI")

    # Changes table (per swap vs baseline)
    changes_df = baseline_deltas(list(st.session_state["xi_baseline"]), list(st.session_state["xi_current"]))

    # Totals (already computed above for banner)
    fig_tot_for = build_totals_bars("xG For (higher is better)", baseF, curF, good_when="higher")
    fig_tot_against = build_totals_bars("xG Against (lower is better)", baseA, curA, good_when="lower")
    fig_tot_dxg = build_totals_bars("ΔxG (higher is better)", baseD, curD, good_when="higher")

    cols_charts = st.columns(3)
    with cols_charts[0]:
        st.plotly_chart(fig_tot_for, config={"responsive": True, "displayModeBar": False})
    with cols_charts[1]:
        st.plotly_chart(fig_tot_against, config={"responsive": True, "displayModeBar": False})
    with cols_charts[2]:
        st.plotly_chart(fig_tot_dxg, config={"responsive": True, "displayModeBar": False})

    st.markdown("#### All changes (vs baseline)")
    if not changes_df.empty:
        changes_df = changes_df.merge(
            players_meta[["player_id","age","country_label"]], left_on="in_pid", right_on="player_id", how="left"
        )
        st.dataframe(
            changes_df.rename(columns={"in_pid":"player_id","in_name":"player","in_team":"team"})[
                ["player_id","player","team","age","country_label","delta_xg_for","delta_xg_against","delta_deltaxg"]
            ],
            height=280
        )
    else:
        st.caption("No changes yet — make a swap to see the table.")


with tab2:
    st.title("Player Comparison Tool")

    # Build roster pool once
    roster = _merge_roster_for_compare(rw_profiles, players_meta, clusters_df)
    per90_cols_rw = _detect_per90_rw_cols(roster)
    # default radar bundle (you can tailor per position later)
    default_stats = [
        "xg_per90_rw", "xA_per90_rw", "shots_per90_rw",
        "key_passes_per90_rw", "passes_cmp_per90_rw", "passes_att_per90_rw",
        "duels_won_per90_rw", "interceptions_per90_rw", "obv_total_per90_rw"
    ]
    stat_cols = [c for c in default_stats if c in per90_cols_rw]

    # Left & right player panes with independent filters
    cA, cB = st.columns(2, gap="large")

    DEFAULT_PID_A = 26400
    DEFAULT_PID_B = 26353

    def _default_index_for_pid(pool: pd.DataFrame, desired_pid: int, fallback_index: int = 0) -> int:
        if "player_id" not in pool.columns or pool.empty:
            return fallback_index
        try:
            idx = pool.index[pool["player_id"].astype(int) == int(desired_pid)]
            if len(idx) > 0:
                # convert absolute index to positional index within pool as listed in selectbox
                pos = pool.reset_index(drop=True).index[pool.reset_index(drop=True)["player_id"].astype(int) == int(desired_pid)]
                if len(pos) > 0:
                    return int(pos[0])
        except Exception:
            pass
        return min(fallback_index, len(pool) - 1) if len(pool) else 0

    # ---- Player A ----
    with cA:
        st.markdown("#### Player A")
        tA = st.selectbox("Team (A)", ["(All)"] + sorted(LIGA_MX_TEAMS), index=0, key="cmp_team_A")
        pA = st.selectbox("Position (A)", ["(All)"] + GROUPED_POSITIONS, index=0, key="cmp_pos_A")

        # cluster options depend on pos filter
        poolA_tmp = _apply_compare_filters(roster, tA, pA, None)
        clustersA = ["(All)"] + sorted([x for x in poolA_tmp["pos_cluster_name"].dropna().unique().tolist()])
        clA = st.selectbox("Cluster (A)", clustersA, index=0, key="cmp_cluster_A")

        poolA = _apply_compare_filters(roster, tA, pA, clA)
        if poolA.empty:
            st.warning("No players with these filters.")
            st.stop()

        # Build labels (unchanged)
        labelA = poolA.apply(lambda x: f"{TEAM_BADGE.get(x['latest_team'],'⚪')} {x['player']} (id {int(x['player_id'])}) — [{x['primary_grouped_pos']} • {x.get('pos_cluster_name','—')}]",
                            axis=1)

        # Use default PID if present; else fallback to 0
        idxA = _default_index_for_pid(poolA, DEFAULT_PID_A, fallback_index=0)
        selA = st.selectbox("Choose Player A", labelA.tolist(), index=idxA, key="cmp_player_A")
        pidA = int(selA.split("(id")[1].split(")")[0])


    # ---- Player B ----
    with cB:
        st.markdown("#### Player B")
        tB = st.selectbox("Team (B)", ["(All)"] + sorted(LIGA_MX_TEAMS), index=0, key="cmp_team_B")
        pB = st.selectbox("Position (B)", ["(All)"] + GROUPED_POSITIONS, index=0, key="cmp_pos_B")

        poolB_tmp = _apply_compare_filters(roster, tB, pB, None)
        clustersB = ["(All)"] + sorted([x for x in poolB_tmp["pos_cluster_name"].dropna().unique().tolist()])
        clB = st.selectbox("Cluster (B)", clustersB, index=0, key="cmp_cluster_B")

        poolB = _apply_compare_filters(roster, tB, pB, clB)
        if poolB.empty:
            st.warning("No players with these filters.")
            st.stop()

        labelB = poolB.apply(lambda x: f"{TEAM_BADGE.get(x['latest_team'],'⚪')} {x['player']} (id {int(x['player_id'])}) — [{x['primary_grouped_pos']} • {x.get('pos_cluster_name','—')}]",
                            axis=1)

        idxB = _default_index_for_pid(poolB, DEFAULT_PID_B, fallback_index=min(1, max(0, len(labelB)-1)))
        selB = st.selectbox("Choose Player B", labelB.tolist(), index=idxB, key="cmp_player_B")
        pidB = int(selB.split("(id")[1].split(")")[0])


        # Desired defaults


    # ---- Radar + percentile bars
    st.markdown("---")
    st.markdown("#### Stats Bundle")
    # let the user tweak which stats go in the radar
    stat_cols = st.multiselect("Pick stats for radar & percentile bars",
                               options=per90_cols_rw,
                               default=stat_cols,
                               key="cmp_stats_bundle")
    if len(stat_cols) < 3:
        st.info("Pick at least 3 stats for a meaningful radar.")
    else:
        # Radar (normalized 0-100 percentiles within each player's position)
        fig_radar = _radar_for_players(roster, pidA, pidB, stat_cols)
        st.plotly_chart(fig_radar, use_container_width=True, config={"responsive": True})

        # Percentile bars (0–100, red→green)
        b1, b2 = st.columns(2, gap="large")
        with b1:
            st.plotly_chart(_percentile_bar(roster, pidA, stat_cols, "Percentiles — Player A"),
                            use_container_width=True, config={"responsive": True})
        with b2:
            st.plotly_chart(_percentile_bar(roster, pidB, stat_cols, "Percentiles — Player B"),
                            use_container_width=True, config={"responsive": True})

        # Raw values table for the two players (for reference)
        cols_show = ["player_id","player","latest_team","age","country_label",
                     "primary_grouped_pos","pos_cluster_name"] + stat_cols
        tbl = (roster.merge(players_meta[["player_id","age","country_label"]], on="player_id", how="left")
                     [roster["player_id"].isin([pidA, pidB])])
        tbl = tbl[cols_show].reset_index(drop=True)
        st.dataframe(tbl, height=220)

    
with tab3:
    st.header("Network Effects — Counterfactual Transfer Impact")

    if not (NETWORK_OK and NETWORK_ARTIFACTS_OK):
        st.info(
            "To use this tab, make sure you've exported network artifacts and installed the bundle:\n"
            "• artifacts: rw_per90_team_agnostic.parquet, team_style.parquet, team_role_mix.parquet,\n"
            "  network_feature_cols.json, network_target_cols.json, network_models.pkl\n"
            "• module: modules/network_effects.py (predict_transfer_bundle)\n"
            "Then reload the app."
        )
        st.stop()

    # ===== Left controls (destination & XI)  vs  Right (choose player) =====
    cL, cR = st.columns([1.2, 1.8], gap="large")

    # ---- Destination team / season / XI ----
    with cL:
        _sorted = sorted(LIGA_MX_TEAMS)
        team_labels = [f"{TEAM_BADGE.get(t, '⚪')}  {t}" for t in _sorted]
        label_to_team = {lbl: t for lbl, t in zip(team_labels, _sorted)}

        st.subheader("Destination context")

        # Team selector (independent of Lineup tab)
        dst_lbl = st.selectbox("Target team:", team_labels, index=_sorted.index("América"))
        dst_team = label_to_team[dst_lbl]

        # Season selector (from team_style seasons)
        if "season" in team_style.columns:
            seasons_sorted = sorted(team_style["season"].dropna().unique(), key=lambda s: int(str(s).split("/")[0]))
            season_sel = st.selectbox("Season:", seasons_sorted, index=max(0, len(seasons_sorted)-1))
        else:
            season_sel = str(datetime.now().year)

        # Latest XI of the destination team
        @st.cache_data(show_spinner=False)
        def _latest_xi_for_team(events_df: pd.DataFrame, team_name: str):
            try:
                ids, mid = get_last_starting_xi_ids(events_df, team_name=team_name)
                return ids
            except Exception:
                return []

        xi_dst_default = _latest_xi_for_team(events_min, dst_team)
        if not xi_dst_default:
            st.warning(f"No XI found for {dst_team}. You can still run a counterfactual without teammate deltas.")
        # Editable target XI text (independent state from lineup tab)
        def _on_xi_dst_change():
            st.session_state["xi_dst_current"] = _parse_xi_text_to_list(st.session_state["xi_dst_text"])
        if "xi_dst_current" not in st.session_state:
            st.session_state["xi_dst_current"] = list(xi_dst_default)
        xi_dst_text_default = ",".join(map(str, st.session_state["xi_dst_current"])) if st.session_state.get("xi_dst_current") else ",".join(map(str, xi_dst_default))
        st.text_input("Target XI (comma-separated ids)", value=xi_dst_text_default, key="xi_dst_text", on_change=_on_xi_dst_change)
        xi_dst = list(st.session_state.get("xi_dst_current", xi_dst_default))

                # (inside with tab3, after xi_dst is built)
        pL, pR = st.columns([1.1, 1.0])
        with pL:
            st.markdown("#### Target XI (destination)")
            if xi_dst:
                draw_pitch_with_xi(xi_dst, dst_team, players_meta, clusters_df, baseline_xi=xi_dst)
        with pR:
            st.empty()  # reserved for future mini charts


        # Optional: who leaves (match-style swap; may be None)
        out_pid_opt = None
        if xi_dst:
            # prettify
            _names = players_meta.set_index("player") if "player" in players_meta.columns else None
            pool_out = (pd.DataFrame({"player_id": xi_dst})
                        .merge(players_meta[["player_id","player"]], on="player_id", how="left"))
            out_opts = ["(None)"] + [f"{row['player']} (id {int(row['player_id'])})" for _, row in pool_out.iterrows()]
            out_sel = st.selectbox("Outgoing player (optional):", out_opts, index=0)
            if out_sel != "(None)":
                out_pid_opt = int(out_sel.split("(id")[1].split(")")[0])

        st.caption("Tip: Outgoing player helps adjust the team's role-mix more realistically.")

    # ---- Pick incoming player & filters (same global style as Lineup) ----
    with cR:
        st.subheader("Incoming player & filters")

        # Global filters (reuse your sidebar semantics here but scoped to this tab)
        g1, g2 = st.columns(2)
        with g1:
            latest_team_filter = ["(All)"] + _sorted
            sel_team_net = st.selectbox("Filter by latest team:", latest_team_filter, index=0, key="net_sel_team")
        with g2:
            pos_filter_opts = ["(All)"] + GROUPED_POSITIONS
            sel_pos_net = st.selectbox("Filter by grouped position:", pos_filter_opts, index=0, key="net_sel_pos")

        # Cluster filter (names)
        cluster_names_all = sorted(
            clusters_df["pos_cluster_name"].dropna().unique().tolist()
            if "pos_cluster_name" in clusters_df.columns else []
        )
        sel_cluster_net = st.selectbox("Filter by cluster (optional):", ["(All)"] + cluster_names_all, index=0, key="net_sel_cluster")

        # Build candidate pool (same schema as Lineup)
        base_cands_net = (players_meta[["player_id","player","latest_team","primary_pos","secondary_pos","third_pos"]]
            .merge(clusters_df[["player_id","primary_grouped_pos","pos_cluster_id","pos_cluster_label","pos_cluster_name"]],
                   on="player_id", how="left"))
        known_players = set(map(int, mapping.get("players", [])))
        base_cands_net = base_cands_net[base_cands_net["player_id"].isin(known_players)].copy()

        if sel_team_net != "(All)":
            base_cands_net = base_cands_net[base_cands_net["latest_team"] == sel_team_net]
        if sel_pos_net != "(All)":
            # include if primary/secondary/third matches OR its grouped primary matches
            mask = (
                (base_cands_net["primary_grouped_pos"] == sel_pos_net) |
                (base_cands_net["primary_pos"] == sel_pos_net) |
                (base_cands_net["secondary_pos"] == sel_pos_net) |
                (base_cands_net["third_pos"] == sel_pos_net)
            )
            base_cands_net = base_cands_net[mask]
        if sel_cluster_net != "(All)" and "pos_cluster_name" in base_cands_net.columns:
            base_cands_net = base_cands_net[base_cands_net["pos_cluster_name"] == sel_cluster_net]

        def badge(team): return TEAM_BADGE.get(team, "⚪")
        if base_cands_net.empty:
            st.warning("No candidates match current filters.")
            in_pid = None
        else:
            base_cands_net["label"] = base_cands_net.apply(
                lambda x: f"{badge(x['latest_team'])}  {x['player']} (id {int(x['player_id'])})  "
                          f"[{x.get('primary_grouped_pos','?')} • {x.get('pos_cluster_name','—')}] — "
                          f"main: {x['primary_pos']} | alt: {x['secondary_pos']}/{x['third_pos']}", axis=1
            )
            sel_in = st.selectbox("Incoming player:", ["(choose)"] + base_cands_net["label"].tolist(), index=0, key="net_in_player")
            in_pid = None if sel_in == "(choose)" else int(sel_in.split("(id")[1].split(")")[0])

        # Choose stats to display
        default_stats_show = [c for c in network_target_cols if c in (
            "xg_per90", "xA_per90", "key_passes_per90", "passes_att_per90", "passes_cmp_per90", "obv_total_per90"
        )]
        stats_show = st.multiselect("Stats to display:", network_target_cols, default=default_stats_show, key="net_stats_show")

        run_btn = st.button("Run counterfactual", type="primary", disabled=(in_pid is None))

    st.markdown("---")

    if run_btn and (in_pid is not None):
        # Run bundle (uses your new network_effects.py)
        bundle = predict_transfer_bundle(
            player_id=in_pid,
            to_team=dst_team,
            season=season_sel,
            rw_profiles=rw_profiles,
            clusters=clusters_df,
            team_style=team_style,
            team_role_mix=team_role_mix,
            network_models=network_models,
            feature_cols=network_feature_cols,
            target_cols=network_target_cols,
            players_meta=players_meta,
            xi_target=xi_dst,
            out_pid=out_pid_opt
        )

        # -------------------------
        # Helpers aligned to module
        # -------------------------
        def _style_delta(v: float) -> str:
            if pd.isna(v): return ""
            if v > 0: return "color: green; font-weight: 600;"
            if v < 0: return "color: red; font-weight: 600;"
            return "color: gray;"

        def _ref_col_for(stat: str) -> str:
            """Map target per90 -> *_per90_rw column."""
            return stat.replace("_per90", "_per90_rw")

        def _pos_for_bundle_ctx() -> str:
            # network_effects.predict_transfer_bundle sets context['position'] to grouped pos
            return bundle["context"]["position"]

        def _percentile_from_rw_dist(value: float, grouped_pos: str, stat: str) -> float:
            """
            Match module's _percentiles_vs_position:
            - filter players by same primary_grouped_pos
            - use *_per90_rw reference column
            - percentile = 100 * mean(vals <= value)
            """
            try:
                ref_col = _ref_col_for(stat)
                pos_players = clusters_df.loc[
                    clusters_df["primary_grouped_pos"] == grouped_pos, "player_id"
                ].unique().tolist()
                ref = rw_profiles[rw_profiles["player_id"].isin(pos_players)]
                vals = pd.to_numeric(ref[ref_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if vals.empty or pd.isna(value):
                    return np.nan
                return float((vals <= float(value)).mean() * 100.0)
            except Exception:
                return np.nan

        # -------------------------
        # Header & new cluster
        # -------------------------
        ctx = bundle["context"]
        newc = bundle["new_cluster"]
        cxa, cxb = st.columns([1.2, 1.8])
        with cxa:
            st.subheader(f"{ctx['player']} → {ctx['to_team']} ({ctx['season']})")
            st.caption(f"From: {ctx.get('from_team','?')} | Position: {ctx['position']} | Current cluster: {ctx.get('current_cluster','—')}")
        with cxb:
            if newc["id"] is not None:
                st.success(f"Counterfactual cluster: {newc['label']} (id {newc['id']})")
            else:
                st.info("Counterfactual cluster not available for this position.")

        # -------------------------
        # KPI chips (pred + % change)
        # -------------------------
        pred = bundle.get("pred_per90", {}) or {}
        pct  = bundle.get("pct_change", {}) or {}
        kcols = st.columns(len(stats_show)) if stats_show else st.columns(1)
        for i, stat in enumerate(stats_show if stats_show else ["xg_per90"]):
            with kcols[i if stats_show else 0]:
                v = pred.get(stat, np.nan)
                d = pct.get(stat, np.nan)
                if "against" in stat.lower():
                    st.metric(stat, f"{v:.3f}" if pd.notna(v) else "—",
                              delta=f"{d:+.1f}%" if pd.notna(d) else "—",
                              delta_color="inverse")
                else:
                    st.metric(stat, f"{v:.3f}" if pd.notna(v) else "—",
                              delta=f"{d:+.1f}%" if pd.notna(d) else "—",
                              delta_color="normal")

        # -------------------------
        # Current vs Predicted table
        # (per-90 + percentiles computed like module)
        # -------------------------
        st.markdown("#### Current vs Predicted — per-90 & percentiles (0–100)")

        curr_per90 = bundle.get("current_per90", {}) or {}
        pred_per90 = bundle.get("pred_per90", {}) or {}
        pred_pct   = bundle.get("percentiles", {}) or {}  # module already computed (pred only)

        grouped_pos = _pos_for_bundle_ctx()

        # Compute CURRENT percentiles the same way the module does for predicted
        curr_pct = {}
        for s in stats_show:
            curr_pct[s] = _percentile_from_rw_dist(curr_per90.get(s, np.nan), grouped_pos, s)

        # Build table
        rows = []
        for s in stats_show:
            c_val = curr_per90.get(s, np.nan)
            p_val = pred_per90.get(s, np.nan)
            rows.append({
                "stat": s,
                "current_per90": c_val,
                "current_pct": curr_pct.get(s, np.nan),
                "pred_per90": p_val,
                "pred_pct": pred_pct.get(s, np.nan),
                "Δ per90": (p_val - c_val) if (pd.notna(p_val) and pd.notna(c_val)) else np.nan,
                "Δ %": pct.get(s, np.nan)
            })
        tbl = pd.DataFrame(rows)
        styler = (tbl.style
                  .format({
                      "current_per90": "{:.3f}",
                      "pred_per90": "{:.3f}",
                      "Δ per90": "{:+.3f}",
                      "current_pct": "{:.1f}",
                      "pred_pct": "{:.1f}",
                      "Δ %": "{:+.1f}"
                  })
                  .applymap(_style_delta, subset=["Δ per90", "Δ %"]))
        st.dataframe(styler, use_container_width=True, height=280)

        # -------------------------
        # Teammates impact — one-row summary (green/red)
        # -------------------------
        st.markdown("### Impact on Teammates (target XI)")
        team_imp = bundle.get("teammates_impact", pd.DataFrame())
        if team_imp is None or team_imp.empty:
            st.info("No teammate deltas (empty XI or artifacts). Provide a target XI and try again.")
        else:
            # Per the module, deltas come from role-mix change applied to each teammate.
            # Detect uniform delta per stat. If uniform, show that number; else show the mean.
            summary = []
            for s in stats_show:
                dcol = f"delta_{s}"
                if dcol not in team_imp.columns:
                    summary.append({"stat": s, "Δ per90 (all teammates)": np.nan})
                    continue
                vec = pd.to_numeric(team_imp[dcol], errors="coerce").dropna()
                if vec.empty:
                    summary.append({"stat": s, "Δ per90 (all teammates)": np.nan})
                    continue
                if vec.nunique(dropna=True) == 1 or (vec.max() - vec.min()) < 1e-9:
                    # truly uniform
                    v = float(vec.iloc[0])
                else:
                    # not exactly uniform: use mean so you still get a single-row view
                    v = float(vec.mean())
                summary.append({"stat": s, "Δ per90 (all teammates)": v})

            df_sum = pd.DataFrame(summary)
            sty = (df_sum.style
                   .format({"Δ per90 (all teammates)": "{:+.3f}"})
                   .applymap(_style_delta, subset=["Δ per90 (all teammates)"]))
            st.dataframe(sty, use_container_width=True, height=120)

            with st.expander("Show full per-teammate before/after/delta table"):
                keep_cols = ["player_id"]
                for s in stats_show:
                    keep_cols += [f"before_{s}", f"after_{s}", f"delta_{s}"]
                show = (team_imp[keep_cols]
                        .merge(players_meta[["player_id","player","latest_team","age","country_label"]],
                               on="player_id", how="left"))

                # color only the delta_* columns
                delta_cols = [c for c in show.columns if c.startswith("delta_")]
                def _apply_rowstyle(df):
                    styles = pd.DataFrame("", index=df.index, columns=df.columns)
                    for c in delta_cols:
                        styles[c] = df[c].apply(_style_delta)
                    return styles
                st.dataframe(show.style.apply(_apply_rowstyle, axis=None),
                             use_container_width=True, height=420)

        st.caption(f"Impact change is not supported for players within the same cluster.")
