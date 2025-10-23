# ================================================================
# Network Effects — Counterfactual bundle for transfers / swaps
# ================================================================
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler

# ---- tuning knob for teammate impacts (1.0 = no boost) ----
IMPACT_GAIN_DEFAULT: float = 1.0

# ---------- If these helpers already exist in your codebase, reuse them ----------
def _detect_per90_cols(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.endswith("_per90")])

def _detect_per90_rw_cols(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.endswith("_per90_rw")])

def _player_rw_vector(rw_profiles: pd.DataFrame, player_id: int) -> pd.Series:
    row = rw_profiles[rw_profiles["player_id"] == player_id]
    if row.empty:
        raise ValueError(f"RW profile not found for player_id={player_id}")
    return row.iloc[0]

def _player_position(clusters: pd.DataFrame, player_id: int) -> str:
    pos = clusters.loc[clusters["player_id"] == player_id, "primary_grouped_pos"]
    return (pos.iloc[0] if not pos.empty else "Center Midfield")

def _player_role_id(clusters: pd.DataFrame, player_id: int) -> Optional[int]:
    rr = clusters.loc[clusters["player_id"] == player_id, "pos_cluster_id"]
    return int(rr.iloc[0]) if not rr.empty and pd.notna(rr.iloc[0]) else None

def _player_team(players_meta: pd.DataFrame, player_id: int) -> str:
    r = players_meta.loc[players_meta["player_id"]==player_id, "latest_team"]
    return r.iloc[0] if not r.empty else ""

def _player_name(players_meta: pd.DataFrame, player_id: int) -> str:
    r = players_meta.loc[players_meta["player_id"]==player_id, "player"]
    return r.iloc[0] if not r.empty else str(player_id)

def _player_pos_label(clusters: pd.DataFrame, player_id: int) -> str:
    r = clusters.loc[clusters["player_id"]==player_id, "pos_cluster_label"]
    return r.iloc[0] if not r.empty else ""

def _pos_onehot_columns(feature_cols: List[str]) -> List[str]:
    return [c for c in feature_cols if c.startswith("pos_")]

def _player_position_onehot(pos: str, pos_onehot_cols: List[str]) -> pd.Series:
    o = {c: 0.0 for c in pos_onehot_cols}
    key = f"pos_{pos}"
    if key in o:
        o[key] = 1.0
    return pd.Series(o, dtype=float)

def build_feature_row_for_context(
    player_id: int,
    rw_profiles: pd.DataFrame,
    clusters: pd.DataFrame,
    team_style: pd.DataFrame,
    team_role_mix: pd.DataFrame,
    season: str,
    team: str,
    feature_cols: List[str]
) -> pd.DataFrame:
    rw_row = _player_rw_vector(rw_profiles, player_id)
    pos = _player_position(clusters, player_id)
    sty = team_style[(team_style["season"]==season)&(team_style["team"]==team)]
    if sty.empty:
        raise ValueError(f"No team_style for team={team} season={season}")
    mix = team_role_mix[(team_role_mix["season"]==season)&(team_role_mix["team"]==team)]
    if mix.empty:
        mix = pd.DataFrame([{"season":season,"team":team}])

    pos_oh_cols = _pos_onehot_columns(feature_cols)
    pos_oh = _player_position_onehot(pos, pos_oh_cols)

    feat = pd.Series(dtype=float)
    for c in feature_cols:
        if c in rw_row.index:
            feat[c] = float(rw_row[c])
        elif c in sty.columns:
            feat[c] = float(sty.iloc[0][c])
        elif c in mix.columns:
            feat[c] = float(mix.iloc[0][c])
        elif c in pos_oh.index:
            feat[c] = float(pos_oh[c])
        else:
            feat[c] = 0.0
    return pd.DataFrame([feat.values], columns=feature_cols)

def predict_player_in_context(models: Dict[str, any], feature_row: pd.DataFrame) -> Dict[str, float]:
    out = {}
    for stat, mdl in models.items():
        out[stat] = float(mdl.predict(feature_row.values)[0])
    return out

def adjust_role_mix_for_swap(team_mix_row: pd.Series,
                             in_role_id: Optional[int],
                             out_role_id: Optional[int] = None) -> pd.Series:
    ser = team_mix_row.copy()
    cols = [c for c in ser.index if c.startswith("role_mix_")]
    if not cols:
        return ser

    def bump(role_id, delta):
        if role_id is None:
            return
        col = f"role_mix_{int(role_id)}"
        if col in ser.index:
            ser[col] = max(0.0, float(ser[col]) + delta)

    bump(in_role_id, +1.0)
    bump(out_role_id, -1.0)

    tot = float(ser[cols].sum())
    if tot > 0:
        ser[cols] = ser[cols] / tot
    return ser

def _apply_role_mix_gain(base_mix: pd.Series,
                         new_mix: pd.Series,
                         impact_gain: float) -> pd.Series:
    """
    Interpolates from base -> new by an amplification factor, then renormalizes.
    """
    cols = [c for c in base_mix.index if c.startswith("role_mix_")]
    out = base_mix.copy()
    for c in cols:
        dv = float(new_mix.get(c, 0.0) - base_mix.get(c, 0.0))
        out[c] = max(0.0, float(base_mix.get(c, 0.0)) + impact_gain * dv)
    tot = float(out[cols].sum())
    if tot > 0:
        out[cols] = out[cols] / tot
    return out

# ---------- Percentiles vs players in SAME grouped position ----------
def _percentiles_vs_position(
    rw_profiles: pd.DataFrame,
    clusters: pd.DataFrame,
    grouped_pos: str,
    preds_per90: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute percentiles (0-100) for the predicted per90 values vs distribution
    of players in the same grouped position using *_per90_rw columns.
    """
    per90_rw_cols = [c for c in preds_per90.keys()]  # already in *_per90 names
    # map targets (per90) -> reference columns (_per90_rw)
    ref_cols = []
    for c in per90_rw_cols:
        rwc = c.replace("_per90", "_per90_rw")
        if rwc in rw_profiles.columns:
            ref_cols.append((c, rwc))

    pos_players = clusters[clusters["primary_grouped_pos"] == grouped_pos]["player_id"].unique().tolist()
    ref = rw_profiles[rw_profiles["player_id"].isin(pos_players)].copy()
    out = {}
    for tgt_col, ref_col in ref_cols:
        vals = ref[ref_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            out[tgt_col] = np.nan
        else:
            v = preds_per90[tgt_col]
            pct = 100.0 * (vals <= v).mean()
            out[tgt_col] = float(pct)
    return out

# ---------- Nearest-centroid cluster assignment for counterfactual ----------
def _fit_position_space_and_centroids(
    rw_profiles: pd.DataFrame,
    clusters: pd.DataFrame,
    grouped_pos: str
):
    """
    Build a StandardScaler on *_per90_rw for players of this grouped_pos,
    compute centroids per (pos_cluster_id) in standardized space.
    """
    pos_players = clusters[clusters["primary_grouped_pos"] == grouped_pos][["player_id","pos_cluster_id"]].dropna()
    if pos_players.empty:
        return None, None, None
    per_cols = _detect_per90_rw_cols(rw_profiles)
    M = (rw_profiles.merge(pos_players, on="player_id", how="inner"))[per_cols + ["pos_cluster_id"]].dropna()
    if M.empty:
        return None, None, None

    X = M[per_cols].astype(float).values
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    lab = M["pos_cluster_id"].astype(int).values

    centroids = {}
    for k in np.unique(lab):
        centroids[int(k)] = Z[lab==k].mean(axis=0)
    return scaler, per_cols, centroids

def _assign_counterfactual_cluster(
    preds_per90: Dict[str, float],                 # per90 dict
    grouped_pos: str,
    rw_profiles: pd.DataFrame,
    clusters: pd.DataFrame
) -> Tuple[Optional[int], Optional[str]]:
    """
    Map predicted per90 -> synthetic *_per90_rw vector, standardize, assign to nearest centroid.
    Returns (cluster_id, cluster_label) if possible.
    """
    scaler, per_cols, cents = _fit_position_space_and_centroids(rw_profiles, clusters, grouped_pos)
    if scaler is None:
        return None, None
    # build vector in same order per_cols (which are *_per90_rw)
    x = []
    for c in per_cols:
        base_name = c.replace("_per90_rw", "_per90")
        x.append(preds_per90.get(base_name, 0.0))
    x = np.array([x], dtype=float)
    z = scaler.transform(x)[0]
    # nearest centroid (euclidean)
    best_k, best_d = None, 1e18
    for k, mu in cents.items():
        d = float(np.linalg.norm(z - mu))
        if d < best_d:
            best_d, best_k = d, k
    # label (if available)
    lab_df = clusters[(clusters["primary_grouped_pos"]==grouped_pos) & (clusters["pos_cluster_id"]==best_k)]
    lab = lab_df["pos_cluster_name"].dropna().unique()
    best_label = lab[0] if len(lab) else None
    return best_k, best_label

# ---------- Teammates' impact given a XI swap ----------
def _get_team_mix_row(team_role_mix: pd.DataFrame, season: str, team: str) -> pd.Series:
    mr = team_role_mix[(team_role_mix["season"]==season)&(team_role_mix["team"]==team)]
    if mr.empty:
        # create zero vector with correct columns
        cols = [c for c in team_role_mix.columns if c.startswith("role_mix_")]
        return pd.Series({c: 0.0 for c in cols})
    return mr.iloc[0]

def _feature_row_with_custom_mix(
    player_id: int,
    rw_profiles: pd.DataFrame,
    clusters: pd.DataFrame,
    season: str,
    team: str,
    team_style: pd.DataFrame,
    mix_series: pd.Series,
    feature_cols: List[str]
) -> pd.DataFrame:
    # reuse style
    sty = team_style[(team_style["season"]==season)&(team_style["team"]==team)]
    if sty.empty:
        raise ValueError(f"No team_style for team={team} season={season}")
    pos = _player_position(clusters, player_id)
    rw_row = _player_rw_vector(rw_profiles, player_id)

    pos_oh_cols = _pos_onehot_columns(feature_cols)
    pos_oh = _player_position_onehot(pos, pos_oh_cols)

    feat = pd.Series(dtype=float)
    for c in feature_cols:
        if c in rw_row.index:
            feat[c] = float(rw_row[c])
        elif c in sty.columns:
            feat[c] = float(sty.iloc[0][c])
        elif c in mix_series.index:
            feat[c] = float(mix_series[c])
        elif c in pos_oh.index:
            feat[c] = float(pos_oh[c])
        else:
            feat[c] = 0.0
    return pd.DataFrame([feat.values], columns=feature_cols)

def teammates_impact_for_swap(
    xi_target: List[int],                # target XI (before adding the player)
    in_pid: int,
    out_pid: Optional[int],
    season: str,
    team: str,
    rw_profiles: pd.DataFrame,
    clusters: pd.DataFrame,
    team_style: pd.DataFrame,
    team_role_mix: pd.DataFrame,
    network_models: Dict[str, any],
    feature_cols: List[str],
    target_cols: List[str],
    impact_gain: float = IMPACT_GAIN_DEFAULT
) -> pd.DataFrame:
    """
    For each teammate in xi_target (excluding out_pid), compute predicted per90 BEFORE/AFTER
    adjusting the team role mix for (in_role, out_role). Returns per-player deltas.
    The AFTER mix is amplified by `impact_gain` to reflect systemic adjustments.
    """
    # current mix row
    base_mix = _get_team_mix_row(team_role_mix, season, team)
    in_role = _player_role_id(clusters, in_pid)
    out_role = _player_role_id(clusters, out_pid) if out_pid is not None else None
    new_mix = adjust_role_mix_for_swap(base_mix, in_role_id=in_role, out_role_id=out_role)
    new_mix_amp = _apply_role_mix_gain(base_mix, new_mix, impact_gain)

    rows = []
    for mate in xi_target:
        if out_pid is not None and mate == out_pid:
            continue  # leaving
        # BEFORE
        Xb = _feature_row_with_custom_mix(mate, rw_profiles, clusters, season, team,
                                          team_style, base_mix, feature_cols)
        pb = predict_player_in_context(network_models, Xb)
        # AFTER (amplified)
        Xa = _feature_row_with_custom_mix(mate, rw_profiles, clusters, season, team,
                                          team_style, new_mix_amp, feature_cols)
        pa = predict_player_in_context(network_models, Xa)

        delt = {f"delta_{k}": pa[k]-pb[k] for k in target_cols}
        rows.append({
            "player_id": int(mate),
            **{f"before_{k}": pb[k] for k in target_cols},
            **{f"after_{k}":  pa[k] for k in target_cols},
            **delt
        })
    return pd.DataFrame(rows)

# ---------- MASTER: predict_transfer_bundle ----------
def predict_transfer_bundle(
    player_id: int,
    to_team: str,
    season: str,
    rw_profiles: pd.DataFrame,
    clusters: pd.DataFrame,
    team_style: pd.DataFrame,
    team_role_mix: pd.DataFrame,
    network_models: Dict[str, any],
    feature_cols: List[str],
    target_cols: List[str],
    players_meta: Optional[pd.DataFrame] = None,
    xi_target: Optional[List[int]] = None,     # target XI (current)
    out_pid: Optional[int] = None,             # who leaves (optional)
    impact_gain: float = IMPACT_GAIN_DEFAULT   # amplification for teammate impacts
) -> Dict[str, any]:
    """
    Returns a dict with:
      - context: player_name, from_team, to_team, season
      - pred_per90: dict stat->value
      - pct_change: dict stat->% delta vs player RW baseline
      - percentiles: dict stat->percentile (0..100) vs SAME grouped position
      - new_cluster: {"id": int|None, "label": str|None}
      - current_per90: dict stat->current RW per90 (baseline)
      - current_cluster_name: str
      - teammates_impact: DataFrame (before/after/delta per stat) for teammates in xi_target
      - impact_gain: float (echo)
    """
    # Build features for player at target context
    Xrow = build_feature_row_for_context(
        player_id=player_id,
        rw_profiles=rw_profiles,
        clusters=clusters,
        team_style=team_style,
        team_role_mix=team_role_mix,
        season=season,
        team=to_team,
        feature_cols=feature_cols
    )
    preds = predict_player_in_context(network_models, Xrow)

    # Baseline RW per90 (current) & % change vs baseline
    rw_row = _player_rw_vector(rw_profiles, player_id)
    current_per90: Dict[str, float] = {}
    pct: Dict[str, float] = {}
    for s in target_cols:
        rw_name = s.replace("_per90", "_per90_rw")
        base = np.nan
        if rw_name in rw_row.index and pd.notna(rw_row[rw_name]):
            base = float(rw_row[rw_name])
        elif s in rw_row.index and pd.notna(rw_row[s]):
            base = float(rw_row[s])
        current_per90[s] = base
        if np.isnan(base) or abs(base) < 1e-9:
            pct[s] = np.nan
        else:
            pct[s] = float((preds[s] / base - 1.0) * 100.0)

    # percentiles vs position
    gpos = _player_position(clusters, player_id)
    perc = _percentiles_vs_position(rw_profiles, clusters, gpos, preds)

    # counterfactual cluster assignment
    new_cid, new_clabel = _assign_counterfactual_cluster(preds, gpos, rw_profiles, clusters)

    # teammates' impact (optional)
    teammates_df = pd.DataFrame()
    if xi_target is not None and len(xi_target):
        teammates_df = teammates_impact_for_swap(
            xi_target=xi_target,
            in_pid=player_id,
            out_pid=out_pid,
            season=season,
            team=to_team,
            rw_profiles=rw_profiles,
            clusters=clusters,
            team_style=team_style,
            team_role_mix=team_role_mix,
            network_models=network_models,
            feature_cols=feature_cols,
            target_cols=target_cols,
            impact_gain=impact_gain
        )

    # Context
    from_team = _player_team(players_meta, player_id) if players_meta is not None else None
    player_name = _player_name(players_meta, player_id) if players_meta is not None else str(player_id)
    pos_label = _player_pos_label(clusters, player_id)

    return {
        "context": {
            "player_id": int(player_id),
            "player": player_name,
            "position": gpos,
            "current_cluster": pos_label,
            "from_team": from_team,
            "to_team": to_team,
            "season": season
        },
        "pred_per90": preds,
        "pct_change": pct,
        "percentiles": perc,
        "new_cluster": {"id": new_cid, "label": new_clabel},
        "current_per90": current_per90,
        "current_cluster_name": pos_label,
        "teammates_impact": teammates_df,
        "impact_gain": float(impact_gain)
    }
