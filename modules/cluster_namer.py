# modules/cluster_namer.py
from __future__ import annotations
import re
from typing import Tuple, List, FrozenSet

# -------------------------
# Helpers de normalización
# -------------------------
_WS_RE = re.compile(r"\s+")
def _norm(s: str) -> str:
    # quita NBSP y colapsa espacios
    s = s.replace("\u00A0", " ").replace("\u2009", " ")
    s = _WS_RE.sub(" ", s.strip())
    return s

def _parse_label(label: str) -> Tuple[str, List[str]]:
    """
    'Center Forward | +xg_per90_rw, -xA_per90_rw, +passes_att_per90_rw'
      -> ('Center Forward', ['+xg_per90_rw','-xA_per90_rw','+passes_att_per90_rw'])
    """
    if not isinstance(label, str):
        return ("", [])
    lab = _norm(label)
    if "|" not in lab:
        return (lab, [])
    pos, feats = lab.split("|", 1)
    pos = _norm(pos)
    feats = [ _norm(f) for f in feats.split(",") if _norm(f) ]
    # valida que cada feat empiece con + o -
    clean = []
    for f in feats:
        if f and f[0] in {"+","-"}:
            clean.append(f)
        else:
            # si viene sin signo, asume '+'
            clean.append("+" + f)
    return (pos, clean)

def _canonical_key(label: str) -> Tuple[str, FrozenSet[str]]:
    pos, feats = _parse_label(label)
    # ordena feats alfabéticamente con su signo
    feats_sorted = tuple(sorted(feats))
    return (pos, frozenset(feats_sorted))

# -------------------------
# Mapa canónico (pos, frozenset(features firmadas)) -> nombre
# Usa las etiquetas que nos pasaste
# -------------------------
def _fs(*feats: str) -> FrozenSet[str]:
    return frozenset(sorted(_norm(f) for f in feats))

CANON_MAP = {
    # === Attacking Midfield ===
    ("Attacking Midfield", _fs("+passes_att_per90_rw", "+xA_per90_rw", "+passes_cmp_per90_rw")): "Creator 10",
    ("Attacking Midfield", _fs("-passes_att_per90_rw", "-xA_per90_rw", "-passes_cmp_per90_rw")): "Shooter AM",

    # === Center Back ===
    ("Center Back", _fs("+xg_per90_rw", "+shots_per90_rw", "+goals_per90_rw")): "Set-piece Threat CB",
    ("Center Back", _fs("-xg_per90_rw", "-shots_per90_rw", "-goals_per90_rw")): "Stay back CB",
    ("Center Back", _fs("+key_passes_per90_rw", "+xA_per90_rw", "+passes_att_per90_rw")): "Libero CB",
    ("Center Back", _fs("-obv_total_per90_rw", "+obv_against_per90_rw", "-passes_att_per90_rw")): "Stopper CB",

    # === Center Forward ===
    ("Center Forward", _fs("-goals_per90_rw", "-obv_total_per90_rw", "-obv_for_per90_rw")): "Deep Lying Forward",
    ("Center Forward", _fs("+goals_per90_rw", "+xg_per90_rw", "+shots_per90_rw")): "Finisher 9",
    ("Center Forward", _fs("+passes_cmp_per90_rw", "+passes_att_per90_rw", "+key_passes_per90_rw")): "Link-up 9",

    # === Center Midfield ===
    ("Center Midfield", _fs("-obv_total_per90_rw", "-obv_for_per90_rw", "-passes_att_per90_rw")): "Holding CM",
    ("Center Midfield", _fs("+obv_total_per90_rw", "+obv_for_per90_rw", "+passes_att_per90_rw")): "Progressor CM",

    # === Defensive Midfield ===
    ("Defensive Midfield", _fs("+xA_per90_rw", "+key_passes_per90_rw", "+assists_per90_rw")): "Regista",
    ("Defensive Midfield", _fs("+passes_cmp_per90_rw", "+passes_att_per90_rw", "+duels_won_per90_rw")): "Balanced DM",
    ("Defensive Midfield", _fs("-obv_total_per90_rw", "-passes_cmp_per90_rw", "-passes_att_per90_rw")): "Destroyer DM",
    ("Defensive Midfield", _fs("-shots_per90_rw", "+clearances_per90_rw", "-xg_per90_rw")): "Sweeper DM",

    # === Goalkeeper ===
    ("Goalkeeper", _fs("-passes_att_per90_rw", "-passes_cmp_per90_rw", "-xg_per90_rw")): "Traditional GK",
    ("Goalkeeper", _fs("+passes_att_per90_rw", "+passes_cmp_per90_rw", "+xg_per90_rw")): "Sweeper-keeper",

    # === Left Back ===
    ("Left Back", _fs("+obv_for_per90_rw", "+xA_per90_rw", "+assists_per90_rw")): "Attacking LB",
    ("Left Back", _fs("-obv_for_per90_rw", "-xA_per90_rw", "-assists_per90_rw")): "Defensive LB",

    # === Left Wing ===
    ("Left Wing", _fs("-passes_att_per90_rw", "-passes_cmp_per90_rw", "-key_passes_per90_rw")): "Off-ball LW",
    ("Left Wing", _fs("+duels_won_per90_rw", "+passes_cmp_per90_rw", "+passes_att_per90_rw")): "Ball-carrying LW",
    ("Left Wing", _fs("+xA_per90_rw", "+obv_for_per90_rw", "+obv_total_per90_rw")): "Creator LW",

    # === Right Back ===
    ("Right Back", _fs("+xA_per90_rw", "+key_passes_per90_rw", "+assists_per90_rw")): "Attacking RB",
    ("Right Back", _fs("-xA_per90_rw", "-key_passes_per90_rw", "-assists_per90_rw")): "Defensive RB",

    # === Right Wing ===
    ("Right Wing", _fs("+xg_per90_rw", "+shots_per90_rw", "+goals_per90_rw")): "Finisher RW (Inverted)",
    ("Right Wing", _fs("-xg_per90_rw", "+duels_won_per90_rw", "-shots_per90_rw")): "Ball-winning RW",
    ("Right Wing", _fs("-xA_per90_rw", "-key_passes_per90_rw", "-passes_att_per90_rw")): "Off-ball RW",
}

# -------------------------
# Fallback suave (por si aparece algo nuevo)
# -------------------------
def _fallback_name(pos: str, feats_signed: List[str]) -> str:
    lab = " ".join(feats_signed).lower()
    def has(*terms: str) -> bool: return all(t in lab for t in terms)

    if pos in ("Center Forward", "Right Wing", "Left Wing"):
        if has("+xg_per90_rw") or has("+goals_per90_rw"):
            return "Finisher"
        if has("+xA_per90_rw") or "key_passes_per90_rw" in lab:
            return "Creator Forward"
        if has("+passes_att_per90_rw", "+passes_cmp_per90_rw"):
            return "Link-up Forward"
        return "Utility Forward"

    if pos == "Attacking Midfield":
        if has("+xA_per90_rw") or "key_passes_per90_rw" in lab:
            return "Creator 10"
        return "Attacking Midfielder"

    if pos == "Center Midfield":
        if has("+obv_for_per90_rw") or has("+passes_att_per90_rw", "+passes_cmp_per90_rw"):
            return "Progressor CM"
        return "Holding CM"

    if pos == "Defensive Midfield":
        if has("+passes_att_per90_rw", "+passes_cmp_per90_rw") and "duels_won_per90_rw" in lab:
            return "Balanced DM"
        if "clearances_per90_rw" in lab:
            return "Screening DM"
        return "Destroyer DM"

    if pos in ("Right Back", "Left Back"):
        if has("+xA_per90_rw") or "assists_per90_rw" in lab:
            return "Attacking FB"
        return "Defensive FB"

    if pos == "Center Back":
        if has("+key_passes_per90_rw") or has("+passes_att_per90_rw"):
            return "Ball-playing CB"
        if has("+xg_per90_rw") or "shots_per90_rw" in lab:
            return "Set-piece Threat CB"
        return "Stopper CB"

    if pos == "Goalkeeper":
        if has("+passes_att_per90_rw") or has("+passes_cmp_per90_rw"):
            return "Sweeper-keeper"
        return "Traditional GK"

    return f"{pos} Archetype"

# -------------------------
# Public API
# -------------------------
def name_cluster_from_label(pos: str, label: str) -> str:
    pos = _norm(pos)
    key = _canonical_key(label)
    if key in CANON_MAP:
        return CANON_MAP[key]
    _, feats = _parse_label(label)
    return _fallback_name(pos, feats)
