"""
IPL 2027 Franchise Squad Predictor — FastAPI Backend
=====================================================
Endpoints:
    GET  /                      → health check
    GET  /players               → all scored players
    POST /predict-xi            → best XI given match context
    POST /predict-squad         → full 25-player squad
    GET  /player/{name}         → score + SHAP explanation for one player
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import joblib
import shap
import os


BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
# ── CONFIG ────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
MODELS_DIR    = os.path.join(BASE_DIR, '..', 'models')
RAW_DIR       = os.path.join(BASE_DIR, '..', 'data', 'raw')
# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "IPL 2027 Squad Predictor",
    description = "Predicts optimal IPL franchise squad using XGBoost + LP optimizer",
    version     = "1.0.0"
)

# Allow Streamlit to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Load models and data at startup ──────────────────────────────
@app.on_event("startup")
def load_assets():
    global bat_model, bowl_model, bat_features, bowl_features
    global combined_scores, auction_pool
    global bat_explainer, bowl_explainer

    bat_model     = joblib.load(MODELS_DIR + 'batting_model.pkl')
    bowl_model    = joblib.load(MODELS_DIR + 'bowling_model.pkl')
    bat_features  = joblib.load(MODELS_DIR + 'batting_features.pkl')
    bowl_features = joblib.load(MODELS_DIR + 'bowling_features.pkl')

    combined_scores = pd.read_parquet(PROCESSED_DIR + 'combined_player_scores.parquet')
    auction_pool    = pd.read_csv(RAW_DIR + 'auction_pool_2027.csv')

    bat_explainer  = shap.TreeExplainer(bat_model)
    bowl_explainer = shap.TreeExplainer(bowl_model)

    print("✓ Models loaded")
    print("✓ Data loaded")
    print(f"✓ {len(combined_scores)} players scored")
    
    # SHAP explainers
    bat_explainer  = shap.TreeExplainer(bat_model)
    bowl_explainer = shap.TreeExplainer(bowl_model)

    print("✓ Models loaded")
    print("✓ Data loaded")
    print(f"✓ {len(combined_scores)} players scored")


# ── Helper: build eligible pool ──────────────────────────────────
def build_pool(venue: Optional[str] = None, opponent: Optional[str] = None):
    """
    Builds the eligible player pool with roles, availability,
    age decay, and overseas flags applied.
    """
    from pulp import (LpProblem, LpMaximize, LpVariable,
                      lpSum, LpBinary, value, PULP_CBC_CMD)

    pool = combined_scores.copy()

    # Availability filter
    available = set(auction_pool[auction_pool['available'] == 1]['player_name'])
    pool = pool[pool['player'].isin(available)].copy()

    # Merge auction pool metadata
    pool = pool.merge(
        auction_pool[['player_name', 'bowling_type', 'nationality']],
        left_on='player', right_on='player_name', how='left'
    ).drop(columns=['player_name'])
    pool['bowling_type'] = pool['bowling_type'].fillna('BAT')

    # Wicketkeepers
    KEEPERS = {
        'RR Pant','KL Rahul','Dhruv Jurel','H Klaasen',
        'Q de Kock','Ishan Kishan','SV Samson','KS Bharat',
        'N Pooran','PD Salt','JM Bairstow',
    }
    pool['is_wk'] = pool['player'].isin(KEEPERS).astype(int)

    # Role classification
    def classify_role(row):
        bat_ok  = pd.notna(row.get('bat_win_prob'))
        bowl_ok = pd.notna(row.get('bowl_win_prob'))
        if row['is_wk'] == 1:
            return 'WK-AR' if (bat_ok and bowl_ok) else 'WK'
        if bat_ok and bowl_ok:
            return 'AR'
        elif bat_ok:
            return 'BAT'
        elif bowl_ok:
            return 'BOWL'
        return 'UNKNOWN'

    pool['role'] = pool.apply(classify_role, axis=1)
    pool = pool[pool['role'] != 'UNKNOWN'].copy()

    # Age decay
    pool['age_decay'] = pool.get('bat_last_season', 2020).apply(
        lambda s: 1.0 if s >= 2020 else (0.95 if s >= 2016 else 0.85)
    )
    pool['adjusted_score'] = pool['combined_score'] * pool['age_decay']

    # Overseas flag
    pool['is_overseas'] = (pool['nationality'] != 'Indian').astype(int)

    return pool[pool['adjusted_score'].notna() & (pool['adjusted_score'] > 0)].copy()


def run_lp_optimizer(pool, n=11, max_overseas=4):
    """
    LP optimizer — selects best N players subject to IPL constraints.
    """
    from pulp import (LpProblem, LpMaximize, LpVariable,
                      lpSum, LpBinary, value, PULP_CBC_CMD)

    df  = pool.reset_index(drop=True)
    idx = range(len(df))
    prob = LpProblem("IPL_XI", LpMaximize)
    x    = [LpVariable(f"x_{i}", cat=LpBinary) for i in idx]

    prob += lpSum(df.loc[i, 'adjusted_score'] * x[i] for i in idx)
    prob += lpSum(x[i] for i in idx) == n

    wk_i   = [i for i in idx if df.loc[i,'role'] in ('WK','WK-AR')]
    bat_i  = [i for i in idx if df.loc[i,'role'] in ('BAT','WK','WK-AR')]
    bowl_i = [i for i in idx if df.loc[i,'role'] in ('BOWL','AR','WK-AR')]
    ar_i   = [i for i in idx if df.loc[i,'role'] in ('AR','WK-AR')]
    ov_i   = [i for i in idx if df.loc[i,'is_overseas'] == 1]
    ov_bat = [i for i in idx if df.loc[i,'is_overseas']==1 and df.loc[i,'role'] in ('BAT','WK')]
    pace_i = [i for i in idx if df.loc[i,'bowling_type'] == 'PACE']
    spin_i = [i for i in idx if df.loc[i,'bowling_type'] == 'SPIN']

    if wk_i:
        prob += lpSum(x[i] for i in wk_i) >= 1
        prob += lpSum(x[i] for i in wk_i) <= 2
    prob += lpSum(x[i] for i in bat_i)  >= 3
    prob += lpSum(x[i] for i in bat_i)  <= 6
    prob += lpSum(x[i] for i in bowl_i) >= 4
    prob += lpSum(x[i] for i in bowl_i) <= 7
    if ar_i:   prob += lpSum(x[i] for i in ar_i)   >= 1
    prob += lpSum(x[i] for i in ov_i)   <= max_overseas
    if ov_bat: prob += lpSum(x[i] for i in ov_bat) <= 2
    if len(pace_i) >= 2: prob += lpSum(x[i] for i in pace_i) >= 2
    if spin_i: prob += lpSum(x[i] for i in spin_i) >= 1

    prob.solve(PULP_CBC_CMD(msg=0))
    return df[[value(x[i]) == 1 for i in idx]].copy()


# ── Request / Response models ─────────────────────────────────────
class PredictXIRequest(BaseModel):
    venue        : Optional[str] = None
    opponent     : Optional[str] = None
    max_overseas : Optional[int] = 4

class PredictSquadRequest(BaseModel):
    max_overseas : Optional[int] = 8
    squad_size   : Optional[int] = 25


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/")
def health():
    return {
        "status" : "running",
        "model"  : "IPL 2027 Squad Predictor v1.0",
        "endpoints": ["/players", "/predict-xi", "/predict-squad", "/player/{name}"]
    }


@app.get("/players")
def get_all_players():
    """Returns all scored players sorted by combined win contribution score."""
    pool = build_pool()
    result = (
        pool[['player','role','bowling_type','is_overseas','adjusted_score']]
        .sort_values('adjusted_score', ascending=False)
        .round(4)
        .to_dict(orient='records')
    )
    return {"total": len(result), "players": result}


@app.post("/predict-xi")
def predict_xi(req: PredictXIRequest):
    """
    Returns the optimal XI for a given match context.
    venue and opponent are optional — used for future context-aware filtering.
    """
    pool = build_pool(venue=req.venue, opponent=req.opponent)

    if len(pool) < 11:
        raise HTTPException(status_code=400, detail="Not enough eligible players to form XI")

    xi = run_lp_optimizer(pool, n=11, max_overseas=req.max_overseas)
    xi = xi.sort_values('adjusted_score', ascending=False)

    result = xi[['player','role','bowling_type','is_overseas','adjusted_score']].round(4)

    return {
        "venue"            : req.venue or "Any",
        "opponent"         : req.opponent or "Any",
        "overseas_count"   : int(xi['is_overseas'].sum()),
        "pace_bowlers"     : int((xi['bowling_type'] == 'PACE').sum()),
        "spin_bowlers"     : int((xi['bowling_type'] == 'SPIN').sum()),
        "total_score"      : round(float(xi['adjusted_score'].sum()), 4),
        "xi"               : result.to_dict(orient='records'),
    }


@app.post("/predict-squad")
def predict_squad(req: PredictSquadRequest):
    """Returns the optimal 25-player franchise squad."""
    from pulp import (LpProblem, LpMaximize, LpVariable,
                      lpSum, LpBinary, value, PULP_CBC_CMD)

    pool = build_pool()
    df   = pool.reset_index(drop=True)
    idx  = range(len(df))

    prob = LpProblem("IPL_Squad", LpMaximize)
    x    = [LpVariable(f"x_{i}", cat=LpBinary) for i in idx]

    prob += lpSum(df.loc[i,'adjusted_score'] * x[i] for i in idx)

    wk_i   = [i for i in idx if df.loc[i,'role'] in ('WK','WK-AR')]
    bat_i  = [i for i in idx if df.loc[i,'role'] in ('BAT','WK','WK-AR')]
    bowl_i = [i for i in idx if df.loc[i,'role'] in ('BOWL','AR','WK-AR')]
    ar_i   = [i for i in idx if df.loc[i,'role'] in ('AR','WK-AR')]
    ov_i   = [i for i in idx if df.loc[i,'is_overseas'] == 1]
    pace_i = [i for i in idx if df.loc[i,'bowling_type'] == 'PACE']

    prob += lpSum(x[i] for i in idx)     == req.squad_size
    prob += lpSum(x[i] for i in wk_i)   >= 2
    prob += lpSum(x[i] for i in wk_i)   <= 4
    prob += lpSum(x[i] for i in bat_i)  >= 7
    prob += lpSum(x[i] for i in bowl_i) >= 9
    prob += lpSum(x[i] for i in ar_i)   >= 2
    prob += lpSum(x[i] for i in ov_i)   <= req.max_overseas
    if len(pace_i) >= 4:
        prob += lpSum(x[i] for i in pace_i) >= 4

    prob.solve(PULP_CBC_CMD(msg=0))
    squad = df[[value(x[i]) == 1 for i in idx]].copy()
    squad = squad.sort_values(['role','adjusted_score'], ascending=[True,False])

    return {
        "squad_size"     : len(squad),
        "overseas_count" : int(squad['is_overseas'].sum()),
        "role_breakdown" : squad['role'].value_counts().to_dict(),
        "squad"          : squad[['player','role','bowling_type','is_overseas','adjusted_score']]
                           .round(4).to_dict(orient='records'),
    }


@app.get("/player/{player_name}")
def get_player(player_name: str):
    """
    Returns win contribution score + top SHAP driver for a specific player.
    URL-encode spaces as %20 e.g. /player/JJ%20Bumrah
    """
    pool = build_pool()
    match = pool[pool['player'].str.lower() == player_name.lower()]

    if match.empty:
        # Try partial match
        match = pool[pool['player'].str.lower().str.contains(player_name.lower())]
        if match.empty:
            raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")

    row = match.iloc[0]

    result = {
        "player"          : row['player'],
        "role"            : row['role'],
        "bowling_type"    : row.get('bowling_type', 'N/A'),
        "is_overseas"     : int(row['is_overseas']),
        "adjusted_score"  : round(float(row['adjusted_score']), 4),
        "bat_win_prob"    : round(float(row['bat_win_prob']), 4) if pd.notna(row.get('bat_win_prob')) else None,
        "bowl_win_prob"   : round(float(row['bowl_win_prob']), 4) if pd.notna(row.get('bowl_win_prob')) else None,
    }

    return result
