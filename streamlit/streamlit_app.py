"""
IPL 2027 Franchise Squad Predictor — Streamlit Frontend
========================================================
Calls the FastAPI backend running at localhost:8000
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── CONFIG ────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

ROLE_COLORS = {
    "BAT"  : "#2196F3",
    "BOWL" : "#4CAF50",
    "AR"   : "#FF9800",
    "WK"   : "#9C27B0",
    "WK-AR": "#E91E63",
}

IPL_VENUES = [
    "Any Venue",
    "Wankhede Stadium, Mumbai",
    "M Chinnaswamy Stadium, Bangalore",
    "MA Chidambaram Stadium, Chennai",
    "Eden Gardens, Kolkata",
    "Rajiv Gandhi Intl Stadium, Hyderabad",
    "Sawai Mansingh Stadium, Jaipur",
    "Punjab Cricket Association Stadium, Mohali",
    "Narendra Modi Stadium, Ahmedabad",
    "Arun Jaitley Stadium, Delhi",
    "Dr DY Patil Sports Academy, Mumbai",
]

IPL_TEAMS = [
    "Any Opponent",
    "Mumbai Indians",
    "Chennai Super Kings",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Punjab Kings",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Lucknow Super Giants",
    "Gujarat Titans",
]

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title = "IPL 2027 Squad Predictor",
    page_icon  = "🏏",
    layout     = "wide",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0 0.2rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #2196F3;
    }
    .player-card {
        background: white;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.3rem 0;
        border-left: 4px solid #ddd;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .overseas-badge {
        background: #ff6b35;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .role-badge {
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_URL}/", timeout=3)
        return r.status_code == 200
    except:
        return False

def get_xi(venue=None, opponent=None, max_overseas=4):
    payload = {
        "venue"        : venue if venue != "Any Venue" else None,
        "opponent"     : opponent if opponent != "Any Opponent" else None,
        "max_overseas" : max_overseas,
    }
    r = requests.post(f"{API_URL}/predict-xi", json=payload, timeout=30)
    return r.json() if r.status_code == 200 else None

def get_squad(max_overseas=8):
    r = requests.post(f"{API_URL}/predict-squad",
                      json={"max_overseas": max_overseas, "squad_size": 25},
                      timeout=30)
    return r.json() if r.status_code == 200 else None

def get_player(name):
    r = requests.get(f"{API_URL}/player/{name}", timeout=10)
    return r.json() if r.status_code == 200 else None

def get_all_players():
    r = requests.get(f"{API_URL}/players", timeout=30)
    return r.json() if r.status_code == 200 else None

def role_color(role):
    return ROLE_COLORS.get(role, "#607D8B")


# ── Header ────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏏 IPL 2027 Franchise Squad Predictor</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">ML-powered squad selection using XGBoost + Linear Programming<br>'
            'Batting model ROC-AUC: 0.86 &nbsp;|&nbsp; Bowling model ROC-AUC: 0.88</div>',
            unsafe_allow_html=True)

# ── API status ────────────────────────────────────────────────────
if not check_api():
    st.error("⚠️ API server not running. Start it with: `python run.py` in the app/ folder.")
    st.stop()

st.success("✓ API connected")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🏏 Best XI", "📋 Full Squad", "🔍 Player Search"])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — Best XI
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Select Match Context")

    col1, col2, col3 = st.columns(3)
    with col1:
        venue = st.selectbox("Venue", IPL_VENUES)
    with col2:
        opponent = st.selectbox("Opponent", IPL_TEAMS)
    with col3:
        max_overseas = st.slider("Max Overseas Players", 1, 4, 4)

    if st.button("🎯 Predict Best XI", type="primary", use_container_width=True):
        with st.spinner("Running optimizer..."):
            result = get_xi(venue, opponent, max_overseas)

        if not result:
            st.error("Failed to get prediction. Check the API server.")
        else:
            xi = pd.DataFrame(result["xi"])

            # ── Summary metrics ───────────────────────────────────
            st.divider()
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Players",        len(xi))
            m2.metric("Overseas",       result["overseas_count"])
            m3.metric("Pace Bowlers",   result["pace_bowlers"])
            m4.metric("Spin Bowlers",   result["spin_bowlers"])
            m5.metric("Total Score",    f"{result['total_score']:.3f}")

            st.divider()

            # ── XI display ────────────────────────────────────────
            col_left, col_right = st.columns([3, 2])

            with col_left:
                st.markdown("### Predicted XI")
                for i, row in xi.iterrows():
                    ovs_badge = (
                        '<span class="overseas-badge">OVS</span>'
                        if row["is_overseas"] else ""
                    )
                    role_style = f'background:{role_color(row["role"])}'
                    st.markdown(
                        f'''<div class="player-card">
                            <span style="font-weight:700;font-size:1rem">{i+1}. {row["player"]}</span>
                            &nbsp;&nbsp;
                            <span class="role-badge" style="{role_style}">{row["role"]}</span>
                            &nbsp;
                            <span style="color:#888;font-size:0.85rem">{row["bowling_type"]}</span>
                            &nbsp; {ovs_badge}
                            <span style="float:right;font-weight:600;color:#333">
                                {row["adjusted_score"]:.3f}
                            </span>
                        </div>''',
                        unsafe_allow_html=True
                    )

            with col_right:
                # Bar chart
                fig = px.bar(
                    xi.sort_values("adjusted_score"),
                    x="adjusted_score",
                    y="player",
                    color="role",
                    color_discrete_map=ROLE_COLORS,
                    orientation="h",
                    title="Win Contribution Score",
                    labels={"adjusted_score": "Score", "player": ""},
                )
                fig.update_layout(
                    height=420,
                    showlegend=True,
                    plot_bgcolor="white",
                    margin=dict(l=0, r=10, t=40, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # ── Role composition donut ────────────────────────────
            role_counts = xi["role"].value_counts().reset_index()
            role_counts.columns = ["role", "count"]
            fig2 = px.pie(
                role_counts, values="count", names="role",
                color="role", color_discrete_map=ROLE_COLORS,
                hole=0.4, title="Role Composition"
            )
            fig2.update_layout(height=300, margin=dict(t=40,b=0,l=0,r=0))

            pace_spin = xi[xi["bowling_type"].isin(["PACE","SPIN"])]["bowling_type"].value_counts().reset_index()
            pace_spin.columns = ["type","count"]
            fig3 = px.pie(
                pace_spin, values="count", names="type",
                color="type",
                color_discrete_map={"PACE":"#e74c3c","SPIN":"#3498db"},
                hole=0.4, title="Bowling Attack"
            )
            fig3.update_layout(height=300, margin=dict(t=40,b=0,l=0,r=0))

            c1, c2 = st.columns(2)
            c1.plotly_chart(fig2, use_container_width=True)
            c2.plotly_chart(fig3, use_container_width=True)

            # ── Download ──────────────────────────────────────────
            st.download_button(
                "⬇️ Download XI as CSV",
                xi.to_csv(index=False),
                file_name="ipl_2027_best_xi.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════════
# TAB 2 — Full Squad
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("25-Player Franchise Squad")
    st.caption("Optimised for full season coverage — role balance, overseas limits, age-adjusted scores")

    max_ov_squad = st.slider("Max Overseas Registrations", 4, 8, 8, key="squad_ovs")

    if st.button("📋 Generate Full Squad", type="primary", use_container_width=True):
        with st.spinner("Building 25-player squad..."):
            result = get_squad(max_ov_squad)

        if not result:
            st.error("Failed to get squad. Check the API server.")
        else:
            squad = pd.DataFrame(result["squad"])

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Squad Size",     result["squad_size"])
            m2.metric("Overseas Slots", result["overseas_count"])
            m3.metric("Role Breakdown", str(result["role_breakdown"]))

            st.divider()

            # Group by role
            for role in ["WK", "WK-AR", "BAT", "AR", "BOWL"]:
                role_players = squad[squad["role"] == role]
                if len(role_players) == 0:
                    continue

                role_label = {
                    "WK":"🧤 Wicketkeepers", "WK-AR":"🧤 WK All-rounders",
                    "BAT":"🏏 Batsmen", "AR":"⚡ All-rounders", "BOWL":"🎳 Bowlers"
                }.get(role, role)

                st.markdown(f"**{role_label}** ({len(role_players)})")

                cols = st.columns(len(role_players))
                for col, (_, p) in zip(cols, role_players.iterrows()):
                    with col:
                        ovs = "🌍" if p["is_overseas"] else "🇮🇳"
                        st.markdown(
                            f"""<div style="background:#f8f9fa;border-radius:8px;padding:0.6rem;
                                text-align:center;border-top:3px solid {role_color(role)}">
                                <div style="font-weight:700;font-size:0.85rem">{p['player']}</div>
                                <div style="color:#888;font-size:0.75rem">{p['bowling_type']} {ovs}</div>
                                <div style="font-weight:600;color:#333">{p['adjusted_score']:.3f}</div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                st.markdown("")

            st.divider()
            st.download_button(
                "⬇️ Download Squad as CSV",
                squad.to_csv(index=False),
                file_name="ipl_2027_squad_25.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════════
# TAB 3 — Player Search
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Player Score Lookup")

    with st.spinner("Loading player pool..."):
        all_players_data = get_all_players()

    if all_players_data:
        all_players = pd.DataFrame(all_players_data["players"])
        st.caption(f"{all_players_data['total']} players in pool")

        # Search box
        search = st.text_input("Search player name", placeholder="e.g. Bumrah, Kohli, Pant")

        if search:
            filtered = all_players[
                all_players["player"].str.lower().str.contains(search.lower())
            ]
            if len(filtered) == 0:
                st.warning("No players found.")
            else:
                for _, row in filtered.iterrows():
                    with st.expander(f"{row['player']}  —  Score: {row['adjusted_score']:.3f}"):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Role",          row["role"])
                        c2.metric("Bowling Type",  row["bowling_type"])
                        c3.metric("Overseas",      "Yes" if row["is_overseas"] else "No")
                        c4.metric("Win Prob Score", f"{row['adjusted_score']:.3f}")
        else:
            # Show top 20
            st.markdown("**Top 20 Players by Win Contribution Score**")
            top20 = all_players.head(20).copy()
            top20["Overseas"] = top20["is_overseas"].map({1:"🌍 OVS", 0:"🇮🇳 IND"})
            st.dataframe(
                top20[["player","role","bowling_type","Overseas","adjusted_score"]]
                .rename(columns={
                    "player":"Player","role":"Role",
                    "bowling_type":"Type","adjusted_score":"Score"
                }),
                use_container_width=True,
                hide_index=True,
            )

            # Full leaderboard chart
            fig = px.bar(
                all_players.head(30).sort_values("adjusted_score"),
                x="adjusted_score", y="player",
                color="role", color_discrete_map=ROLE_COLORS,
                orientation="h",
                title="Top 30 Players — Win Contribution Score",
                labels={"adjusted_score":"Score","player":""},
            )
            fig.update_layout(height=700, plot_bgcolor="white",
                              margin=dict(l=0,r=10,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#aaa;font-size:0.8rem'>"
    "IPL 2027 Squad Predictor &nbsp;|&nbsp; "
    "XGBoost + PuLP LP Optimizer &nbsp;|&nbsp; "
    "Data: IPL 2016–2025 &nbsp;|&nbsp; "
    "Batting AUC: 0.86 · Bowling AUC: 0.88"
    "</div>",
    unsafe_allow_html=True
)
