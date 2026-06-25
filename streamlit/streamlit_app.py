"""
IPL 2027 Franchise Squad Predictor — Streamlit Frontend
Production version: API_URL points to Railway deployment
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# ── UPDATE THIS after Railway deployment ──────────────────────────
API_URL = "https://ipl-2027-squad-predictor-production.up.railway.app"
# ─────────────────────────────────────────────────────────────────

ROLE_COLORS = {
    "BAT":"#2196F3","BOWL":"#4CAF50",
    "AR":"#FF9800","WK":"#9C27B0","WK-AR":"#E91E63",
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
    "Mumbai Indians","Chennai Super Kings","Royal Challengers Bengaluru",
    "Kolkata Knight Riders","Delhi Capitals","Punjab Kings",
    "Rajasthan Royals","Sunrisers Hyderabad",
    "Lucknow Super Giants","Gujarat Titans",
]

st.set_page_config(
    page_title="IPL 2027 Squad Predictor",
    page_icon="🏏",
    layout="wide",
)

st.markdown("""
<style>
.main-header{font-size:2.4rem;font-weight:800;color:#1a1a2e;text-align:center;padding:1rem 0 0.2rem 0}
.sub-header{font-size:1rem;color:#666;text-align:center;margin-bottom:2rem}
.player-card{background:white;border-radius:8px;padding:0.8rem 1rem;margin:0.3rem 0;
             border-left:4px solid #ddd;box-shadow:0 1px 3px rgba(0,0,0,0.1)}
.role-badge{padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:600;color:white}
.ovs-badge{background:#ff6b35;color:white;padding:2px 8px;border-radius:12px;font-size:0.75rem;font-weight:600}
</style>
""", unsafe_allow_html=True)

def check_api():
    try:
        r = requests.get(f"{API_URL}/", timeout=5)
        return r.status_code == 200
    except:
        return False

def role_color(role):
    return ROLE_COLORS.get(role, "#607D8B")

# Header
st.markdown('<div class="main-header">🏏 IPL 2027 Franchise Squad Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ML-powered squad selection · XGBoost + Linear Programming<br>'
            'Batting ROC-AUC: 0.86 &nbsp;|&nbsp; Bowling ROC-AUC: 0.88 &nbsp;|&nbsp; Data: IPL 2016–2025</div>',
            unsafe_allow_html=True)

if not check_api():
    st.error("⚠️ API server unreachable. Please try again in a moment.")
    st.stop()

st.success("✓ API connected")
st.divider()

tab1, tab2, tab3 = st.tabs(["🏏 Best XI", "📋 Full Squad", "🔍 Player Search"])

# ── TAB 1: Best XI ────────────────────────────────────────────────
with tab1:
    st.subheader("Select Match Context")
    c1, c2, c3 = st.columns(3)
    venue        = c1.selectbox("Venue", IPL_VENUES)
    opponent     = c2.selectbox("Opponent", IPL_TEAMS)
    max_overseas = c3.slider("Max Overseas", 1, 4, 4)

    if st.button("🎯 Predict Best XI", type="primary", use_container_width=True):
        with st.spinner("Running optimizer..."):
            payload = {
                "venue"        : venue if venue != "Any Venue" else None,
                "opponent"     : opponent if opponent != "Any Opponent" else None,
                "max_overseas" : max_overseas,
            }
            r = requests.post(f"{API_URL}/predict-xi", json=payload, timeout=60)

        if r.status_code != 200:
            st.error(f"Prediction failed: {r.text}")
        else:
            result = r.json()
            xi     = pd.DataFrame(result["xi"])

            st.divider()
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Players",      len(xi))
            m2.metric("Overseas",     result["overseas_count"])
            m3.metric("Pace Bowlers", result["pace_bowlers"])
            m4.metric("Spin Bowlers", result["spin_bowlers"])
            m5.metric("Total Score",  f"{result['total_score']:.3f}")
            st.divider()

            col_l, col_r = st.columns([3,2])
            with col_l:
                st.markdown("### Predicted XI")
                for i, row in xi.iterrows():
                    ovs  = '<span class="ovs-badge">OVS</span>' if row["is_overseas"] else ""
                    rstyle = f'background:{role_color(row["role"])}'
                    st.markdown(
                        f'<div class="player-card">'
                        f'<b>{i+1}. {row["player"]}</b>&nbsp;&nbsp;'
                        f'<span class="role-badge" style="{rstyle}">{row["role"]}</span>&nbsp;'
                        f'<span style="color:#888;font-size:0.85rem">{row["bowling_type"]}</span>&nbsp;{ovs}'
                        f'<span style="float:right;font-weight:600">{row["adjusted_score"]:.3f}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            with col_r:
                fig = px.bar(
                    xi.sort_values("adjusted_score"),
                    x="adjusted_score", y="player",
                    color="role", color_discrete_map=ROLE_COLORS,
                    orientation="h", title="Win Contribution Score",
                    labels={"adjusted_score":"Score","player":""},
                )
                fig.update_layout(height=420,plot_bgcolor="white",
                                  margin=dict(l=0,r=10,t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            c1,c2 = st.columns(2)
            role_counts = xi["role"].value_counts().reset_index()
            role_counts.columns = ["role","count"]
            fig2 = px.pie(role_counts,values="count",names="role",
                          color="role",color_discrete_map=ROLE_COLORS,
                          hole=0.4,title="Role Composition")
            fig2.update_layout(height=280,margin=dict(t=40,b=0,l=0,r=0))
            c1.plotly_chart(fig2,use_container_width=True)

            ps = xi[xi["bowling_type"].isin(["PACE","SPIN"])]["bowling_type"].value_counts().reset_index()
            ps.columns=["type","count"]
            fig3 = px.pie(ps,values="count",names="type",
                          color="type",color_discrete_map={"PACE":"#e74c3c","SPIN":"#3498db"},
                          hole=0.4,title="Bowling Attack")
            fig3.update_layout(height=280,margin=dict(t=40,b=0,l=0,r=0))
            c2.plotly_chart(fig3,use_container_width=True)

            st.download_button("⬇️ Download XI as CSV", xi.to_csv(index=False),
                               "ipl_2027_best_xi.csv","text/csv")

# ── TAB 2: Full Squad ─────────────────────────────────────────────
with tab2:
    st.subheader("25-Player Franchise Squad")
    max_ov = st.slider("Max Overseas Registrations", 4, 8, 8, key="squad_ov")

    if st.button("📋 Generate Full Squad", type="primary", use_container_width=True):
        with st.spinner("Building 25-player squad..."):
            r = requests.post(f"{API_URL}/predict-squad",
                              json={"max_overseas":max_ov,"squad_size":25},timeout=60)

        if r.status_code != 200:
            st.error(f"Failed: {r.text}")
        else:
            result = r.json()
            squad  = pd.DataFrame(result["squad"])

            m1,m2,m3 = st.columns(3)
            m1.metric("Squad Size",     result["squad_size"])
            m2.metric("Overseas Slots", result["overseas_count"])
            m3.metric("Roles",          str(result["role_breakdown"]))
            st.divider()

            for role, label in [("WK","🧤 Wicketkeepers"),("WK-AR","🧤 WK All-rounders"),
                                 ("BAT","🏏 Batsmen"),("AR","⚡ All-rounders"),("BOWL","🎳 Bowlers")]:
                rp = squad[squad["role"]==role]
                if len(rp)==0: continue
                st.markdown(f"**{label}** ({len(rp)})")
                cols = st.columns(min(len(rp),5))
                for col,(_, p) in zip(cols, rp.iterrows()):
                    with col:
                        ovs = "🌍" if p["is_overseas"] else "🇮🇳"
                        st.markdown(
                            f'<div style="background:#f8f9fa;border-radius:8px;padding:0.6rem;'
                            f'text-align:center;border-top:3px solid {role_color(role)}">'
                            f'<div style="font-weight:700;font-size:0.85rem">{p["player"]}</div>'
                            f'<div style="color:#888;font-size:0.75rem">{p["bowling_type"]} {ovs}</div>'
                            f'<div style="font-weight:600">{p["adjusted_score"]:.3f}</div></div>',
                            unsafe_allow_html=True
                        )
                st.markdown("")

            st.divider()
            st.download_button("⬇️ Download Squad as CSV",squad.to_csv(index=False),
                               "ipl_2027_squad_25.csv","text/csv")

# ── TAB 3: Player Search ──────────────────────────────────────────
with tab3:
    st.subheader("Player Score Lookup")
    with st.spinner("Loading player pool..."):
        r = requests.get(f"{API_URL}/players",timeout=30)
    if r.status_code == 200:
        data    = r.json()
        players = pd.DataFrame(data["players"])
        st.caption(f"{data['total']} players in pool")

        search = st.text_input("Search player name", placeholder="e.g. Bumrah, Kohli, Pant")
        if search:
            filtered = players[players["player"].str.lower().str.contains(search.lower())]
            if filtered.empty:
                st.warning("No players found.")
            else:
                for _,row in filtered.iterrows():
                    with st.expander(f"{row['player']}  —  Score: {row['adjusted_score']:.3f}"):
                        c1,c2,c3,c4 = st.columns(4)
                        c1.metric("Role",         row["role"])
                        c2.metric("Type",         row["bowling_type"])
                        c3.metric("Overseas",     "Yes" if row["is_overseas"] else "No")
                        c4.metric("Win Prob",     f"{row['adjusted_score']:.3f}")
        else:
            top = players.head(20).copy()
            top["Overseas"] = top["is_overseas"].map({1:"🌍 OVS",0:"🇮🇳 IND"})
            st.dataframe(
                top[["player","role","bowling_type","Overseas","adjusted_score"]]
                .rename(columns={"player":"Player","role":"Role",
                                 "bowling_type":"Type","adjusted_score":"Score"}),
                use_container_width=True, hide_index=True,
            )
            fig = px.bar(
                players.head(30).sort_values("adjusted_score"),
                x="adjusted_score",y="player",
                color="role",color_discrete_map=ROLE_COLORS,
                orientation="h",title="Top 30 Players — Win Contribution Score",
                labels={"adjusted_score":"Score","player":""},
            )
            fig.update_layout(height=700,plot_bgcolor="white",
                              margin=dict(l=0,r=10,t=40,b=0))
            st.plotly_chart(fig,use_container_width=True)

st.divider()
st.markdown(
    "<div style='text-align:center;color:#aaa;font-size:0.8rem'>"
    "IPL 2027 Squad Predictor · XGBoost + PuLP LP · "
    "Data: IPL 2016–2025 · Batting AUC 0.86 · Bowling AUC 0.88 · "
    "Built by @Anandapriya-T"
    "</div>",
    unsafe_allow_html=True
)
