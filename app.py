import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import math
from scipy.stats import norm

# Page configuration
st.set_page_config(page_title="Mini simulateur de produits financier", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stMetric { background-color: #1e2129; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Mini simulateur de produits financier")

# --- Mathematical Model: Black-Scholes ---
def black_scholes_call(S, K, T, sigma, r=0.0):
    if T <= 0:
        return max(0, S - K)
    if sigma <= 0:
        return max(0, S - K)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# --- STEP 1: Global Selection ---
st.sidebar.header("🎯 Choix des Titres")
ticker1_input = st.sidebar.text_input("Ticker 1", value="MC.PA")
ticker2_input = st.sidebar.text_input("Ticker 2", value="^FCHI")

period_options = {
    "1 mois": "1mo", "3 mois": "3mo", "6 mois": "6mo",
    "1 an": "1y", "2 ans": "2y", "5 ans": "5y",
    "YTD": "ytd", "Max": "max"
}
period_label = st.sidebar.selectbox("Période d'Analyse des Marchés", options=list(period_options.keys()), index=3)
period_code = period_options[period_label]

# --- STEP 2: Fetch Data ---
@st.cache_data
def get_data(tickers, period):
    data = yf.download(tickers, period=period, interval="1d")
    return data

with st.spinner('Chargement des données...'):
    data = get_data([ticker1_input, ticker2_input], "max")

if data.empty:
    st.error("Aucune donnée trouvée. Veuillez vérifier les tickers.")
    st.stop()

# Preparation
if isinstance(data.columns, pd.MultiIndex):
    close_prices = data['Close']
else:
    close_prices = data[['Close']]
close_prices.index = pd.to_datetime(close_prices.index)

today = close_prices.index[-1]
if period_code == "1y": start_mkt = today - timedelta(days=365)
elif period_code == "1mo": start_mkt = today - timedelta(days=30)
elif period_code == "3mo": start_mkt = today - timedelta(days=90)
elif period_code == "6mo": start_mkt = today - timedelta(days=180)
elif period_code == "2y": start_mkt = today - timedelta(days=730)
elif period_code == "5y": start_mkt = today - timedelta(days=5*365)
elif period_code == "ytd": start_mkt = datetime(today.year, 1, 1)
else: start_mkt = close_prices.index[0]

analysis_prices = close_prices[close_prices.index >= pd.to_datetime(start_mkt)]

# --- STEP 3: Sidebar Simulation Settings ---
st.sidebar.divider()
st.sidebar.header("⚙️ Configuration Simulation")

def get_price_at_date(series, target_date):
    available_dates = series.dropna().index
    valid_dates = available_dates[available_dates <= pd.to_datetime(target_date)]
    if len(valid_dates) == 0:
        return series.dropna().iloc[0]
    return series.loc[valid_dates[-1]]

def sidebar_simulation_params(ticker, series, key_suffix):
    st.sidebar.subheader(f"Stratégie {ticker}")
    invest = st.sidebar.number_input(f"Somme investie ({ticker})", value=1000, key=f"inv_{key_suffix}")
    
    min_date = series.index[0].to_pydatetime()
    max_date = series.index[-1].to_pydatetime()
    default_date = max(min_date, (datetime.now() - timedelta(days=365)))
    date_p = st.sidebar.date_input(f"Date d'achat", value=default_date, min_value=min_date, max_value=max_date, key=f"date_{key_suffix}")
    
    price_at_purchase = get_price_at_date(series, date_p)
    default_strike_5x = float(price_at_purchase * 0.8)
    
    with st.sidebar.expander(f"Produits Dérivés {ticker}"):
        st.markdown(f"*(Sous-jacent à l'achat : {price_at_purchase:,.2f} €)*")
        
        st.markdown("**Turbo Call**")
        t_strike = st.number_input("Strike Turbo", value=default_strike_5x, key=f"tst_{key_suffix}")
        t_ratio = st.number_input("Parité Turbo", value=10.0, key=f"tra_{key_suffix}")
        
        st.markdown("---")
        st.markdown("**Warrant Call (BS Model)**")
        w_strike = st.number_input("Strike Warrant", value=default_strike_5x, key=f"wst_{key_suffix}")
        w_ratio = st.number_input("Parité Warrant", value=10.0, key=f"wra_{key_suffix}")
        w_vol = st.slider("Volatilité Implicite (%)", 5, 100, 25, key=f"wvo_{key_suffix}")
        w_expiry = st.date_input("Date d'échéance", value=date_p + timedelta(days=365), key=f"wex_{key_suffix}")
        
    return invest, date_p, t_strike, t_ratio, w_strike, w_ratio, w_vol, w_expiry

p1_config = sidebar_simulation_params(ticker1_input, close_prices[ticker1_input], "t1")
p2_config = sidebar_simulation_params(ticker2_input, close_prices[ticker2_input], "t2")

# --- STEP 4: Render Market Analysis ---
st.header("📈 Analyse des Marchés")

t1_s = analysis_prices[ticker1_input].dropna().iloc[0]
p_ev1 = (analysis_prices[ticker1_input] / t1_s - 1) * 100
t2_s = analysis_prices[ticker2_input].dropna().iloc[0]
p_ev2 = (analysis_prices[ticker2_input] / t2_s - 1) * 100

fig_mkt = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[f"Cours {ticker1_input}", f"Cours {ticker2_input}"],
                        specs=[[{"secondary_y": True}], [{"secondary_y": True}]])

def add_perf_layer(fig, row, series, name):
    fig.add_trace(go.Scatter(x=series.index, y=series, name=f"Var {name}", mode='lines', 
                             line=dict(color='white', width=1, dash='dot')), row=row, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=series.index, y=series.map(lambda x: x if x > 0 else 0), fill='tozeroy', 
                             fillcolor='rgba(0,255,0,0.1)', line=dict(width=0), showlegend=False), row=row, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=series.index, y=series.map(lambda x: x if x < 0 else 0), fill='tozeroy', 
                             fillcolor='rgba(255,0,0,0.1)', line=dict(width=0), showlegend=False), row=row, col=1, secondary_y=True)

fig_mkt.add_trace(go.Scatter(x=analysis_prices.index, y=analysis_prices[ticker1_input], name=ticker1_input), row=1, col=1, secondary_y=False)
add_perf_layer(fig_mkt, 1, p_ev1, ticker1_input)
fig_mkt.add_trace(go.Scatter(x=analysis_prices.index, y=analysis_prices[ticker2_input], name=ticker2_input), row=2, col=1, secondary_y=False)
add_perf_layer(fig_mkt, 2, p_ev2, ticker2_input)

fig_mkt.update_layout(height=600, template="plotly_dark", showlegend=False, hovermode="x unified")
fig_mkt.update_yaxes(title_text="Prix (€)", secondary_y=False)
fig_mkt.update_yaxes(title_text="Var %", secondary_y=True, zerolinecolor='white')
st.plotly_chart(fig_mkt, use_container_width=True, key="mkt_analysis")

diff = p_ev1.iloc[-1] - p_ev2.iloc[-1]
st.info(f"Performance Relative : **{ticker1_input}** {'surperforme' if diff > 0 else 'sous-performe'} **{ticker2_input}** de **{abs(diff):.2f}%**.")

st.divider()

# --- STEP 5: Simulation Logic ---
st.header("🧪 Simulateur de Produits")

def run_simulation(ticker, params):
    invest, date_p, t_strike, t_ratio, w_strike, w_ratio, w_vol, w_expiry = params
    
    mask = close_prices.index >= pd.to_datetime(date_p)
    prices = close_prices[ticker].loc[mask]
    if prices.empty: return None, None, None, None, 0, 0
    
    start_price = prices.iloc[0]
    shares_stock = invest / start_price
    sim_stock = prices * shares_stock
    
    # 1. Turbo Simulation
    t_val_unit = (prices - t_strike) / t_ratio
    t_val_unit = t_val_unit.apply(lambda x: max(0, x))
    ko_mask = (prices <= t_strike).cummax()
    t_val_unit[ko_mask] = 0
    
    t_val_start = max(0, (start_price - t_strike) / t_ratio)
    shares_turbo = invest / t_val_start if t_val_start > 0 else 0
    sim_turbo = t_val_unit * shares_turbo
    
    # 2. Warrant Simulation (Black-Scholes)
    def calc_warrant_series(row_price, current_date):
        expiry_dt = pd.to_datetime(w_expiry)
        curr_dt = pd.to_datetime(current_date)
        
        if curr_dt >= expiry_dt:
            # Freeze at intrinsic value on the last available price before or at expiry
            price_at_expiry = get_price_at_date(close_prices[ticker], w_expiry)
            return max(0, price_at_expiry - w_strike) / w_ratio
            
        days_to_expiry = (expiry_dt - curr_dt).days
        T = max(0, days_to_expiry / 365.0)
        return black_scholes_call(row_price, w_strike, T, w_vol/100.0) / w_ratio

    w_val_unit = pd.Series([calc_warrant_series(p, d) for p, d in zip(prices, prices.index)], index=prices.index)
    w_val_start = calc_warrant_series(start_price, date_p)
    shares_warrant = invest / w_val_start if w_val_start > 0 else 0
    sim_warrant = w_val_unit * shares_warrant
    
    # Leverage at start
    lev_t = (start_price / (t_val_start * t_ratio)) if (t_val_start > 0) else 0
    # Effective Leverage (Omega) for Warrant = Delta * S / WarrantPrice
    # Simplified approximation for display
    lev_w = (start_price / (w_val_start * w_ratio)) if (w_val_start > 0) else 0
    
    return pd.DataFrame({'Action': sim_stock, 'Turbo': sim_turbo, 'Warrant': sim_warrant, 'Underlying': prices}, index=prices.index), start_price, t_val_start, w_val_start, lev_t, lev_w

sim1, sp1, t1v, w1v, lt1, lw1 = run_simulation(ticker1_input, p1_config)
sim2, sp2, t2v, w2v, lt2, lw2 = run_simulation(ticker2_input, p2_config)

# Display Context
cols_ctx = st.columns(2)
with cols_ctx[0]:
    if sp1:
        st.markdown(f"#### 📅 Contexte {ticker1_input} au {p1_config[1]}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Cours Sous-jacent", f"{sp1:,.2f} €")
        c2.metric("Val. Init Turbo", f"{t1v:,.2f} €", delta=f"Levier: {lt1:.1f}x", delta_color="normal")
        c3.metric("Val. Init Warrant", f"{w1v:,.2f} €", delta=f"Levier: {lw1:.1f}x", delta_color="normal")

with cols_ctx[1]:
    if sp2:
        st.markdown(f"#### 📅 Contexte {ticker2_input} au {p2_config[1]}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Cours Sous-jacent", f"{sp2:,.2f} €")
        c2.metric("Val. Init Turbo", f"{t2v:,.2f} €", delta=f"Levier: {lt2:.1f}x", delta_color="normal")
        c3.metric("Val. Init Warrant", f"{w2v:,.2f} €", delta=f"Levier: {lw2:.1f}x", delta_color="normal")

def plot_sim(sim_df, ticker, t_strike, w_strike):
    if sim_df is None: 
        st.warning(f"Pas de données disponibles pour {ticker} à cette date.")
        return
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=sim_df.index, y=sim_df['Underlying'], name='Cours Sous-jacent', 
                             line=dict(color='rgba(255, 255, 255, 0.3)', width=1.5), opacity=0.8), secondary_y=True)
    
    fig.add_trace(go.Scatter(x=sim_df.index, y=sim_df['Action'], name='Action', line=dict(color='rgba(52, 152, 219, 0.8)')), secondary_y=False)
    fig.add_trace(go.Scatter(x=sim_df.index, y=sim_df['Turbo'], name='Turbo (KO)', line=dict(color='rgba(231, 76, 60, 0.9)', width=2.5)), secondary_y=False)
    fig.add_trace(go.Scatter(x=sim_df.index, y=sim_df['Warrant'], name='Warrant (BS)', line=dict(color='rgba(241, 196, 15, 0.8)')), secondary_y=False)
    
    # Strike lines tied to secondary Y
    fig.add_trace(go.Scatter(x=[sim_df.index[0], sim_df.index[-1]], y=[t_strike, t_strike], name="Strike Turbo",
                             mode='lines', line=dict(color='rgba(231, 76, 60, 0.3)', dash='dash')), secondary_y=True)
    fig.add_trace(go.Scatter(x=[sim_df.index[0], sim_df.index[-1]], y=[w_strike, w_strike], name="Strike Warrant",
                             mode='lines', line=dict(color='rgba(241, 196, 15, 0.3)', dash='dot')), secondary_y=True)
    
    fig.update_layout(title=f"Evolution de l'investissement ({ticker})", height=500, template="plotly_dark",
                      hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Valeur Investissement (€)", secondary_y=False)
    fig.update_yaxes(title_text="Cours Sous-jacent (€)", secondary_y=True, showgrid=False)
    
    st.plotly_chart(fig, use_container_width=True, key=f"sim_{ticker}")

col_res = st.columns(2)
with col_res[0]: plot_sim(sim1, ticker1_input, p1_config[2], p1_config[4])
with col_res[1]: plot_sim(sim2, ticker2_input, p2_config[2], p2_config[4])

# --- Methodology Section ---
st.divider()
with st.expander("📚 Méthodologie et Détails des Calculs"):
    st.markdown("""
    ### 1. Modèle Black-Scholes (Warrants)
    Le prix du Warrant est calculé à l'aide de la formule de Black-Scholes pour un Call :
    $$C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)$$
    
    Où :
    - **$S$ (Spot)** : Cours actuel du sous-jacent.
    - **$K$ (Strike)** : Prix d'exercice du warrant.
    - **$T$ (Time)** : Temps restant jusqu'à l'échéance (en années).
    - **$\sigma$ (Sigma)** : Volatilité implicite.
    - **$r$** : Taux d'intérêt sans risque (fixé à 0% dans ce simulateur).
    - **$N(x)$** : Fonction de répartition de la loi normale standard.
    
    ### 2. Les "Grecques" simulées
    - **Theta (Érosion Temporelle)** : Représente la perte de valeur du warrant due au passage du temps. Plus $T$ diminue, plus $C$ baisse, toutes choses égales par ailleurs. Cette perte s'accélère à l'approche de l'échéance.
    - **Vega (Sensibilité Volatilité)** : Mesure l'impact d'un changement de volatilité implicite. Une hausse de $\sigma$ augmente la probabilité que le warrant finisse "dans la monnaie", augmentant ainsi son prix.
    - **L'Échéance** : À la date d'échéance, la valeur du warrant est gelée à sa valeur intrinsèque : $\max(0, S_{\text{échéance}} - K)$.
    
    ### 3. Modèle Turbo (Barrière Désactivante)
    Contrairement au Warrant, le Turbo a une valeur linéaire :
    $$\text{Prix} = \frac{\max(0, S - \text{Strike})}{\text{Parité}}$$
    
    **Knock-Out (KO)** : Si à n'importe quel moment $S \leq \text{Strike}$, le produit est immédiatement désactivé et sa valeur devient définitivement **0 €**.
    """)
