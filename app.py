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

# --- Mathematical Models ---
def black_scholes(S, K, T, sigma, type='Call', r=0.0):
    if T <= 0:
        return max(0, S - K) if type == 'Call' else max(0, K - S)
    if sigma <= 0:
        return max(0, S - K) if type == 'Call' else max(0, K - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if type == 'Call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else: # Put
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

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
    
    with st.sidebar.expander(f"Produits Dérivés {ticker}"):
        st.markdown(f"*(Sous-jacent à l'achat : {price_at_purchase:,.2f} €)*")
        
        st.markdown("**Turbo**")
        t_type = st.selectbox("Type Turbo", ["Call", "Put"], key=f"tty_{key_suffix}")
        default_t_strike = float(price_at_purchase * 0.8) if t_type == "Call" else float(price_at_purchase * 1.2)
        t_strike = st.number_input(f"Strike Turbo ({t_type})", value=default_t_strike, key=f"tst_{key_suffix}")
        t_ratio = st.number_input("Parité Turbo", value=10.0, key=f"tra_{key_suffix}")
        
        st.markdown("---")
        st.markdown("**Warrant (BS Model)**")
        w_type = st.selectbox("Type Warrant", ["Call", "Put"], key=f"wty_{key_suffix}")
        default_w_strike = float(price_at_purchase * 0.8) if w_type == "Call" else float(price_at_purchase * 1.2)
        w_strike = st.number_input(f"Strike Warrant ({w_type})", value=default_w_strike, key=f"wst_{key_suffix}")
        w_ratio = st.number_input("Parité Warrant", value=10.0, key=f"wra_{key_suffix}")
        w_vol = st.slider("Volatilité Implicite (%)", 5, 100, 25, key=f"wvo_{key_suffix}")
        w_expiry = st.date_input("Date d'échéance", value=date_p + timedelta(days=365), key=f"wex_{key_suffix}")
        
    return {
        'invest': invest, 'date_p': date_p, 
        't_type': t_type, 't_strike': t_strike, 't_ratio': t_ratio,
        'w_type': w_type, 'w_strike': w_strike, 'w_ratio': w_ratio,
        'w_vol': w_vol, 'w_expiry': w_expiry
    }

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

def run_simulation(ticker, cfg):
    date_p = cfg['date_p']
    mask = close_prices.index >= pd.to_datetime(date_p)
    prices = close_prices[ticker].loc[mask]
    if prices.empty: return None, None, None, None, 0, 0
    
    start_price = prices.iloc[0]
    shares_stock = cfg['invest'] / start_price
    sim_stock = prices * shares_stock
    
    # 1. Turbo Simulation
    t_strike = cfg['t_strike']
    t_ratio = cfg['t_ratio']
    if cfg['t_type'] == "Call":
        t_val_unit = (prices - t_strike) / t_ratio
        t_val_start = max(0, (start_price - t_strike) / t_ratio)
        ko_mask = (prices <= t_strike).cummax()
    else: # Put
        t_val_unit = (t_strike - prices) / t_ratio
        t_val_start = max(0, (t_strike - start_price) / t_ratio)
        ko_mask = (prices >= t_strike).cummax()
    
    t_val_unit = t_val_unit.apply(lambda x: max(0, x))
    t_val_unit[ko_mask] = 0
    shares_turbo = cfg['invest'] / t_val_start if t_val_start > 0 else 0
    sim_turbo = t_val_unit * shares_turbo
    
    # 2. Warrant Simulation (Black-Scholes)
    w_strike = cfg['w_strike']
    w_ratio = cfg['w_ratio']
    w_expiry = cfg['w_expiry']
    w_vol = cfg['w_vol']
    w_type = cfg['w_type']

    def calc_warrant_full(row_price, current_date):
        expiry_dt = pd.to_datetime(w_expiry)
        curr_dt = pd.to_datetime(current_date)
        
        if w_type == "Call":
            intrinsic = max(0, row_price - w_strike) / w_ratio
        else:
            intrinsic = max(0, w_strike - row_price) / w_ratio
            
        if curr_dt >= expiry_dt:
            price_at_expiry = get_price_at_date(close_prices[ticker], w_expiry)
            if w_type == "Call":
                final_val = max(0, price_at_expiry - w_strike) / w_ratio
            else:
                final_val = max(0, w_strike - price_at_expiry) / w_ratio
            return final_val, final_val, 0
            
        days_to_expiry = (expiry_dt - curr_dt).days
        T = max(0, days_to_expiry / 365.0)
        full_price = black_scholes(row_price, w_strike, T, w_vol/100.0, type=w_type) / w_ratio
        time_value = max(0, full_price - intrinsic)
        
        return full_price, intrinsic, time_value

    w_stats = [calc_warrant_full(p, d) for p, d in zip(prices, prices.index)]
    w_val_unit = pd.Series([s[0] for s in w_stats], index=prices.index)
    w_intrinsic_unit = pd.Series([s[1] for s in w_stats], index=prices.index)
    w_time_unit = pd.Series([s[2] for s in w_stats], index=prices.index)
    
    w_val_start, _, _ = calc_warrant_full(start_price, date_p)
    shares_warrant = cfg['invest'] / w_val_start if w_val_start > 0 else 0
    
    sim_warrant = w_val_unit * shares_warrant
    sim_w_intrinsic = w_intrinsic_unit * shares_warrant
    sim_w_time = w_time_unit * shares_warrant
    
    # Leverage at start
    lev_t = (start_price / (t_val_start * t_ratio)) if (t_val_start > 0) else 0
    lev_w = (start_price / (w_val_start * w_ratio)) if (w_val_start > 0) else 0
    
    return pd.DataFrame({
        'Action': sim_stock, 
        'Turbo': sim_turbo, 
        'Warrant': sim_warrant, 
        'W_Intrinsic': sim_w_intrinsic, 
        'W_Time': sim_w_time, 
        'Underlying': prices
    }, index=prices.index), start_price, t_val_start, w_val_start, lev_t, lev_w

sim1, sp1, t1v, w1v, lt1, lw1 = run_simulation(ticker1_input, p1_config)
sim2, sp2, t2v, w2v, lt2, lw2 = run_simulation(ticker2_input, p2_config)

# Display Context
cols_ctx = st.columns(2)
with cols_ctx[0]:
    if sp1:
        st.markdown(f"#### 📅 Contexte {ticker1_input} au {p1_config['date_p']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Cours Sous-jacent", f"{sp1:,.2f} €")
        c2.metric(f"Val. Init Turbo {p1_config['t_type']}", f"{t1v:,.2f} €", delta=f"Levier: {lt1:.1f}x", delta_color="normal")
        c3.metric(f"Val. Init Warrant {p1_config['w_type']}", f"{w1v:,.2f} €", delta=f"Levier: {lw1:.1f}x", delta_color="normal")

with cols_ctx[1]:
    if sp2:
        st.markdown(f"#### 📅 Contexte {ticker2_input} au {p2_config['date_p']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Cours Sous-jacent", f"{sp2:,.2f} €")
        c2.metric(f"Val. Init Turbo {p2_config['t_type']}", f"{t2v:,.2f} €", delta=f"Levier: {lt2:.1f}x", delta_color="normal")
        c3.metric(f"Val. Init Warrant {p2_config['w_type']}", f"{w1v:,.2f} €", delta=f"Levier: {lw2:.1f}x", delta_color="normal")

def plot_sim(sim_df, ticker, t_cfg, w_cfg):
    if sim_df is None: 
        st.warning(f"Pas de données disponibles pour {ticker} à cette date.")
        return
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=sim_df.index, y=sim_df['Underlying'], name='Cours Sous-jacent', 
                             line=dict(color='rgba(255, 255, 255, 0.3)', width=1.5), opacity=0.8), secondary_y=True)
    
    fig.add_trace(go.Scatter(x=sim_df.index, y=sim_df['Action'], name='Action', line=dict(color='rgba(52, 152, 219, 0.8)')), secondary_y=False)
    fig.add_trace(go.Scatter(x=sim_df.index, y=sim_df['Turbo'], name=f"Turbo {t_cfg['t_type']} (KO)", line=dict(color='rgba(231, 76, 60, 0.9)', width=2.5)), secondary_y=False)
    fig.add_trace(go.Scatter(x=sim_df.index, y=sim_df['Warrant'], name=f"Warrant {w_cfg['w_type']} (BS)", line=dict(color='rgba(241, 196, 15, 0.8)')), secondary_y=False)
    
    # Time Value decomposition (Theta potential)
    fig.add_trace(go.Scatter(x=sim_df.index, y=sim_df['W_Time'], name='Valeur Temps (Theta)', 
                             line=dict(color='rgba(241, 196, 15, 0.4)', dash='dot'), fill='tozeroy'), secondary_y=False)
    
    # Strike lines tied to secondary Y
    fig.add_trace(go.Scatter(x=[sim_df.index[0], sim_df.index[-1]], y=[t_cfg['t_strike'], t_cfg['t_strike']], name="Strike Turbo",
                             mode='lines', line=dict(color='rgba(231, 76, 60, 0.3)', dash='dash')), secondary_y=True)
    fig.add_trace(go.Scatter(x=[sim_df.index[0], sim_df.index[-1]], y=[w_cfg['w_strike'], w_cfg['w_strike']], name="Strike Warrant",
                             mode='lines', line=dict(color='rgba(241, 196, 15, 0.3)', dash='dot')), secondary_y=True)
    
    fig.update_layout(title=f"Evolution de l'investissement ({ticker})", height=500, template="plotly_dark",
                      hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Valeur Investissement (€)", secondary_y=False)
    fig.update_yaxes(title_text="Cours Sous-jacent (€)", secondary_y=True, showgrid=False)
    
    st.plotly_chart(fig, use_container_width=True, key=f"sim_{ticker}")

col_res = st.columns(2)
with col_res[0]: plot_sim(sim1, ticker1_input, p1_config, p1_config)
with col_res[1]: plot_sim(sim2, ticker2_input, p2_config, p2_config)

# --- Methodology Section ---
st.divider()
with st.expander("📚 Méthodologie et Détails des Calculs"):
    st.markdown(r"""
    ### 1. Modèle Black-Scholes (Warrants)
    Le prix du Warrant est calculé à l'aide de la formule de Black-Scholes :
    - **Call** : $C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)$
    - **Put** : $P = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)$
    
    Où :
    - **$S$ (Spot)** : Cours actuel du sous-jacent.
    - **$K$ (Strike)** : Prix d'exercice du warrant.
    - **$T$ (Time)** : Temps restant jusqu'à l'échéance (en années).
    - **$\sigma$ (Sigma)** : Volatilité implicite.
    - **$N(x)$** : Fonction de répartition de la loi normale standard.
    
    ### 2. Les "Grecques" simulées
    - **Theta (Érosion Temporelle)** : Représente la perte de valeur du warrant due au passage du temps. Cette perte s'accélère à l'approche de l'échéance.
    - **Vega (Sensibilité Volatilité)** : Mesure l'impact d'un changement de volatilité implicite. Une hausse de $\sigma$ augmente la valeur temporelle du warrant (Call et Put).
    - **L'Échéance** : À la date d'échéance, la valeur du warrant est gelée à sa valeur intrinsèque : $\max(0, S-K)$ pour un Call, $\max(0, K-S)$ pour un Put.
    
    ### 3. Modèle Turbo (Barrière Désactivante)
    - **Turbo Call** : $\text{Prix} = \frac{\max(0, S - \text{Strike})}{\text{Parité}}$. KO si $S \leq \text{Strike}$.
    - **Turbo Put** : $\text{Prix} = \frac{\max(0, \text{Strike} - S)}{\text{Parité}}$. KO si $S \geq \text{Strike}$.
    
    **Knock-Out (KO)** : Si la barrière est touchée (Strike), le produit est immédiatement désactivé et sa valeur devient **0 €**.
    
    ### 4. Visualisation Pédagogique (Exemple sur Ticker 1)
    """)
    
    # --- Pedagogical Charts Logic ---
    # Using Ticker 1 parameters as baseline
    bs_s = sp1 if sp1 else 100
    bs_k = p1_config['w_strike']
    bs_v = p1_config['w_vol'] / 100.0
    bs_type = p1_config['w_type']
    bs_ratio = p1_config['w_ratio']
    
    # 1. Theta Visualization (Time decay)
    days_range = np.linspace(365, 0, 100)
    theta_prices = [black_scholes(bs_s, bs_k, d/365.0, bs_v, type=bs_type) / bs_ratio for d in days_range]
    
    fig_theta = go.Figure()
    fig_theta.add_trace(go.Scatter(x=days_range, y=theta_prices, name="Prix du Warrant", line=dict(color="#f1c40f", width=3)))
    fig_theta.update_layout(title="Impact du Temps (Theta)", xaxis_title="Jours restants", yaxis_title="Prix (€)", 
                            height=350, template="plotly_dark", xaxis_autorange="reversed")
    
    # 2. Vega Visualization (Volatility impact)
    vol_range = np.linspace(0.05, 1.0, 100)
    # Using a fixed T = 0.5 year for vega chart
    vega_prices = [black_scholes(bs_s, bs_k, 0.5, v, type=bs_type) / bs_ratio for v in vol_range]
    
    fig_vega = go.Figure()
    fig_vega.add_trace(go.Scatter(x=vol_range * 100, y=vega_prices, name="Prix du Warrant", line=dict(color="#9b59b6", width=3)))
    fig_vega.update_layout(title="Impact de la Volatilité (Vega)", xaxis_title="Volatilité Implicite (%)", yaxis_title="Prix (€)", 
                           height=350, template="plotly_dark")
    
    col_ped1, col_ped2 = st.columns(2)
    with col_ped1:
        st.markdown("**Theta : Plus le temps passe, plus le warrant perd de sa valeur.**")
        st.plotly_chart(fig_theta, use_container_width=True, key="ped_theta")
    with col_ped2:
        st.markdown("**Vega : Plus le marché est volatil, plus le warrant est cher.**")
        st.plotly_chart(fig_vega, use_container_width=True, key="ped_vega")

    # 3. Convergence Visualization (Warrant vs Turbo)
    st.markdown("---")
    st.markdown("**Convergence : Pourquoi le Warrant et le Turbo se ressemblent à faible volatilité ?**")
    
    # Calculate Turbo price (intrinsic)
    t_price = (bs_s - bs_k) / bs_ratio if bs_type == "Call" else (bs_k - bs_s) / bs_ratio
    t_price = max(0, t_price)
    
    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(x=vol_range * 100, y=vega_prices, name="Prix Warrant", line=dict(color="#f1c40f", width=3)))
    fig_conv.add_hline(y=t_price, line_dash="dash", line_color="#e74c3c", annotation_text="Prix Turbo (Valeur Intrinsèque)")
    
    fig_conv.update_layout(title="Convergence Warrant vs Turbo", xaxis_title="Volatilité Implicite (%)", yaxis_title="Prix (€)", 
                           height=400, template="plotly_dark")
    
    st.plotly_chart(fig_conv, use_container_width=True, key="ped_conv")
    st.info("""
    **Observation** : À une volatilité très faible (proche de 0%), le prix du Warrant tend vers le prix du Turbo (valeur intrinsèque). 
    C'est parce que la probabilité que le cours s'éloigne du strike devient minime, annulant la "valeur temps" (Theta) du warrant.
    """)
