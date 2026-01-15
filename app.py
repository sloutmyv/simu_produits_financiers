import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(page_title="Mini simulateur de produits financier", layout="wide")

st.title("📊 Mini simulateur de produits financier")

# Sidebar for parameters
st.sidebar.header("Paramètres")

ticker1_input = st.sidebar.text_input("Ticker 1", value="MC.PA")
ticker2_input = st.sidebar.text_input("Ticker 2", value="^FCHI")

period_options = {
    "1 mois": "1mo",
    "3 mois": "3mo",
    "6 mois": "6mo",
    "1 an": "1y",
    "2 ans": "2y",
    "5 ans": "5y",
    "YTD": "ytd",
    "Max": "max"
}

period_label = st.sidebar.selectbox("Période", options=list(period_options.keys()), index=3)
period_code = period_options[period_label]

st.sidebar.divider()
st.sidebar.header("Simulateur d'Investissement")

investment_sum = st.sidebar.number_input("Somme à investir (€)", value=1000, min_value=1, step=100)
purchase_date = st.sidebar.date_input("Date d'achat", value=datetime.now() - timedelta(days=365))
split_ratio = st.sidebar.slider(f"Répartition : % {ticker1_input}", 0, 100, 50)

# Data fetching function
@st.cache_data
def get_data(tickers, period):
    data = yf.download(tickers, period=period, interval="1d")
    return data

# Loading screen
with st.spinner('Chargement des données...'):
    try:
        data = get_data([ticker1_input, ticker2_input], period_code)
        
        if data.empty:
            st.error("Aucune donnée trouvée pour ces tickers.")
        else:            
            # Accessing Close prices
            if isinstance(data.columns, pd.MultiIndex):
                close_prices = data['Close']
            else:
                close_prices = data[['Close']]
            
            # Ensure index is datetime
            close_prices.index = pd.to_datetime(close_prices.index)
            
            # --- Performance Calculations ---
            # Ticker 1 Perf
            ticker1_start = close_prices[ticker1_input].dropna().iloc[0]
            perf_evol1 = (close_prices[ticker1_input] / ticker1_start - 1) * 100
            
            # Ticker 2 Perf
            ticker2_start = close_prices[ticker2_input].dropna().iloc[0]
            perf_evol2 = (close_prices[ticker2_input] / ticker2_start - 1) * 100
            
            # --- Portfolio Calculation ---
            # Sum allocated to each
            sum1 = investment_sum * (split_ratio / 100)
            sum2 = investment_sum * (1 - split_ratio / 100)
            
            # Find closest date in index to purchase_date
            mask = close_prices.index >= pd.to_datetime(purchase_date)
            sim_data = close_prices.loc[mask]
            
            if sim_data.empty:
                st.warning("La date d'achat est postérieure aux données disponibles ou aucun trade n'a eu lieu depuis.")
                portfolio_evol = None
            else:
                # Price at purchase (first available)
                p1_start = sim_data[ticker1_input].iloc[0]
                p2_start = sim_data[ticker2_input].iloc[0]
                
                # Number of shares
                shares1 = sum1 / p1_start
                shares2 = sum2 / p2_start
                
                # Portfolio value over time
                portfolio_evol = (sim_data[ticker1_input] * shares1) + (sim_data[ticker2_input] * shares2)
                portfolio_evol.name = "Valeur Portefeuille"
                
                # Percentage variation evolution
                perf_evol_port = (portfolio_evol / investment_sum - 1) * 100

            # --- Visualisation ---
            # Create Plotly figure with 3 subplots vertically if portfolio exists
            rows = 3 if portfolio_evol is not None else 2
            titles = [f"Prix {ticker1_input} (€ & %)", f"Prix {ticker2_input} (€ & %)"]
            
            specs = [[{"secondary_y": True}], [{"secondary_y": True}]]
            if portfolio_evol is not None:
                titles.append("Évolution du Portefeuille (€ & %)")
                specs.append([{"secondary_y": True}])
                
            fig = make_subplots(
                rows=rows, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.05,
                subplot_titles=titles,
                specs=specs
            )

            # Helper function to add gated traces
            def add_perf_traces(fig, row, perf_series, name_suffix):
                # Positive part
                fig.add_trace(
                    go.Scatter(
                        x=perf_series.index, 
                        y=perf_series.map(lambda x: x if x > 0 else 0), 
                        name=f"Gain {name_suffix} (%)", 
                        mode='lines', 
                        line=dict(width=0), 
                        fill='tozeroy', 
                        fillcolor='rgba(0, 255, 0, 0.2)'
                    ),
                    row=row, col=1, secondary_y=True
                )
                # Negative part
                fig.add_trace(
                    go.Scatter(
                        x=perf_series.index, 
                        y=perf_series.map(lambda x: x if x < 0 else 0), 
                        name=f"Perte {name_suffix} (%)", 
                        mode='lines', 
                        line=dict(width=0), 
                        fill='tozeroy', 
                        fillcolor='rgba(255, 0, 0, 0.2)'
                    ),
                    row=row, col=1, secondary_y=True
                )
                # Main Line for %
                fig.add_trace(
                    go.Scatter(
                        x=perf_series.index, 
                        y=perf_series, 
                        name=f"Variation {name_suffix} (%)", 
                        mode='lines', 
                        line=dict(color='white', width=1.5, dash='dot')
                    ),
                    row=row, col=1, secondary_y=True
                )

            # --- Subplot 1: Ticker 1 ---
            fig.add_trace(
                go.Scatter(x=close_prices.index, y=close_prices[ticker1_input], name=f"Prix {ticker1_input}", mode='lines', line=dict(color='#1f77b4')),
                row=1, col=1, secondary_y=False
            )
            add_perf_traces(fig, 1, perf_evol1, ticker1_input)

            # --- Subplot 2: Ticker 2 ---
            fig.add_trace(
                go.Scatter(x=close_prices.index, y=close_prices[ticker2_input], name=f"Prix {ticker2_input}", mode='lines', line=dict(color='#ff7f0e')),
                row=2, col=1, secondary_y=False
            )
            add_perf_traces(fig, 2, perf_evol2, ticker2_input)
            
            # --- Subplot 3: Portfolio ---
            if portfolio_evol is not None:
                fig.add_trace(
                    go.Scatter(x=portfolio_evol.index, y=portfolio_evol, name="Valeur Portefeuille (€)", mode='lines', line=dict(color='lightgrey', width=1)),
                    row=3, col=1, secondary_y=False
                )
                add_perf_traces(fig, 3, perf_evol_port, "Portfolio")

            # Add figure title
            fig.update_layout(
                height=1000 if rows == 3 else 700,
                title_text=f"Analyse et Simulation ({period_label})",
                hovermode="x unified",
                template="plotly_dark",
                showlegend=False
            )

            # Set x-axis title
            fig.update_xaxes(title_text="Date", row=rows, col=1)

            # Set y-axes titles
            fig.update_yaxes(title_text="Prix (€)", row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="Var. (%)", row=1, col=1, secondary_y=True, zeroline=True, zerolinecolor='white')
            
            fig.update_yaxes(title_text="Prix (€)", row=2, col=1, secondary_y=False)
            fig.update_yaxes(title_text="Var. (%)", row=2, col=1, secondary_y=True, zeroline=True, zerolinecolor='white')
            
            if rows == 3:
                fig.update_yaxes(title_text="Valeur (€)", row=3, col=1, secondary_y=False)
                fig.update_yaxes(title_text="Var. (%)", row=3, col=1, secondary_y=True, zeroline=True, zerolinecolor='white')

            # Display plot
            st.plotly_chart(fig, use_container_width=True, key="financial_simulation_charts")
            
            # --- Metrics ---
            st.divider()
            
            # Asset Metrics
            col1, col2 = st.columns(2)
            
            last_p1 = close_prices[ticker1_input].dropna().iloc[-1]
            first_p1 = close_prices[ticker1_input].dropna().iloc[0]
            perf1 = (last_p1 / first_p1 - 1) * 100
            
            last_p2 = close_prices[ticker2_input].dropna().iloc[-1]
            first_p2 = close_prices[ticker2_input].dropna().iloc[0]
            perf2 = (last_p2 / first_p2 - 1) * 100
            
            col1.metric(label=f"Cours {ticker1_input}", value=f"{last_p1:,.2f}", delta=f"{perf1:.2f}%")
            col2.metric(label=f"Cours {ticker2_input}", value=f"{last_p2:,.2f}", delta=f"{perf2:.2f}%")
            
            # Portfolio Metrics
            if portfolio_evol is not None:
                st.subheader("Bilan du Portefeuille")
                m1, m2, m3 = st.columns(3)
                
                current_val = portfolio_evol.iloc[-1]
                total_perf = (current_val / investment_sum - 1) * 100
                total_gain = current_val - investment_sum
                
                m1.metric("Valeur Actuelle", f"{current_val:,.2f} €")
                m2.metric("Plus-value / Moins-value", f"{total_gain:,.2f} €", delta=f"{total_perf:.2f}%")
                m3.metric("Investissement Initial", f"{investment_sum:,.2f} €")

    except Exception as e:
        st.error(f"Erreur lors de la récupération des données : {e}")
        st.exception(e)
