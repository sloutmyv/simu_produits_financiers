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
            # If yfinance returns a MultiIndex column DataFrame
            if isinstance(data.columns, pd.MultiIndex):
                close_prices = data['Close']
            else:
                close_prices = data[['Close']]
                
            # Create Plotly figure with secondary Y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Add Traces
            fig.add_trace(
                go.Scatter(x=close_prices.index, y=close_prices[ticker1_input], name=ticker1_input, mode='lines'),
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(x=close_prices.index, y=close_prices[ticker2_input], name=ticker2_input, mode='lines'),
                secondary_y=True,
            )

            # Set x-axis title
            fig.update_xaxes(title_text="Date")

            # Set y-axes titles
            fig.update_yaxes(title_text=f"Prix {ticker1_input}", secondary_y=False)
            fig.update_yaxes(title_text=f"Prix {ticker2_input}", secondary_y=True)

            # Display plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display some metrics
            col1, col2 = st.columns(2)
            
            last_price1 = close_prices[ticker1_input].dropna().iloc[-1]
            prev_price1 = close_prices[ticker1_input].dropna().iloc[0]
            perf1 = (last_price1 / prev_price1 - 1) * 100
            
            last_price2 = close_prices[ticker2_input].dropna().iloc[-1]
            prev_price2 = close_prices[ticker2_input].dropna().iloc[0]
            perf2 = (last_price2 / prev_price2 - 1) * 100
            
            col1.metric(label=ticker1_input, value=f"{last_price1:,.2f}", delta=f"{perf1:.2f}%")
            col2.metric(label=ticker2_input, value=f"{last_price2:,.2f}", delta=f"{perf2:.2f}%")

    except Exception as e:
        st.error(f"Erreur lors de la récupération des données : {e}")
