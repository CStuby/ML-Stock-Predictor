import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.io as pio
pio.templates.default = "plotly_dark"
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0a0e1a;
    }
    .stMetric {
        background-color: #1a2332;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(0, 217, 255, 0.3);
    }
    .stMetric label {
        color: #00d9ff !important;
    }
    h1, h2, h3 {
        color: #00d9ff;
    }
    .css-1d391kg {
        background-color: #1a2332;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📈 Stock Price Predictor")
st.markdown("### AI-Powered Stock Market Forecasting with Machine Learning")
st.markdown("---")

# Model parameters
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 ML Model Settings")
changepoint_prior_scale = st.sidebar.slider(
    "Model flexibility (higher = more flexible)",
    0.001, 0.5, 0.05
)
seasonality_mode = st.sidebar.selectbox(
    "Seasonality mode",
    ["multiplicative", "additive"],
    key="seasonality_select"
)

# Stock selection
popular_stocks = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "Meta": "META",
    "NVIDIA": "NVDA",
    "Netflix": "NFLX",
    "AMD": "AMD",
    "Intel": "INTC"
}

selected_stock_name = st.sidebar.selectbox(
    "Select Stock",
    options=list(popular_stocks.keys())
)
selected_stock = popular_stocks[selected_stock_name]

# Custom stock input
custom_stock = st.sidebar.text_input(
    "Or enter custom ticker:",
    placeholder="e.g., AAPL, TSLA"
)
if custom_stock:
    selected_stock = custom_stock.upper()
    selected_stock_name = custom_stock.upper()

# Date range
years_back = st.sidebar.slider("Historical data (years)", 1, 10, 3, key="years_slider")
start_date = datetime.now() - timedelta(days=years_back*365)
end_date = datetime.now()

# Prediction period
prediction_days = st.sidebar.slider("Prediction period (days)", 7, 90, 30, key="days_slider")

st.sidebar.markdown("---")

# Start prediction button
predict_button = st.sidebar.button("🚀 Generate Prediction", type="primary")

# Main content
@st.cache_data
def load_data(ticker, start, end):
    """Load stock data from Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date']).dt.date
        data['Date'] = pd.to_datetime(data['Date'])  
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def prepare_data_for_prophet(df):
    """Prepare data for Prophet model"""
    prophet_df = df[['Date', 'Close']].copy()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

def train_prophet_model(df, changepoint_scale, season_mode):
    """Train Prophet model"""
    model = Prophet(
        changepoint_prior_scale=changepoint_scale,
        seasonality_mode=season_mode,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(df)
    return model

def calculate_metrics(actual, predicted):
    """Calculate prediction accuracy metrics"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape

# Display stock info
col1, col2, col3 = st.columns([1, 1, 1]) 

with col1:
    st.metric("Selected Stock", selected_stock_name)

# Load data
with st.spinner(f"Loading data for {selected_stock}..."):
    data = load_data(selected_stock, start_date, end_date)

if data is not None and not data.empty:
    # Display current price
    current_price = float(data['Close'].iloc[-1])
    prev_price = float(data['Close'].iloc[-2])
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    with col2:
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
            f"{price_change_pct:+.2f}%"
        )
    
    with col3:
        st.metric("Data Points", len(data))
    
    # Historical price chart
    st.markdown("### 📊 Historical Stock Price")
    
    fig_history = go.Figure()
    fig_history.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#00d9ff', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 217, 255, 0.1)'
    ))
    fig_history.update_layout(
        template='plotly_dark',
        height=500,
        hovermode='x unified',
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        showlegend=True,
        xaxis=dict(rangeslider=dict(visible=False)),
        yaxis=dict(autorange=True, fixedrange=False)
    )
    st.plotly_chart(fig_history, width='stretch', config={'displayModeBar': False})
    
    # Volume chart
    st.markdown("### 📊 Trading Volume")
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(
        x=data['Date'],
        y=data['Volume'],
        name='Volume',
        marker_color='#6366f1'
    ))
    fig_volume.update_layout(
        template='plotly_dark',
        height=300,
        xaxis_title="Date",
        yaxis_title="Volume",
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    st.plotly_chart(fig_volume, width='stretch', config={'displayModeBar': False})
    
    # Prediction section
    if predict_button:
        st.markdown("---")
        st.markdown("### 🤖 AI Prediction Model")
        
        with st.spinner("Training ML model..."):
            # Prepare data
            prophet_df = prepare_data_for_prophet(data)
            
            # Train model
            model = train_prophet_model(
                prophet_df,
                changepoint_prior_scale,
                seasonality_mode
            )
            
            # Make future predictions
            future = model.make_future_dataframe(periods=prediction_days)
            forecast = model.predict(future)
            
            # Calculate accuracy on historical data
            historical_predictions = forecast[forecast['ds'].isin(prophet_df['ds'])]
            mae, rmse, mape = calculate_metrics(
                prophet_df['y'].values,
                historical_predictions['yhat'].values
            )
        
        # Display metrics
        st.success("✅ Model trained successfully!")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("MAE (Mean Absolute Error)", f"${mae:.2f}")
        
        with metric_col2:
            st.metric("RMSE (Root Mean Squared Error)", f"${rmse:.2f}")
        
        with metric_col3:
            st.metric("MAPE (Mean Absolute % Error)", f"{mape:.2f}%")
        
        # Prediction chart
        st.markdown("### 📈 Price Forecast")
        
        fig_forecast = plot_plotly(model, forecast)
        fig_forecast.update_layout(
            template='plotly_dark',
            height=500,
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig_forecast, width='stretch', config={'displayModeBar': False})
        
        # Future predictions table
        st.markdown("### 📅 Predicted Prices")
        
        future_predictions = forecast[forecast['ds'] > data['Date'].max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        future_predictions.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
        future_predictions['Date'] = pd.to_datetime(future_predictions['Date']).dt.date
        
        # Format numbers
        for col in ['Predicted Price', 'Lower Bound', 'Upper Bound']:
            future_predictions[col] = future_predictions[col].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(
            future_predictions.head(30),
            width=True,
            hide_index=True
        )
        
        # Download predictions
        csv = future_predictions.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions (CSV)",
            data=csv,
            file_name=f"{selected_stock}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Model components
        st.markdown("### 🔍 Model Components")
        
        fig_components = model.plot_components(forecast)
        st.pyplot(fig_components)
        
        # Insights
        st.markdown("### 💡 Key Insights")
        
        last_actual = float(data['Close'].iloc[-1])
        last_predicted = float(forecast['yhat'].iloc[-1])
        future_predicted = float(forecast['yhat'].iloc[-1 - prediction_days])
        
        predicted_change = future_predicted - last_actual
        predicted_change_pct = (predicted_change / last_actual) * 100
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.info(f"""
            **Current Price**: ${last_actual:.2f}
            
            **Predicted Price ({prediction_days} days)**: ${future_predicted:.2f}
            
            **Expected Change**: {predicted_change_pct:+.2f}% ({'+' if predicted_change > 0 else ''}{predicted_change:.2f})
            """)
        
        with insight_col2:
            trend = "📈 **Upward Trend**" if predicted_change > 0 else "📉 **Downward Trend**"
            confidence = "High" if mape < 5 else "Medium" if mape < 10 else "Low"
            
            st.info(f"""
            **Trend Direction**: {trend}
            
            **Model Confidence**: {confidence} (MAPE: {mape:.2f}%)
            
            **Recommendation**: {"Consider buying" if predicted_change > 0 else "Exercise caution"}
            """)
        
        st.warning("⚠️ **Disclaimer**: This is a prediction model for educational purposes. Not financial advice. Always do your own research before investing.")

else:
    st.error("❌ Could not load data. Please check the stock ticker and try again.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #6b7280;'>
        <p>Built with Prophet ML | Data from Yahoo Finance | Created by Sami Sivén</p>
    </div>
""", unsafe_allow_html=True)
