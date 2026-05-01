import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader
from risk_engine import RiskEngine
from portfolio import Portfolio
from ml_model import RiskPredictor
from utils import (
    format_currency, format_percentage, detect_high_correlation,
    generate_alert, check_var_threshold, check_liquidity_risk,
    check_ml_risk_signal, get_tadawul_top_stocks
)

st.set_page_config(
    page_title="Risk Management System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_market_data(tickers, period):
    loader = DataLoader(data_dir='../data')
    prices = loader.get_close_prices(tickers, period=period)
    returns = loader.get_returns(tickers, period=period)
    volume = loader.get_volume_data(tickers, period=period)
    return prices, returns, volume


def main():
    st.title("🎯 Risk Management System (RMS)")
    st.markdown("### Professional Portfolio Risk Analysis & Prediction")
    
    st.sidebar.header("⚙️ Configuration")
    
    market_type = st.sidebar.selectbox(
        "Select Market",
        ["Saudi Stock Market (Tadawul)", "Global Stocks", "Custom"]
    )
    
    if market_type == "Saudi Stock Market (Tadawul)":
        default_tickers = get_tadawul_top_stocks()[:5]
    elif market_type == "Global Stocks":
        default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    else:
        default_tickers = []
    
    tickers_input = st.sidebar.text_area(
        "Enter Stock Tickers (one per line)",
        value="\n".join(default_tickers),
        height=150,
        help="Edit this list to add/remove stocks. For Saudi stocks, use format: 2222.SR"
    )
    tickers = [t.strip() for t in tickers_input.split('\n') if t.strip()]
    
    st.sidebar.info("💡 **Tip**: You can edit the tickers above directly!")
    
    period = st.sidebar.selectbox(
        "Time Period",
        ['1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=3
    )
    
    portfolio_value = st.sidebar.number_input(
        "Portfolio Value ($)",
        min_value=10000,
        max_value=100000000,
        value=1000000,
        step=10000
    )
    
    confidence_levels = st.sidebar.multiselect(
        "VaR Confidence Levels",
        [0.90, 0.95, 0.99],
        default=[0.95, 0.99]
    )
    
    if not tickers:
        st.warning("⚠️ Please enter at least one stock ticker.")
        return
    
    with st.spinner("Loading market data..."):
        try:
            prices, returns, volume = load_market_data(tickers, period)
            
            if prices.empty or returns.empty:
                st.error("❌ No data available for the selected tickers.")
                return
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return
    
    weights_input = st.sidebar.text_input(
        "Portfolio Weights (comma-separated)",
        value=", ".join([f"{1/len(tickers):.2f}"] * len(tickers))
    )
    
    try:
        weights = np.array([float(w.strip()) for w in weights_input.split(',')])
        if len(weights) != len(tickers):
            st.sidebar.error("⚠️ Number of weights must match number of tickers.")
            weights = np.array([1.0 / len(tickers)] * len(tickers))
        else:
            weights = weights / weights.sum()
    except:
        st.sidebar.error("⚠️ Invalid weights format. Using equal weights.")
        weights = np.array([1.0 / len(tickers)] * len(tickers))
    
    portfolio = Portfolio(
        name="My Portfolio",
        initial_value=portfolio_value,
        tickers=tickers,
        weights=weights
    )
    
    latest_prices = prices.iloc[-1]
    portfolio.update_prices(latest_prices)
    
    risk_engine = RiskEngine(returns, confidence_levels=confidence_levels)
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "📈 Risk Metrics", 
        "🔮 Monte Carlo", 
        "🤖 ML Prediction",
        "🔔 Alerts",
        "📋 Reports"
    ])
    
    with tab1:
        st.header("Portfolio Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        perf_metrics = portfolio.get_performance_metrics(returns)
        
        with col1:
            st.metric(
                "Portfolio Value",
                format_currency(portfolio.current_value),
                delta=format_currency(perf_metrics['profit_loss'])
            )
        
        with col2:
            st.metric(
                "Total Return",
                format_percentage(perf_metrics['total_return']),
                delta=format_percentage(perf_metrics['annualized_return'])
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{perf_metrics['sharpe_ratio']:.4f}"
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                format_percentage(perf_metrics['max_drawdown'])
            )
        
        st.subheader("Portfolio Allocation")
        summary_df = portfolio.get_portfolio_summary()
        st.dataframe(summary_df, use_container_width=True)
        
        fig_allocation = go.Figure(data=[go.Pie(
            labels=tickers,
            values=weights,
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        fig_allocation.update_layout(
            title="Portfolio Weight Distribution",
            height=400
        )
        st.plotly_chart(fig_allocation, use_container_width=True)
        
        st.subheader("Cumulative Returns")
        cumulative_returns = (1 + returns).cumprod()
        portfolio_cumulative = (1 + (returns * weights).sum(axis=1)).cumprod()
        
        fig_returns = go.Figure()
        for ticker in tickers:
            fig_returns.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[ticker],
                name=ticker,
                mode='lines'
            ))
        fig_returns.add_trace(go.Scatter(
            x=portfolio_cumulative.index,
            y=portfolio_cumulative,
            name='Portfolio',
            mode='lines',
            line=dict(width=3, color='black', dash='dash')
        ))
        fig_returns.update_layout(
            title="Cumulative Returns Comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig_returns, use_container_width=True)
    
    with tab2:
        st.header("Risk Metrics")
        
        correlation_matrix = risk_engine.calculate_correlation_matrix()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Correlation Matrix")
            fig_corr = px.imshow(
                correlation_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                aspect='auto',
                zmin=-1,
                zmax=1
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            high_corr = detect_high_correlation(correlation_matrix, threshold=0.7)
            if high_corr:
                st.warning(f"⚠️ {len(high_corr)} high correlation pair(s) detected")
                for ticker1, ticker2, corr in high_corr[:3]:
                    st.write(f"• {ticker1} ↔ {ticker2}: {corr:.3f}")
        
        with col2:
            st.subheader("EWMA Volatility")
            ewma_vol = risk_engine.calculate_ewma_volatility(span=30)
            
            fig_vol = go.Figure()
            for ticker in tickers:
                fig_vol.add_trace(go.Scatter(
                    x=ewma_vol.index,
                    y=ewma_vol[ticker],
                    name=ticker,
                    mode='lines'
                ))
            fig_vol.update_layout(
                xaxis_title="Date",
                yaxis_title="Annualized Volatility",
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        
        st.subheader("Value at Risk (VaR) Analysis")
        
        var_param = risk_engine.calculate_parametric_var(portfolio_value, weights)
        var_hist = risk_engine.calculate_historical_var(portfolio_value, weights)
        cvar = risk_engine.calculate_cvar(portfolio_value, weights)
        
        var_data = []
        for conf in confidence_levels:
            var_data.append({
                'Confidence Level': f"{conf*100}%",
                'Parametric VaR': format_currency(var_param[conf]),
                'Historical VaR': format_currency(var_hist[conf]),
                'CVaR': format_currency(cvar[conf])
            })
        
        var_df = pd.DataFrame(var_data)
        st.dataframe(var_df, use_container_width=True)
        
        fig_var = go.Figure()
        conf_labels = [f"{c*100}%" for c in confidence_levels]
        fig_var.add_trace(go.Bar(
            x=conf_labels,
            y=[var_param[c] for c in confidence_levels],
            name='Parametric VaR',
            marker_color='lightblue'
        ))
        fig_var.add_trace(go.Bar(
            x=conf_labels,
            y=[var_hist[c] for c in confidence_levels],
            name='Historical VaR',
            marker_color='lightcoral'
        ))
        fig_var.add_trace(go.Bar(
            x=conf_labels,
            y=[cvar[c] for c in confidence_levels],
            name='CVaR',
            marker_color='lightgreen'
        ))
        fig_var.update_layout(
            title="VaR Comparison by Method",
            xaxis_title="Confidence Level",
            yaxis_title="Value at Risk ($)",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig_var, use_container_width=True)
        
        st.subheader("Stress Testing")
        stress_results = risk_engine.stress_test(portfolio_value, weights)
        
        stress_data = []
        for scenario, result in stress_results.items():
            stress_data.append({
                'Scenario': scenario.replace('_', ' ').title(),
                'Portfolio Value': format_currency(result['shocked_value']),
                'Loss': format_currency(result['loss']),
                'Loss %': format_percentage(result['loss_percentage'] / 100)
            })
        
        stress_df = pd.DataFrame(stress_data)
        st.dataframe(stress_df, use_container_width=True)
        
        max_dd = risk_engine.calculate_maximum_drawdown(prices)
        st.subheader("Maximum Drawdown")
        st.metric("Portfolio Max Drawdown", format_percentage(max_dd['portfolio']))
    
    with tab3:
        st.header("Monte Carlo Simulation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_simulations = st.number_input(
                "Number of Simulations",
                min_value=1000,
                max_value=50000,
                value=10000,
                step=1000
            )
        
        with col2:
            time_horizon = st.number_input(
                "Time Horizon (days)",
                min_value=30,
                max_value=1260,
                value=252,
                step=30
            )
        
        if st.button("Run Monte Carlo Simulation"):
            with st.spinner("Running simulation..."):
                mc_results = risk_engine.monte_carlo_simulation(
                    portfolio_value=portfolio_value,
                    weights=weights,
                    num_simulations=num_simulations,
                    time_horizon=time_horizon
                )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Mean Final Value",
                    format_currency(mc_results['mean_final_value'])
                )
            
            with col2:
                st.metric(
                    "Std Dev",
                    format_currency(mc_results['std_final_value'])
                )
            
            with col3:
                st.metric(
                    "5th Percentile",
                    format_currency(mc_results['percentiles']['5th'])
                )
            
            fig_mc = go.Figure()
            
            sample_paths = min(100, num_simulations)
            for i in range(sample_paths):
                fig_mc.add_trace(go.Scatter(
                    y=mc_results['simulated_paths'][i] * portfolio_value,
                    mode='lines',
                    line=dict(width=0.5),
                    opacity=0.3,
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            mean_path = mc_results['simulated_paths'].mean(axis=0) * portfolio_value
            fig_mc.add_trace(go.Scatter(
                y=mean_path,
                mode='lines',
                name='Mean Path',
                line=dict(color='red', width=3)
            ))
            
            fig_mc.update_layout(
                title=f"Monte Carlo Simulation - {num_simulations} paths",
                xaxis_title="Days",
                yaxis_title="Portfolio Value ($)",
                height=600,
                showlegend=True
            )
            st.plotly_chart(fig_mc, use_container_width=True)
            
            st.subheader("Distribution of Final Portfolio Values")
            fig_dist = go.Figure(data=[go.Histogram(
                x=mc_results['final_values'],
                nbinsx=50,
                marker_color='lightblue'
            )])
            fig_dist.add_vline(
                x=portfolio_value,
                line_dash="dash",
                line_color="red",
                annotation_text="Initial Value"
            )
            fig_dist.update_layout(
                xaxis_title="Final Portfolio Value ($)",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            st.subheader("Monte Carlo VaR")
            mc_var_data = []
            for conf in confidence_levels:
                mc_var_data.append({
                    'Confidence Level': f"{conf*100}%",
                    'MC VaR': format_currency(mc_results['mc_var'][conf])
                })
            st.dataframe(pd.DataFrame(mc_var_data), use_container_width=True)
    
    with tab4:
        st.header("Machine Learning Risk Prediction")
        
        st.info("🤖 Train a model to predict future portfolio risk levels")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ml_model_type = st.selectbox(
                "Model Type",
                ['Random Forest', 'XGBoost']
            )
        
        with col2:
            test_size = st.slider(
                "Test Size (%)",
                min_value=10,
                max_value=40,
                value=20
            ) / 100
        
        if st.button("Train ML Model"):
            with st.spinner("Training model..."):
                try:
                    predictor = RiskPredictor(model_type='classification')
                    
                    features = predictor.prepare_features(returns, prices, volume)
                    target = predictor.create_target_classification(returns, weights)
                    
                    use_xgboost = (ml_model_type == 'XGBoost')
                    
                    train_results = predictor.train_model(
                        features,
                        target,
                        test_size=test_size,
                        use_xgboost=use_xgboost
                    )
                    
                    st.session_state['ml_predictor'] = predictor
                    st.session_state['ml_results'] = train_results
                    
                    st.success("✅ Model trained successfully!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Training Accuracy",
                            f"{train_results['train_accuracy']:.4f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Test Accuracy",
                            f"{train_results['test_accuracy']:.4f}"
                        )
                    
                    st.subheader("Feature Importance")
                    feature_imp = train_results['feature_importance'].head(15)
                    
                    fig_imp = go.Figure(go.Bar(
                        x=feature_imp['importance'],
                        y=feature_imp['feature'],
                        orientation='h',
                        marker_color='lightblue'
                    ))
                    fig_imp.update_layout(
                        xaxis_title="Importance",
                        yaxis_title="Feature",
                        height=500,
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                    
                    st.subheader("Classification Report")
                    class_report = train_results['classification_report']
                    
                    report_data = []
                    for label in ['0', '1', '2']:
                        if label in class_report:
                            report_data.append({
                                'Risk Level': ['LOW', 'MEDIUM', 'HIGH'][int(label)],
                                'Precision': f"{class_report[label]['precision']:.3f}",
                                'Recall': f"{class_report[label]['recall']:.3f}",
                                'F1-Score': f"{class_report[label]['f1-score']:.3f}"
                            })
                    
                    st.dataframe(pd.DataFrame(report_data), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")
        
        if 'ml_predictor' in st.session_state:
            st.subheader("Current Risk Prediction")
            
            try:
                predictor = st.session_state['ml_predictor']
                
                features = predictor.prepare_features(returns, prices, volume)
                
                latest_features = features.iloc[-1:]
                
                risk_levels = predictor.predict_risk_level(latest_features)
                current_risk = risk_levels[0]
                
                risk_colors = {
                    'LOW': '🟢',
                    'MEDIUM': '🟡',
                    'HIGH': '🔴'
                }
                
                st.markdown(f"### {risk_colors.get(current_risk, '⚪')} Current Risk Level: **{current_risk}**")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
    
    with tab5:
        st.header("Risk Alerts")
        
        alerts = []
        
        var_threshold = portfolio_value * 0.05
        for conf in confidence_levels:
            alert = check_var_threshold(var_hist[conf], var_threshold, conf)
            if alert:
                alerts.append(alert)
        
        high_corr = detect_high_correlation(correlation_matrix, threshold=0.8)
        if high_corr:
            alert = generate_alert(
                alert_type='HIGH_CORRELATION',
                message=f"{len(high_corr)} stock pair(s) with correlation > 0.8 detected",
                severity='MEDIUM'
            )
            alerts.append(alert)
        
        if 'ml_predictor' in st.session_state:
            try:
                features = st.session_state['ml_predictor'].prepare_features(returns, prices, volume)
                latest_features = features.iloc[-1:]
                risk_levels = st.session_state['ml_predictor'].predict_risk_level(latest_features)
                alert = check_ml_risk_signal(risk_levels[0])
                if alert:
                    alerts.append(alert)
            except:
                pass
        
        if alerts:
            st.warning(f"⚠️ {len(alerts)} active alert(s)")
            
            for alert in alerts:
                severity_icons = {
                    'HIGH': '🔴',
                    'MEDIUM': '🟡',
                    'LOW': '🟢',
                    'INFO': 'ℹ️'
                }
                icon = severity_icons.get(alert['severity'], 'ℹ️')
                
                with st.expander(f"{icon} {alert['type']} - {alert['severity']}"):
                    st.write(f"**Message:** {alert['message']}")
                    st.write(f"**Time:** {alert['timestamp']}")
        else:
            st.success("✅ No critical alerts at this time")
        
        st.subheader("Alert Configuration")
        
        var_threshold_pct = st.slider(
            "VaR Alert Threshold (% of portfolio)",
            min_value=1,
            max_value=20,
            value=5
        )
        
        correlation_threshold = st.slider(
            "High Correlation Threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.05
        )
    
    with tab6:
        st.header("Risk Reports")
        
        st.subheader("Risk Summary")
        risk_summary = risk_engine.get_risk_summary(portfolio_value, weights)
        st.dataframe(risk_summary, use_container_width=True)
        
        st.subheader("Portfolio Performance Report")
        perf_report_data = {
            'Metric': [
                'Initial Value',
                'Current Value',
                'Total Return',
                'Annualized Return',
                'Annualized Volatility',
                'Sharpe Ratio',
                'Max Drawdown',
                'Win Rate'
            ],
            'Value': [
                format_currency(portfolio.initial_value),
                format_currency(portfolio.current_value),
                format_percentage(perf_metrics['total_return']),
                format_percentage(perf_metrics['annualized_return']),
                format_percentage(perf_metrics['annualized_volatility']),
                f"{perf_metrics['sharpe_ratio']:.4f}",
                format_percentage(perf_metrics['max_drawdown']),
                format_percentage(perf_metrics['win_rate'])
            ]
        }
        st.dataframe(pd.DataFrame(perf_report_data), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Portfolio Data"):
                portfolio_data = portfolio.export_portfolio()
                st.download_button(
                    label="Download Portfolio JSON",
                    data=str(portfolio_data),
                    file_name="portfolio_data.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("Export Risk Metrics"):
                risk_data = risk_engine.generate_risk_report()
                st.download_button(
                    label="Download Risk Report JSON",
                    data=str(risk_data),
                    file_name="risk_report.json",
                    mime="application/json"
                )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **RMS v1.0**
        
        Professional Risk Management System
        for Equity Portfolio Analysis
        
        © 2026
        """
    )


if __name__ == "__main__":
    main()
