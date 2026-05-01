import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader
from risk_engine import RiskEngine
from portfolio import Portfolio
from ml_model import RiskPredictor
from position_manager import PositionManager
from validation_engine import SymbolUniverse, ValidationEngine
from alerts_engine import AlertsEngine, RecommendationEngine
from utils import format_currency, format_percentage, detect_high_correlation

st.set_page_config(
    page_title="Risk Management System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.alert-high   { background:#fff0f0; border-left:4px solid #e53935; padding:8px 12px; border-radius:4px; margin:4px 0; }
.alert-medium { background:#fff8e1; border-left:4px solid #fb8c00; padding:8px 12px; border-radius:4px; margin:4px 0; }
.alert-low    { background:#f1f8e9; border-left:4px solid #43a047; padding:8px 12px; border-radius:4px; margin:4px 0; }
.rec-high     { background:#fce4ec; border-left:4px solid #c62828; padding:8px 12px; border-radius:4px; margin:4px 0; }
.rec-medium   { background:#e3f2fd; border-left:4px solid #1565c0; padding:8px 12px; border-radius:4px; margin:4px 0; }
.rec-low      { background:#e8f5e9; border-left:4px solid #2e7d32; padding:8px 12px; border-radius:4px; margin:4px 0; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

@st.cache_data(ttl=3600)
def load_market_data(tickers_tuple, period):
    tickers = list(tickers_tuple)
    loader = DataLoader(data_dir=DATA_DIR)
    prices = loader.get_close_prices(tickers, period=period)
    returns = loader.get_returns(tickers, period=period)
    volume = loader.get_volume_data(tickers, period=period)
    return prices, returns, volume

@st.cache_data
def load_universe():
    return SymbolUniverse(data_dir=DATA_DIR)


def render_sidebar():
    st.sidebar.title("⚙️ Portfolio Setup")

    st.sidebar.markdown("### 1. Input Mode")
    input_mode = st.sidebar.radio(
        "Select input method:",
        ["Manual Selection", "Excel Upload", "Default (Demo)"],
        index=2
    )

    st.sidebar.markdown("### 2. Market & Period")
    market_type = st.sidebar.selectbox(
        "Market",
        ["Saudi (Tadawul)", "Global", "Mixed"]
    )
    period = st.sidebar.selectbox(
        "Analysis Period",
        ['1mo', '3mo', '6mo', '1y', '2y'],
        index=3
    )
    portfolio_value_override = st.sidebar.number_input(
        "Portfolio Value Override ($) — leave 0 to use position values",
        min_value=0,
        max_value=100_000_000,
        value=0,
        step=10_000
    )

    st.sidebar.markdown("### 3. Risk Settings")
    confidence_levels = st.sidebar.multiselect(
        "VaR Confidence Levels",
        [0.90, 0.95, 0.99],
        default=[0.95, 0.99]
    )
    var_threshold_pct = st.sidebar.slider("VaR Alert Threshold (% of portfolio)", 1, 20, 5) / 100
    corr_threshold = st.sidebar.slider("Correlation Alert Threshold", 0.50, 1.00, 0.80, 0.05)
    conc_threshold = st.sidebar.slider("Concentration Alert Threshold (%)", 20, 80, 40) / 100

    return {
        'input_mode': input_mode,
        'market_type': market_type,
        'period': period,
        'portfolio_value_override': portfolio_value_override,
        'confidence_levels': confidence_levels or [0.95],
        'var_threshold_pct': var_threshold_pct,
        'corr_threshold': corr_threshold,
        'conc_threshold': conc_threshold
    }


def build_positions(cfg: dict, universe: SymbolUniverse):
    mode = cfg['input_mode']
    market = cfg['market_type']
    pm = PositionManager()
    errors = []

    if mode == "Excel Upload":
        st.info("📂 Upload an Excel file with columns: **Symbol** | **Quantity**")
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded = st.file_uploader("Upload Portfolio Excel", type=['xlsx', 'xls'])
        with col2:
            st.download_button(
                "⬇️ Download Template",
                data=PositionManager.generate_excel_template(),
                file_name="portfolio_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        if uploaded is None:
            st.warning("Please upload your portfolio Excel file.")
            return None, None

        ok, msg, df = pm.from_excel(uploaded)
        if not ok:
            st.error(f"❌ {msg}")
            return None, None
        if msg != "OK":
            st.warning(f"⚠️ {msg}")

    elif mode == "Manual Selection":
        market_key = 'tasi' if 'Saudi' in market else ('global' if market == 'Global' else 'all')
        options = universe.get_display_options(market=market_key)

        if not options:
            st.warning("Symbol universe not loaded. Using free-text input.")
            raw = st.text_area("Enter symbols (one per line)", "2222.SR\n1120.SR\n2010.SR", height=120)
            symbols = [s.strip().upper() for s in raw.split('\n') if s.strip()]
        else:
            labels = list(options.keys())
            selected_labels = st.multiselect(
                "Select stocks from universe:",
                labels,
                default=labels[:5] if len(labels) >= 5 else labels
            )
            symbols = [options[l] for l in selected_labels]

        if not symbols:
            st.warning("Please select at least one stock.")
            return None, None

        st.markdown("#### Quantities (optional — leave 1 for equal weighting)")
        qty_data = []
        cols = st.columns(min(len(symbols), 5))
        for i, sym in enumerate(symbols):
            with cols[i % 5]:
                qty = st.number_input(sym, min_value=1, value=100, step=10, key=f"qty_{sym}")
                qty_data.append(qty)

        ok, msg, df = pm.from_manual(symbols, qty_data)
        if not ok:
            st.error(f"❌ {msg}")
            return None, None

    else:
        default_syms = ['2222.SR', '1120.SR', '2010.SR', '7010.SR', '1211.SR'] \
            if 'Saudi' in market else ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        ok, msg, df = pm.from_manual(default_syms)
        if not ok:
            st.error(msg)
            return None, None
        st.info(f"🔖 Demo mode — using default symbols: {', '.join(default_syms)}")

    with st.spinner("Fetching live prices..."):
        positions_df, failed = pm.fetch_prices()

    if failed:
        st.warning(f"⚠️ Could not fetch prices for: {', '.join(failed)}. They will be excluded.")
        pm.positions = pm.positions[~pm.positions['Symbol'].isin(failed)].reset_index(drop=True)
        if pm.positions.empty:
            st.error("No valid positions remain after price fetch.")
            return None, None
        total = pm.positions['Value'].sum()
        pm.positions['Weight'] = pm.positions['Value'] / total

    return pm, errors


def main():
    st.title("📊 Risk Management System (RMS)")
    st.markdown("##### Professional Portfolio Risk Analysis — Saudi & Global Markets")
    st.markdown("---")

    cfg = render_sidebar()
    universe = load_universe()

    st.header("🗂️ Portfolio Input")
    pm, _ = build_positions(cfg, universe)

    if pm is None or pm.positions.empty:
        return

    tickers = pm.get_tickers()
    weights = pm.get_weights()
    portfolio_value = cfg['portfolio_value_override'] if cfg['portfolio_value_override'] > 0 else pm.get_portfolio_value()
    if portfolio_value == 0:
        portfolio_value = 1_000_000

    val_engine = ValidationEngine(universe)
    weights, w_msg = val_engine.validate_weights(weights)
    if w_msg != "OK":
        st.info(f"ℹ️ {w_msg}")

    conc_alerts = val_engine.check_concentration(weights, tickers, cfg['conc_threshold'])
    if conc_alerts:
        for ca in conc_alerts:
            st.warning(f"⚠️ {ca['message']}")

    with st.expander("📋 Portfolio Positions", expanded=True):
        st.dataframe(pm.get_summary_df(), use_container_width=True)
        st.metric("Total Portfolio Value", format_currency(portfolio_value))

    with st.spinner("Loading historical market data..."):
        try:
            prices, returns, volume = load_market_data(tuple(tickers), cfg['period'])
            valid_tickers = [t for t in tickers if t in prices.columns]
            if not valid_tickers:
                st.error("No historical data available for the selected symbols.")
                return
            if len(valid_tickers) < len(tickers):
                missing = [t for t in tickers if t not in valid_tickers]
                st.warning(f"⚠️ No historical data for: {', '.join(missing)}")
                prices = prices[valid_tickers]
                returns = returns[valid_tickers]
                tickers = valid_tickers
                idx_map = {t: i for i, t in enumerate(pm.positions['Symbol'].tolist())}
                keep_idx = [idx_map[t] for t in tickers if t in idx_map]
                weights = weights[keep_idx]
                weights = weights / weights.sum()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return

    risk_engine = RiskEngine(returns, confidence_levels=cfg['confidence_levels'])

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview", "📈 Risk Metrics", "🔮 Monte Carlo",
        "🤖 ML Prediction", "🔔 Alerts", "💡 Recommendations", "📋 Reports"
    ])

    corr_matrix = risk_engine.calculate_correlation_matrix()
    returns_stats = risk_engine.calculate_returns_stats()
    ann_vol = float((returns * weights).sum(axis=1).std() * np.sqrt(252))
    sharpe = risk_engine.calculate_sharpe_ratio(weights)
    sortino = risk_engine.calculate_sortino_ratio(weights)
    max_dd_result = risk_engine.calculate_maximum_drawdown(prices)
    max_dd = float(max_dd_result['portfolio'])
    var_param = risk_engine.calculate_parametric_var(portfolio_value, weights)
    var_hist = risk_engine.calculate_historical_var(portfolio_value, weights)
    cvar = risk_engine.calculate_cvar(portfolio_value, weights)
    primary_conf = cfg['confidence_levels'][0]

    alerts_engine = AlertsEngine(
        var_threshold_pct=cfg['var_threshold_pct'],
        correlation_threshold=cfg['corr_threshold'],
        concentration_threshold=cfg['conc_threshold']
    )
    alerts = alerts_engine.run_all_checks(
        var_value=var_hist[primary_conf],
        portfolio_value=portfolio_value,
        corr_matrix=corr_matrix,
        weights=weights,
        symbols=tickers,
        ann_volatility=ann_vol,
        max_drawdown=max_dd
    )
    sev_counts = alerts_engine.get_severity_count()

    rec_engine = RecommendationEngine()
    recommendations = rec_engine.analyze(
        weights=weights,
        symbols=tickers,
        corr_matrix=corr_matrix,
        ann_volatility=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        var_pct=var_hist[primary_conf] / portfolio_value
    )

    with tab1:
        st.header("Portfolio Overview")

        c1, c2, c3, c4, c5 = st.columns(5)
        perf = (returns * weights).sum(axis=1)
        total_ret = float((1 + perf).prod() - 1)

        alert_badge = ""
        if sev_counts['HIGH'] > 0:
            alert_badge = f" 🔴 {sev_counts['HIGH']} High"
        if sev_counts['MEDIUM'] > 0:
            alert_badge += f" 🟡 {sev_counts['MEDIUM']} Medium"

        c1.metric("Portfolio Value", format_currency(portfolio_value))
        c2.metric("Total Return", format_percentage(total_ret))
        c3.metric("Sharpe Ratio", f"{sharpe:.3f}")
        c4.metric("Ann. Volatility", format_percentage(ann_vol))
        c5.metric("Active Alerts", f"{len(alerts)}{alert_badge}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Asset Allocation")
            fig_pie = go.Figure(data=[go.Pie(
                labels=tickers,
                values=weights,
                hole=0.4,
                textinfo='label+percent',
                marker=dict(colors=px.colors.qualitative.Set3)
            )])
            fig_pie.update_layout(height=380, margin=dict(t=20, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            st.subheader("Cumulative Returns")
            cum = (1 + returns).cumprod()
            port_cum = (1 + perf).cumprod()
            fig_cum = go.Figure()
            for t in tickers:
                fig_cum.add_trace(go.Scatter(x=cum.index, y=cum[t], name=t, mode='lines', opacity=0.6))
            fig_cum.add_trace(go.Scatter(
                x=port_cum.index, y=port_cum,
                name='Portfolio', mode='lines',
                line=dict(width=3, color='black', dash='dash')
            ))
            fig_cum.update_layout(height=380, hovermode='x unified', margin=dict(t=20, b=20))
            st.plotly_chart(fig_cum, use_container_width=True)

    with tab2:
        st.header("Risk Metrics")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Correlation Matrix")
            fig_corr = px.imshow(
                corr_matrix, text_auto='.2f',
                color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect='auto'
            )
            fig_corr.update_layout(height=420)
            st.plotly_chart(fig_corr, use_container_width=True)

        with col2:
            st.subheader("EWMA Volatility (Annualized)")
            ewma_vol = risk_engine.calculate_ewma_volatility(span=30)
            fig_vol = go.Figure()
            for t in tickers:
                fig_vol.add_trace(go.Scatter(x=ewma_vol.index, y=ewma_vol[t], name=t, mode='lines'))
            fig_vol.update_layout(height=420, hovermode='x unified')
            st.plotly_chart(fig_vol, use_container_width=True)

        st.subheader("Value at Risk (VaR) Summary")
        var_rows = []
        for conf in cfg['confidence_levels']:
            var_rows.append({
                'Confidence': f"{conf*100:.0f}%",
                'Parametric VaR': format_currency(var_param[conf]),
                'Historical VaR': format_currency(var_hist[conf]),
                'CVaR (ES)': format_currency(cvar[conf]),
                '% of Portfolio': f"{var_hist[conf]/portfolio_value*100:.2f}%"
            })
        st.dataframe(pd.DataFrame(var_rows), use_container_width=True)

        st.subheader("Stress Test Scenarios")
        stress = risk_engine.stress_test(portfolio_value, weights)
        stress_rows = [{
            'Scenario': k.replace('_', ' ').title(),
            'Shocked Value': format_currency(v['shocked_value']),
            'Loss': format_currency(v['loss']),
            'Loss %': f"{v['loss_percentage']:.1f}%"
        } for k, v in stress.items()]
        st.dataframe(pd.DataFrame(stress_rows), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpe Ratio", f"{sharpe:.4f}")
        c2.metric("Sortino Ratio", f"{sortino:.4f}")
        c3.metric("Max Drawdown", format_percentage(max_dd))

    with tab3:
        st.header("Monte Carlo Simulation")
        c1, c2 = st.columns(2)
        with c1:
            n_sims = st.number_input("Simulations", 1000, 50000, 10000, 1000)
        with c2:
            horizon = st.number_input("Horizon (days)", 30, 1260, 252, 30)

        if st.button("▶️ Run Monte Carlo"):
            with st.spinner("Simulating..."):
                mc = risk_engine.monte_carlo_simulation(portfolio_value, weights, int(n_sims), int(horizon))

            c1, c2, c3 = st.columns(3)
            c1.metric("Mean Final Value", format_currency(mc['mean_final_value']))
            c2.metric("5th Percentile", format_currency(mc['percentiles']['5th']))
            c3.metric("95th Percentile", format_currency(mc['percentiles']['95th']))

            fig_mc = go.Figure()
            for i in range(min(200, int(n_sims))):
                fig_mc.add_trace(go.Scatter(
                    y=mc['simulated_paths'][i] * portfolio_value,
                    mode='lines', line=dict(width=0.4),
                    opacity=0.15, showlegend=False, hoverinfo='skip'
                ))
            mean_path = mc['simulated_paths'].mean(axis=0) * portfolio_value
            fig_mc.add_trace(go.Scatter(
                y=mean_path, name='Mean', mode='lines',
                line=dict(color='red', width=2.5)
            ))
            fig_mc.add_hline(y=portfolio_value, line_dash='dash', line_color='gray',
                             annotation_text='Initial Value')
            fig_mc.update_layout(
                title=f"Monte Carlo — {n_sims:,} Paths × {horizon} Days",
                xaxis_title="Days", yaxis_title="Portfolio Value ($)",
                height=520
            )
            st.plotly_chart(fig_mc, use_container_width=True)

            fig_hist = px.histogram(mc['final_values'], nbins=60, title="Distribution of Final Values")
            fig_hist.add_vline(x=portfolio_value, line_dash='dash', line_color='red',
                               annotation_text='Initial')
            fig_hist.update_layout(height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

    with tab4:
        st.header("ML Risk Prediction")
        c1, c2 = st.columns(2)
        with c1:
            ml_model_type = st.selectbox("Model", ['Random Forest', 'XGBoost'])
        with c2:
            test_size = st.slider("Test Split (%)", 10, 40, 20) / 100

        if st.button("🚀 Train Model"):
            with st.spinner("Training..."):
                try:
                    predictor = RiskPredictor(model_type='classification')
                    features = predictor.prepare_features(returns, prices, volume if not volume.empty else None)
                    target = predictor.create_target_classification(returns, weights)
                    results = predictor.train_model(
                        features, target, test_size=test_size,
                        use_xgboost=(ml_model_type == 'XGBoost')
                    )
                    st.session_state['ml_predictor'] = predictor
                    st.session_state['ml_results'] = results
                    st.success("✅ Model trained")

                    c1, c2 = st.columns(2)
                    c1.metric("Train Accuracy", f"{results['train_accuracy']:.4f}")
                    c2.metric("Test Accuracy", f"{results['test_accuracy']:.4f}")

                    fi = results['feature_importance'].head(15)
                    fig_fi = go.Figure(go.Bar(
                        x=fi['importance'], y=fi['feature'],
                        orientation='h', marker_color='steelblue'
                    ))
                    fig_fi.update_layout(
                        title="Top Feature Importances",
                        yaxis={'categoryorder': 'total ascending'}, height=450
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)

                except Exception as e:
                    st.error(f"Training failed: {str(e)}")

        if 'ml_predictor' in st.session_state:
            try:
                predictor = st.session_state['ml_predictor']
                features = predictor.prepare_features(returns, prices, volume if not volume.empty else None)
                risk_level = predictor.predict_risk_level(features.iloc[-1:])[0]
                color = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🔴'}.get(risk_level, '⚪')
                st.markdown(f"### Current Risk Level: {color} **{risk_level}**")

                alerts_engine.check_ml_signal(risk_level)
            except Exception as e:
                st.warning(f"Prediction unavailable: {str(e)}")

    with tab5:
        st.header("🔔 Risk Alerts")

        if not alerts:
            st.success("✅ No active alerts — portfolio risk is within acceptable thresholds.")
        else:
            h = sev_counts['HIGH']
            m = sev_counts['MEDIUM']
            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 High Alerts", h)
            c2.metric("🟡 Medium Alerts", m)
            c3.metric("Total Alerts", len(alerts))

            st.markdown("---")
            for alert in alerts:
                sev = alert['severity']
                css = 'alert-high' if sev == 'HIGH' else ('alert-medium' if sev == 'MEDIUM' else 'alert-low')
                icon = '🔴' if sev == 'HIGH' else ('🟡' if sev == 'MEDIUM' else '🟢')
                st.markdown(
                    f'<div class="{css}">'
                    f'<strong>{icon} [{sev}] {alert["type"]}</strong><br>'
                    f'{alert["message"]}<br>'
                    f'<small>{alert["detail"]}</small>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    with tab6:
        st.header("💡 Recommendations")

        if not recommendations:
            st.info("No recommendations at this time.")
        else:
            for rec in recommendations:
                pri = rec['priority']
                css = 'rec-high' if pri == 'HIGH' else ('rec-medium' if pri == 'MEDIUM' else 'rec-low')
                icon = '🔴' if pri == 'HIGH' else ('🔵' if pri == 'MEDIUM' else '🟢')
                st.markdown(
                    f'<div class="{css}">'
                    f'<strong>{icon} {rec["category"]}</strong><br>'
                    f'{rec["message"]}<br>'
                    f'<em>→ {rec["action"]}</em>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    with tab7:
        st.header("📋 Risk Report")

        report_rows = [
            {'Metric': 'Portfolio Value', 'Value': format_currency(portfolio_value)},
            {'Metric': 'Total Return', 'Value': format_percentage(total_ret)},
            {'Metric': 'Annualized Return', 'Value': format_percentage(float((returns * weights).sum(axis=1).mean() * 252))},
            {'Metric': 'Annualized Volatility', 'Value': format_percentage(ann_vol)},
            {'Metric': 'Sharpe Ratio', 'Value': f"{sharpe:.4f}"},
            {'Metric': 'Sortino Ratio', 'Value': f"{sortino:.4f}"},
            {'Metric': 'Maximum Drawdown', 'Value': format_percentage(max_dd)},
            {'Metric': 'Active Alerts', 'Value': str(len(alerts))},
        ]
        for conf in cfg['confidence_levels']:
            report_rows.append({'Metric': f'Historical VaR ({conf*100:.0f}%)', 'Value': format_currency(var_hist[conf])})
            report_rows.append({'Metric': f'CVaR ({conf*100:.0f}%)', 'Value': format_currency(cvar[conf])})

        st.dataframe(pd.DataFrame(report_rows), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            positions_csv = pm.positions.to_csv(index=False).encode()
            st.download_button("⬇️ Export Positions CSV", positions_csv, "positions.csv", "text/csv")
        with col2:
            alerts_df = alerts_engine.get_alerts_df()
            if not alerts_df.empty:
                alerts_csv = alerts_df.to_csv(index=False).encode()
                st.download_button("⬇️ Export Alerts CSV", alerts_csv, "alerts.csv", "text/csv")

    st.sidebar.markdown("---")
    st.sidebar.info("**RMS v2.0** — Professional Risk Management System\n\n© 2026")


if __name__ == "__main__":
    main()
