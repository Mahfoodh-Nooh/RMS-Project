import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import api_client
from api_client import APIError


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
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Session state helpers
# --------------------------------------------------------------------------- #

def _init_session():
    st.session_state.setdefault("token", None)
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("selected_portfolio_id", None)


def _is_logged_in() -> bool:
    return bool(st.session_state.get("token"))


def _logout():
    st.session_state["token"] = None
    st.session_state["user"] = None
    st.session_state["selected_portfolio_id"] = None


def _format_currency(value: float) -> str:
    try:
        return f"${value:,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _format_percentage(value: float) -> str:
    try:
        return f"{value * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------- #
# Auth UI
# --------------------------------------------------------------------------- #

def render_auth_screen():
    st.title("📊 Risk Management System")
    st.markdown("##### Please sign in to continue")

    if not api_client.health_check():
        st.error("⚠️ Backend is not reachable at `http://127.0.0.1:8000`. Start the FastAPI server first.")
        st.code("uvicorn backend.main:app --reload", language="bash")
        st.stop()

    tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submit = st.form_submit_button("Login")

        if submit:
            if not email or not password:
                st.error("Email and password are required.")
            else:
                try:
                    token = api_client.login_user(email, password)
                    user = api_client.get_current_user(token)
                    st.session_state["token"] = token
                    st.session_state["user"] = user
                    st.success(f"Welcome back, {user.get('full_name', email)}!")
                    st.rerun()
                except APIError as e:
                    st.error(str(e))

    with tab_register:
        with st.form("register_form"):
            r_email = st.text_input("Email", key="reg_email")
            r_name = st.text_input("Full name", key="reg_name")
            r_password = st.text_input("Password", type="password", key="reg_password")
            r_submit = st.form_submit_button("Create account")

        if r_submit:
            if not (r_email and r_name and r_password):
                st.error("All fields are required.")
            else:
                try:
                    api_client.register_user(r_email, r_name, r_password)
                    st.success("Account created. You can now log in.")
                except APIError as e:
                    st.error(str(e))


# --------------------------------------------------------------------------- #
# Sidebar — portfolio selector
# --------------------------------------------------------------------------- #

def render_sidebar():
    token = st.session_state["token"]
    user = st.session_state.get("user") or {}

    st.sidebar.title("👤 Account")
    st.sidebar.write(f"**{user.get('full_name', '-')}**")
    st.sidebar.write(user.get("email", ""))
    if st.sidebar.button("Logout"):
        _logout()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.title("📁 Portfolios")

    try:
        portfolios = api_client.list_portfolios(token)
    except APIError as e:
        st.sidebar.error(str(e))
        portfolios = []

    if portfolios:
        options = {f"{p['name']} (#{p['id']})": p["id"] for p in portfolios}
        current_label = None
        if st.session_state["selected_portfolio_id"]:
            for label, pid in options.items():
                if pid == st.session_state["selected_portfolio_id"]:
                    current_label = label
                    break

        labels = list(options.keys())
        index = labels.index(current_label) if current_label in labels else 0
        selected_label = st.sidebar.selectbox("Select portfolio", labels, index=index)
        st.session_state["selected_portfolio_id"] = options[selected_label]
    else:
        st.sidebar.info("No portfolios yet. Create one below.")
        st.session_state["selected_portfolio_id"] = None

    st.sidebar.markdown("---")
    st.sidebar.subheader("➕ Create Portfolio")
    with st.sidebar.form("create_portfolio_form", clear_on_submit=True):
        new_name = st.text_input("Name")
        new_value = st.number_input("Initial value ($)", min_value=1000.0, value=1_000_000.0, step=10_000.0)
        new_tickers = st.text_area("Tickers (one per line)", placeholder="AAPL\nMSFT\n2222.SR")
        new_quantities = st.text_area("Quantities (one per line, matching order)", placeholder="100\n50\n200")
        create_submit = st.form_submit_button("Create")

    if create_submit:
        tickers = [t.strip().upper() for t in new_tickers.splitlines() if t.strip()]
        try:
            quantities = [float(q.strip()) for q in new_quantities.splitlines() if q.strip()]
        except ValueError:
            st.sidebar.error("Quantities must be numeric.")
            return

        if not new_name:
            st.sidebar.error("Portfolio name is required.")
        elif not tickers:
            st.sidebar.error("At least one ticker is required.")
        elif len(tickers) != len(quantities):
            st.sidebar.error("Tickers and quantities must have the same count.")
        else:
            positions = [{"ticker": t, "quantity": q} for t, q in zip(tickers, quantities)]
            try:
                created = api_client.create_portfolio(token, new_name, new_value, positions)
                st.sidebar.success(f"Created portfolio #{created['id']}")
                st.session_state["selected_portfolio_id"] = created["id"]
                st.rerun()
            except APIError as e:
                st.sidebar.error(str(e))

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Risk Settings")
    period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
    confidence_levels = st.sidebar.multiselect(
        "VaR Confidence Levels",
        [0.90, 0.95, 0.99],
        default=[0.95, 0.99]
    ) or [0.95]

    return {"period": period, "confidence_levels": confidence_levels}


# --------------------------------------------------------------------------- #
# Main dashboard
# --------------------------------------------------------------------------- #

def render_portfolio_overview(portfolio: dict):
    st.subheader("📋 Positions")
    positions = portfolio.get("positions", [])
    if not positions:
        st.info("This portfolio has no positions yet.")
        return

    df = pd.DataFrame(positions)
    st.dataframe(df[["id", "ticker", "quantity"]], use_container_width=True)

    st.metric("Initial Value", _format_currency(portfolio.get("initial_value", 0)))

    st.subheader("Add Position")
    with st.form("add_position_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            new_ticker = st.text_input("Ticker")
        with c2:
            new_qty = st.number_input("Quantity", min_value=0.0, value=100.0, step=1.0)
        with c3:
            submit = st.form_submit_button("Add")
    if submit:
        if not new_ticker:
            st.error("Ticker is required.")
        else:
            try:
                api_client.add_position(
                    st.session_state["token"],
                    portfolio["id"],
                    new_ticker.strip().upper(),
                    float(new_qty),
                )
                st.success("Position added.")
                st.rerun()
            except APIError as e:
                st.error(str(e))

    fig = go.Figure(data=[go.Pie(
        labels=[p["ticker"] for p in positions],
        values=[p["quantity"] for p in positions],
        hole=0.4,
        textinfo="label+percent",
        marker=dict(colors=px.colors.qualitative.Set3),
    )])
    fig.update_layout(title="Quantity Allocation", height=380, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_risk_metrics(portfolio_id: int, cfg: dict):
    token = st.session_state["token"]

    if st.button("▶️ Run Risk Analysis", key="btn_risk"):
        with st.spinner("Calling backend..."):
            try:
                result = api_client.calculate_risk(
                    token,
                    portfolio_id,
                    period=cfg["period"],
                    confidence_levels=cfg["confidence_levels"],
                )
                st.session_state["risk_result"] = result
            except APIError as e:
                st.error(str(e))
                return

    result = st.session_state.get("risk_result")
    if not result or result.get("portfolio_id") != portfolio_id:
        st.info("Click **Run Risk Analysis** to fetch metrics from the backend.")
        return

    c1, c2 = st.columns(2)
    c1.metric("Annualized Volatility", _format_percentage(result["annualized_volatility"]))
    c2.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.4f}")

    st.subheader("Value at Risk (VaR)")
    rows = []
    for conf in cfg["confidence_levels"]:
        key = str(conf)
        rows.append({
            "Confidence": f"{conf * 100:.0f}%",
            "Parametric VaR": _format_currency(result["parametric_var"].get(key, 0)),
            "Historical VaR": _format_currency(result["historical_var"].get(key, 0)),
            "CVaR (ES)": _format_currency(result["cvar"].get(key, 0)),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.caption(f"Analyzed tickers: {', '.join(result.get('tickers', []))}")


def render_stress_test(portfolio_id: int, cfg: dict):
    token = st.session_state["token"]

    if st.button("▶️ Run Stress Test", key="btn_stress"):
        with st.spinner("Running stress scenarios..."):
            try:
                result = api_client.stress_test(
                    token,
                    portfolio_id,
                    period=cfg["period"],
                    confidence_levels=cfg["confidence_levels"],
                )
                st.session_state["stress_result"] = result
            except APIError as e:
                st.error(str(e))
                return

    result = st.session_state.get("stress_result")
    if not result:
        st.info("Click **Run Stress Test** to simulate shock scenarios.")
        return

    rows = [
        {
            "Scenario": scenario.replace("_", " ").title(),
            "Shocked Value": _format_currency(v["shocked_value"]),
            "Loss": _format_currency(v["loss"]),
            "Loss %": f"{v['loss_pct']:.2f}%",
        }
        for scenario, v in result.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_monte_carlo(portfolio_id: int, cfg: dict):
    token = st.session_state["token"]

    c1, c2 = st.columns(2)
    with c1:
        n_sims = st.number_input("Simulations", 100, 50000, 1000, 100)
    with c2:
        horizon = st.number_input("Horizon (days)", 30, 1260, 252, 30)

    if st.button("▶️ Run Monte Carlo", key="btn_mc"):
        with st.spinner("Simulating..."):
            try:
                result = api_client.monte_carlo(
                    token,
                    portfolio_id,
                    period=cfg["period"],
                    confidence_levels=cfg["confidence_levels"],
                    num_simulations=int(n_sims),
                    time_horizon=int(horizon),
                )
                st.session_state["mc_result"] = result
            except APIError as e:
                st.error(str(e))
                return

    result = st.session_state.get("mc_result")
    if not result:
        st.info("Configure parameters and click **Run Monte Carlo**.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Final Value", _format_currency(result["mean_final_value"]))
    c2.metric("Std Deviation", _format_currency(result["std_final_value"]))
    percentiles = result.get("percentiles", {})
    c3.metric("5th Percentile", _format_currency(percentiles.get("5th", 0)))

    st.subheader("Percentile Summary")
    pct_rows = [{"Percentile": k, "Value": _format_currency(v)} for k, v in percentiles.items()]
    st.dataframe(pd.DataFrame(pct_rows), use_container_width=True)

    st.subheader("Monte Carlo VaR")
    mc_var = result.get("mc_var", {})
    var_rows = [{"Confidence": f"{float(k) * 100:.0f}%", "VaR": _format_currency(v)} for k, v in mc_var.items()]
    st.dataframe(pd.DataFrame(var_rows), use_container_width=True)


def render_dashboard():
    cfg = render_sidebar()
    if cfg is None:
        return

    st.title("📊 Risk Management System (RMS)")
    st.caption("API-driven dashboard — all data served by FastAPI backend")
    st.markdown("---")

    portfolio_id = st.session_state.get("selected_portfolio_id")
    if not portfolio_id:
        st.info("👈 Create or select a portfolio from the sidebar to begin.")
        return

    try:
        portfolio = api_client.get_portfolio(st.session_state["token"], portfolio_id)
    except APIError as e:
        st.error(str(e))
        return

    st.header(f"Portfolio: {portfolio['name']}")

    col_del1, _ = st.columns([1, 5])
    with col_del1:
        if st.button("🗑️ Delete Portfolio"):
            try:
                api_client.delete_portfolio(st.session_state["token"], portfolio_id)
                st.session_state["selected_portfolio_id"] = None
                st.session_state.pop("risk_result", None)
                st.session_state.pop("stress_result", None)
                st.session_state.pop("mc_result", None)
                st.success("Portfolio deleted.")
                st.rerun()
            except APIError as e:
                st.error(str(e))

    tab_overview, tab_risk, tab_stress, tab_mc = st.tabs([
        "📊 Overview", "📈 Risk Metrics", "💥 Stress Test", "🔮 Monte Carlo"
    ])

    with tab_overview:
        render_portfolio_overview(portfolio)

    with tab_risk:
        render_risk_metrics(portfolio_id, cfg)

    with tab_stress:
        render_stress_test(portfolio_id, cfg)

    with tab_mc:
        render_monte_carlo(portfolio_id, cfg)

    st.sidebar.markdown("---")
    st.sidebar.caption("**RMS v2.0** — SaaS Edition © 2026")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def main():
    _init_session()

    if not _is_logged_in():
        render_auth_screen()
        return

    render_dashboard()


if __name__ == "__main__":
    main()
