# Risk Management System (RMS) – Progress Report

**Date:** May 2026  
**Status:** MVP Backend Complete, Frontend Integration Pending  

---

## 1. Project Overview

The Risk Management System (RMS) is a platform designed to help portfolio managers and investors analyze and monitor financial risks. The system computes industry-standard metrics—including Value at Risk (VaR), Conditional Value at Risk (CVaR), and volatility—and presents them through an interactive dashboard. 

Originally built as a monolithic Streamlit application, the project is currently transitioning to a multi-tenant Software-as-a-Service (SaaS) architecture. This involves decoupling the user interface from the core logic by introducing a FastAPI backend. This new architecture will enable independent scaling, multi-user support, and easier deployment.

---

## 2. Completed Work

The transition to a SaaS architecture has achieved several key milestones:

### Original System (Phase 1)
- **Quantitative Engine:** Developed comprehensive Python modules (`src/risk_engine.py`, `src/portfolio.py`) for VaR, volatility, Monte Carlo simulations, stress testing, and portfolio optimization.
- **Interactive UI:** Built a functional Streamlit dashboard to visualize risk metrics, simulate portfolios, and generate basic alerts.
- **Data Integration:** Implemented a data loader for fetching historical market data.

### SaaS Refactoring (Phase 2 - Backend MVP)
- **FastAPI Backend:** Created a structured, high-performance API backend under the `backend/` directory.
- **Authentication:** Implemented user registration, secure login, and JWT (JSON Web Token) issuance to protect endpoints.
- **API Endpoints:** Built RESTful routes for:
  - **Auth:** `/api/v1/auth` (User identity management)
  - **Portfolios:** `/api/v1/portfolios` (Create, read, update, and delete portfolios and positions)
  - **Risk:** `/api/v1/risk` (Trigger analysis and stress testing)
- **Database Modeling:** Integrated SQLAlchemy with an SQLite database to store user credentials, portfolio configurations, and positions.
- **API Documentation:** Auto-generated interactive documentation available via Swagger (`/docs`).

---

## 3. Current System State

- **Backend:** The FastAPI backend is fully operational locally. It successfully handles authentication, manages portfolios in the database, and processes risk analysis requests using the core quantitative engine.
- **Frontend:** The Streamlit dashboard is fully functional but currently operates independently. It still imports the local Python modules directly rather than communicating with the new API.
- **Infrastructure:** The system is using SQLite, which is suitable for development but not for production concurrency. The application is not yet deployed to the cloud.

The primary limitation holding the system back from being a true SaaS product is the lack of connection between the frontend and the new backend API.

---

## 4. Remaining Work

To complete the SaaS transition and prepare for launch, the following tasks must be completed:

- **API Integration:** The Streamlit dashboard must be refactored to consume the FastAPI endpoints via HTTP requests, abandoning direct module imports.
- **Session Management:** The frontend needs a login screen and a mechanism to store and pass JWT tokens with every API request.
- **Database Upgrade:** Migrate from SQLite to a production-grade relational database (PostgreSQL) to handle concurrent multi-user traffic.
- **Security Hardening:** Secure API configurations (e.g., restricting CORS, managing environment variables).
- **Asynchronous Processing:** Offload heavy computations (like Monte Carlo simulations) to background workers (e.g., Celery) to prevent API timeouts.

---

## 5. Next Steps

1. **Frontend-Backend Connection (Immediate Priority):** Refactor the Streamlit dashboard to authenticate users and fetch all portfolio and risk data via the FastAPI backend.
2. **Database Migration:** Replace SQLite with PostgreSQL and set up database migration tools (Alembic).
3. **Containerization & Deployment:** Finalize Docker configurations for all services (frontend, backend, database) and deploy the system to a cloud provider.