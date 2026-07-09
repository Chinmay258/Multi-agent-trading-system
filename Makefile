# =============================================================================
# Trading System — Makefile
# =============================================================================
# One-liners for the common dev/ops flows. Recipes assume a POSIX shell
# (Linux/macOS/WSL/Git-Bash). On native Windows, prefer the Docker targets
# (`make up`, `make run`) or run the underlying command shown in each recipe.
# =============================================================================

COMPOSE ?= docker compose
PY      ?= python
VENV    ?= .venv
TF      ?= terraform
TF_DIR  ?= infra/terraform
PROD    ?= -f docker-compose.yml -f docker-compose.prod.yml

.DEFAULT_GOAL := help

.PHONY: help setup migrate seed run up infra down clean test e2e verify lint format train eval logs ps \
        prod-up prod-down deploy deploy-plan destroy

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup & reproducibility
# ---------------------------------------------------------------------------
setup: ## Create a venv and install the project (dev extras). Then: cp .env.example .env
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"
	@test -f .env || cp .env.example .env
	@echo "Setup complete. TA-Lib is optional (NumPy fallback). For real TA-Lib: pip install -e '.[talib]'"

migrate: ## Apply database migrations (alembic upgrade head)
	$(COMPOSE) run --rm api alembic upgrade head

seed: ## Load the bundled offline sample dataset into the DB (zero external calls)
	$(COMPOSE) run --rm api python scripts/seed.py

# ---------------------------------------------------------------------------
# Run the system
# ---------------------------------------------------------------------------
run: up ## Alias for `up` — start the full keyless paper-trading stack

up: ## Start the full stack (infra + 7 agents + API), keyless paper mode
	$(COMPOSE) up -d

infra: ## Start only Redis + Postgres
	$(COMPOSE) up -d postgres redis

down: ## Stop all services (keep volumes/data)
	$(COMPOSE) down

clean: ## Stop services, remove volumes, and delete local caches (DESTRUCTIVE)
	$(COMPOSE) down -v
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------
test: ## Run the full test suite
	$(PY) -m pytest tests/ -q

e2e: ## Run the end-to-end pipeline test (keyless, bundled data)
	$(PY) -m pytest tests/integration/test_e2e_pipeline.py -q

lint: ## Lint + format check (ruff)
	ruff check .
	ruff format --check .

format: ## Auto-format the codebase (ruff)
	ruff format .
	ruff check --fix .

verify: ## Print a green/red health table for all services
	$(PY) scripts/healthcheck.py

# ---------------------------------------------------------------------------
# ML / evaluation (fleshed out in Phases 4 & 5)
# ---------------------------------------------------------------------------
train: ## Train the XGBoost model from historical data
	$(PY) scripts/train_model.py

eval: ## Run the evaluation harness → backtest/results/ (JSON + HTML + PDF)
	$(PY) -m backtest.run

# ---------------------------------------------------------------------------
# Production stack (local) + cloud deploy (Terraform). See docs/DEPLOYMENT.md.
# ---------------------------------------------------------------------------
prod-up: ## Run the production stack locally (Caddy HTTPS front door); DOMAIN=localhost
	DOMAIN=$${DOMAIN:-localhost} $(COMPOSE) $(PROD) up -d --build

prod-down: ## Stop the local production stack
	$(COMPOSE) $(PROD) down

deploy-plan: ## Preview the cloud deployment (terraform plan), no changes made
	cd $(TF_DIR) && $(TF) init && $(TF) plan

deploy: ## Provision the cloud VM + stack (Oracle free tier). See docs/DEPLOYMENT.md
	cd $(TF_DIR) && $(TF) init && $(TF) apply

destroy: ## Tear DOWN all cloud infrastructure (removes any billing risk)
	cd $(TF_DIR) && $(TF) destroy

# ---------------------------------------------------------------------------
# Observability helpers
# ---------------------------------------------------------------------------
logs: ## Tail logs for the core trading agents
	$(COMPOSE) logs -f market_data_agent technical_analysis_agent decision_agent risk_agent execution_agent

ps: ## Show container status
	$(COMPOSE) ps
