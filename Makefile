SHELL := /bin/bash

-include .env
export

COMPOSE ?= docker compose
PYTHON ?= python3
NPM ?= npm
REDIS_URL_LOCAL ?= redis://localhost:6379/0
API_HOST ?= 0.0.0.0
API_PORT ?= 8000
FRONTEND_HOST ?= 0.0.0.0
FRONTEND_PORT ?= 3000

UID := $(shell id -u)
GID := $(shell id -g)

.PHONY: help dirs redis-up redis-down redis-logs docker-up docker-down docker-logs dev-api dev-worker dev-frontend dev test test-py test-frontend test-e2e

help:
	@echo "Available targets:"
	@echo "  make redis-up        Start Redis in Docker"
	@echo "  make redis-down      Stop Redis container"
	@echo "  make redis-logs      Tail Redis logs"
	@echo "  make docker-up       Start full app with Docker Compose"
	@echo "  make docker-down     Stop full app stack"
	@echo "  make docker-logs     Tail Docker Compose logs"
	@echo "  make dev-api         Run FastAPI locally against Docker Redis"
	@echo "  make dev-worker      Run Celery worker locally against Docker Redis"
	@echo "  make dev-frontend    Run Vite locally"
	@echo "  make dev             Start Redis in Docker and run api/worker/frontend locally"
	@echo "  make test            Run Python tests, frontend unit tests, and E2E tests"
	@echo "  make test-py         Run Python tests"
	@echo "  make test-frontend   Run frontend unit tests"
	@echo "  make test-e2e        Run Playwright E2E tests"
	@echo "  make dirs            Create host volume directories (outputs, checkpoints)"

dirs:
	mkdir -p outputs checkpoints

redis-up:
	$(COMPOSE) up -d redis

redis-down:
	$(COMPOSE) stop redis

redis-logs:
	$(COMPOSE) logs -f redis

docker-up: dirs
	UID=$(UID) GID=$(GID) $(COMPOSE) up --build

docker-down:
	$(COMPOSE) down

docker-logs:
	$(COMPOSE) logs -f

dev-api:
	CELERY_BROKER_URL=$(REDIS_URL_LOCAL) \
	CELERY_RESULT_BACKEND=$(REDIS_URL_LOCAL) \
	$(PYTHON) -m uvicorn src.webapp.api:app --host $(API_HOST) --port $(API_PORT) --reload

dev-worker:
	CELERY_BROKER_URL=$(REDIS_URL_LOCAL) \
	CELERY_RESULT_BACKEND=$(REDIS_URL_LOCAL) \
	$(PYTHON) -m celery -A src.webapp.tasks worker --loglevel=$${CELERY_WORKER_LOGLEVEL:-info} --concurrency=$${CELERY_WORKER_CONCURRENCY:-1}

webapp/node_modules: webapp/package.json
	$(NPM) --prefix webapp install

dev-frontend: webapp/node_modules
	cd webapp && VITE_API_PROXY_TARGET=http://localhost:$(API_PORT) $(NPM) run dev -- --host $(FRONTEND_HOST) --port $(FRONTEND_PORT)

dev: webapp/node_modules
	@$(MAKE) redis-up
	@trap 'kill 0' INT TERM EXIT; \
	CELERY_BROKER_URL=$(REDIS_URL_LOCAL) \
	CELERY_RESULT_BACKEND=$(REDIS_URL_LOCAL) \
	$(PYTHON) -m uvicorn src.webapp.api:app --host $(API_HOST) --port $(API_PORT) --reload & \
	CELERY_BROKER_URL=$(REDIS_URL_LOCAL) \
	CELERY_RESULT_BACKEND=$(REDIS_URL_LOCAL) \
	$(PYTHON) -m celery -A src.webapp.tasks worker --loglevel=$${CELERY_WORKER_LOGLEVEL:-info} --concurrency=$${CELERY_WORKER_CONCURRENCY:-1} & \
	(cd webapp && VITE_API_PROXY_TARGET=http://localhost:$(API_PORT) $(NPM) run dev -- --host $(FRONTEND_HOST) --port $(FRONTEND_PORT)) & \
	wait

test: test-py test-frontend test-e2e

test-py:
	$(PYTHON) -m pytest -q

test-frontend: webapp/node_modules
	$(NPM) --prefix webapp test

test-e2e: webapp/node_modules
	$(NPM) --prefix webapp run test:e2e
