-include .env
export

PYTHON ?= python3

.PHONY: install db load test stop

install:
	$(PYTHON) -m venv .venv && .venv/bin/pip install -e ".[dev]"

db:
	docker compose up -d db

load:
	.venv/bin/python -m welldata.load

test:
	.venv/bin/pytest

stop:
	docker compose down
