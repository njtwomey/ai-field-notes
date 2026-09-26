# Common tasks. `make help` lists them.
.DEFAULT_GOAL := help
.PHONY: help install dev contracts assets content lint format typecheck test check build preview clean

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Install Node and Python dependencies
	npm install
	uv sync

dev: ## Start the Vite dev server
	npm run dev

contracts: ## Regenerate JSON Schema and TypeScript types from the pydantic contracts
	uv run mlc schema
	node scripts/gen-types.ts

assets: contracts ## Build example outputs and figure data (cached), then the manifest
	uv run mlc build

content: ## Validate content/ (frontmatter, links, citations, taxonomy)
	node scripts/check-content.ts

lint: ## Lint TypeScript and Python
	npx oxlint
	npx prettier --check .
	uv run ruff check python
	uv run ruff format --check python

format: ## Format TypeScript and Python
	npx prettier --write .
	uv run ruff check --fix python
	uv run ruff format python

typecheck: ## Type-check TypeScript and Python
	npx tsc -b
	uv run pyright

test: ## Run Python core tests
	uv run pytest

check: contracts content lint typecheck test ## Everything CI runs before a build
	uv run mlc check
	@git diff --quiet -- site/src/generated || (echo "contracts out of date: run make contracts" && exit 1)

build: assets content ## Production build into dist/
	npm run build

preview: ## Serve dist/ locally
	npm run preview

clean: ## Remove build output and caches (keeps generated assets)
	rm -rf dist node_modules/.tmp node_modules/.vite .ruff_cache .pytest_cache
