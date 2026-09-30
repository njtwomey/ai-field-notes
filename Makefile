# Common tasks. `make help` lists them.
.DEFAULT_GOAL := help
.PHONY: help install dev contracts assets content doctor links lint aifn-layers format typecheck test check build preview clean

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

links: ## Suggest <Gloss> and <NoteLink> markup (SCOPE="taxonomy/path slug ...", ARGS="--apply" to write it)
	node scripts/suggest-links.ts $(ARGS) $(SCOPE)

wrap: ## Rewrap note prose to 120 characters, verified by parse and render (SCOPE="taxonomy/path slug ...", ARGS="--check" to report only)
	node scripts/wrap-mdx.ts $(ARGS) $(SCOPE)

doctor: ## Check the content tree (SCOPE="taxonomy/path slug ..." limits per-note checks to those notes)
	node scripts/doctor.ts $(SCOPE)

lint: aifn-layers ## Lint TypeScript, Python and note prose
	npx oxlint
	git ls-files -z | xargs -0 npx prettier --check --ignore-unknown
	node scripts/wrap-mdx.ts --check
	uv run ruff check python
	uv run ruff format --check python

aifn-layers: ## Check that aifn-js modules import only from lower tiers (table in aifn-js/README.md)
	node scripts/aifn-layers.ts

format: ## Format TypeScript, Python and note prose
	git ls-files -z | xargs -0 npx prettier --write --ignore-unknown --log-level warn
	node scripts/wrap-mdx.ts
	uv run ruff check --fix python
	uv run ruff format python

typecheck: ## Type-check TypeScript and Python
	npx tsc -b
	uv run pyright

test: aifn-layers ## Run the aifn-js tests and the Python core tests
	npx vitest run --config aifn-js/vitest.config.ts
	uv run pytest

fixtures: ## Regenerate aifn-js golden test values from Python (FIXTURES="module ..." for some)
	uv run python aifn-js/test/fixtures/generate.py $(FIXTURES)

lab-check: ## Render every aifn lab specimen on the server and report any that throw
	node aifn-lab/check.ts

lab-shots: ## Screenshot aifn lab pages and figures to .scratch/lab-shots (ARGS="--only module/slug --theme dark ...")
	node aifn-lab/screenshot.ts $(ARGS)

lab: ## Start the aifn lab (standalone explorer for aifn) → http://localhost:5190/
	@echo "aifn lab → http://localhost:5190/  (pages at /<module>/<specimen>, figures at #<figure-id>; UI kit at /ui-kit)"
	npx vite --config aifn-lab/vite.config.ts

check: contracts doctor lint typecheck test ## Everything CI runs before a build
	uv run mlc check
	@# In CI the tree starts clean, so any change after regenerating means the committed contracts were stale.
	@if [ -n "$$CI" ]; then git diff --quiet -- site/src/generated || (echo "contracts out of date: run make contracts" && exit 1); fi

build: assets content ## Production build into dist/
	npm run build

preview: ## Serve dist/ locally
	npm run preview

clean: ## Remove build output and caches (keeps generated assets)
	rm -rf dist node_modules/.tmp node_modules/.vite .ruff_cache .pytest_cache
