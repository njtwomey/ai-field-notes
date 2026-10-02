# Common tasks. `make help` lists them.
.DEFAULT_GOAL := help
# Python sources that ruff lints and formats; Pyright reads its include list from pyproject.toml.
PY_SRC := python aifn-js/core/test/fixtures aifn-js/applications/test/fixtures

.PHONY: help install dev contracts assets content doctor links wrap lint aifn-layers aifn-names catalog catalog-check format typecheck test bench aifn-package fixtures fixtures-check lab-check lab-shots lab check build preview clean

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

lint: aifn-layers aifn-names ## Lint TypeScript, Python and note prose
	npx oxlint
	node aifn-lab/check.ts --imports-only
	git ls-files -z -co --exclude-standard | xargs -0 sh -c 'for f; do [ -f "$$f" ] && printf "%s\0" "$$f"; done' _ | xargs -0 npx prettier --check --ignore-unknown
	node scripts/wrap-mdx.ts --check
	uv run ruff check $(PY_SRC)
	uv run ruff format --check $(PY_SRC)

aifn-names: ## Check that no two aifn modules export different values under one name (allowlist in the test)
	npx vitest run --config aifn-js/applications/vitest.config.ts test/names.test.ts

catalog: ## Collect every aifn registry entry into aifn-js/generated/catalog.json and check its note, glossary and reference links
	node scripts/aifn-catalog.ts

catalog-check: ## Check that the aifn catalog is fresh and its links exist; report fixture coverage
	node scripts/aifn-catalog.ts --check

aifn-layers: ## Check aifn-js imports: core tiers, the application area DAG, core never imports applications (aifn-js/modules.json)
	node scripts/aifn-layers.ts

format: ## Format TypeScript, Python and note prose
	git ls-files -z -co --exclude-standard | xargs -0 sh -c 'for f; do [ -f "$$f" ] && printf "%s\0" "$$f"; done' _ | xargs -0 npx prettier --write --ignore-unknown --log-level warn
	node scripts/wrap-mdx.ts
	uv run ruff check --fix $(PY_SRC)
	uv run ruff format $(PY_SRC)

typecheck: ## Type-check TypeScript and Python
	npx tsc -b
	uv run pyright

test: aifn-layers aifn-names ## Run the aifn-js tests (core and applications) and the Python core tests
	npx vitest run --config aifn-js/core/vitest.config.ts
	@# The name lint (test/names.test.ts) already ran as the aifn-names prerequisite; make runs a prerequisite once.
	npx vitest run --config aifn-js/applications/vitest.config.ts --exclude test/names.test.ts
	uv run pytest

bench: ## Run the aifn core micro-benchmarks (reported, not gated; not part of check)
	npx vitest bench --run --config aifn-js/core/vitest.config.ts

aifn-package: ## Build aifn (aifn-js/core) as a publishable package in aifn-js/core/dist (ARGS="--version x.y.z")
	node scripts/aifn-package.ts $(ARGS)

fixtures: ## Regenerate aifn-js golden test values from Python (FIXTURES="numerics/linalg numerics ..." for some)
	uv run python aifn-js/core/test/fixtures/generate.py $(FIXTURES)

fixtures-check: ## Regenerate every aifn-js fixture in memory and fail if any differs from its committed file (slow; not in check)
	uv run python aifn-js/core/test/fixtures/generate.py --check $(FIXTURES)

lab-check: ## Render every aifn lab specimen on the server and report any that throw
	node aifn-lab/check.ts

lab-shots: ## Screenshot aifn lab pages and figures to .scratch/lab-shots (ARGS="--only module/slug --theme dark ...")
	node aifn-lab/screenshot.ts $(ARGS)

lab: ## Start the aifn lab (standalone explorer for aifn) → http://localhost:5190/
	@echo "aifn lab → http://localhost:5190/  (pages at /<module>/<specimen>, figures at #<figure-id>; UI kit at /ui-kit)"
	npx vite --config aifn-lab/vite.config.ts

check: contracts lint doctor typecheck test catalog-check ## Everything CI runs before a build (cheap checks first)
	uv run mlc check
	@# In CI the tree starts clean, so any change after regenerating means the committed contracts were stale.
	@if [ -n "$$CI" ]; then git diff --quiet -- site/src/generated || (echo "contracts out of date: run make contracts" && exit 1); fi

build: assets content ## Production build into dist/
	npm run build

preview: ## Serve dist/ locally
	npm run preview

clean: ## Remove build output and caches (keeps generated assets)
	rm -rf dist node_modules/.tmp node_modules/.vite .ruff_cache .pytest_cache
