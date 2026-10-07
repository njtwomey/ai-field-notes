# Common tasks. `make help` lists them.
.DEFAULT_GOAL := help
# Python sources that ruff lints and formats; Pyright reads its include list from pyproject.toml.
PY_SRC := python

.PHONY: ci help install dev contracts assets content doctor links wrap lint catalog-check format typecheck test lab-check lab-shots lab check pre-push build preview clean

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

lint: ## Lint TypeScript, Python and note prose
	npx oxlint
	node lab/check.ts --imports-only
	git ls-files -z -co --exclude-standard | xargs -0 sh -c 'for f; do [ -f "$$f" ] && printf "%s\0" "$$f"; done' _ | xargs -0 npx prettier --check --ignore-unknown
	node scripts/wrap-mdx.ts --check
	uv run ruff check $(PY_SRC)
	uv run ruff format --check $(PY_SRC)

catalog-check: ## Check that every registry entry of the installed aifn-compute and aifn-methods links to notes, glossary keys and references that exist
	node scripts/aifn-catalog.ts

format: ## Format TypeScript, Python and note prose
	git ls-files -z -co --exclude-standard | xargs -0 sh -c 'for f; do [ -f "$$f" ] && printf "%s\0" "$$f"; done' _ | xargs -0 npx prettier --write --ignore-unknown --log-level warn
	node scripts/wrap-mdx.ts
	uv run ruff check --fix $(PY_SRC)
	uv run ruff format $(PY_SRC)

typecheck: ## Type-check TypeScript and Python
	npx tsc -b
	uv run pyright

test: ## Run the Python core tests (the engine's own tests run in its repository)
	uv run pytest

lab-check: ## Render every aifn lab specimen on the server and report any that throw
	node lab/check.ts

lab-shots: ## Screenshot aifn lab pages and figures to .scratch/lab-shots (ARGS="--only module/slug --theme dark ...")
	node lab/screenshot.ts $(ARGS)

lab: ## Start the aifn lab (an explorer for the aifn engine packages) → http://localhost:5190/
	@echo "aifn lab → http://localhost:5190/  (pages at /<module>/<specimen>, figures at #<figure-id>; UI kit at /ui-kit)"
	npx vite --config lab/vite.config.ts

ci: contracts lint typecheck test catalog-check ## What CI runs before a build: make check without the doctor
	uv run mlc check
	@# In CI the tree starts clean, so any change after regenerating means the committed contracts were stale.
	@if [ -n "$$CI" ]; then git diff --quiet -- site/src/generated || (echo "contracts out of date: run make contracts" && exit 1); fi

check: doctor ci ## Everything: the doctor (slow; local only, before every push) plus what CI runs

pre-push: check ## Run before every push: make check, the lockfile checks and the production build CI deploys
	uv lock --check
	npm ci --dry-run --ignore-scripts > /dev/null
	npm run build
	@git diff --quiet -- site/src/generated || echo "note: site/src/generated changed; commit it with the push, or CI fails on stale contracts"

build: assets content ## Production build into dist/
	npm run build

preview: ## Serve dist/ locally
	npm run preview

clean: ## Remove build output and caches (keeps generated assets)
	rm -rf dist node_modules/.tmp node_modules/.vite .ruff_cache .pytest_cache
