# CLAUDE.md

AI Field Notes: a personal, interactive reference of machine-learning concepts, published to GitHub Pages
(`<user>.github.io/ai-field-notes/`). Each page is one atomic note: a single idea that reads on its own, with
interactive figures and citations.

| Where      | What                                                                                                |
| ---------- | --------------------------------------------------------------------------------------------------- |
| `content/` | Notes (MDX), `taxonomy.yaml`, `reading-order.yaml`, `glossary.yaml`, `references.yaml`, `macros.ts` |
| `site/`    | The React app (Vite root). The npm project is at the repo root.                                     |
| `python/`  | Optional runnable examples and the `mlc` build tool. Only when the owner asks for Python.           |
| `lab/`     | A standalone explorer of the engine (`make lab`). Never imports the site.                           |

## Ground rules

- **Git:** never stage, commit or push unless the current message asks. One request covers one action; "commit" does
  not imply "push". Run `make pre-push` before every push unless the owner waives it for that push. Never add
  `docs/field-notes-survey.md`.
- **Figures and maths are JavaScript on the aifn-engine** (`aifn-compute`, `aifn-methods`, `aifn-render`, installed
  from release tarballs pinned in `package.json`; docs at https://njtwomey.github.io/aifn-engine/). Use Python only
  when the owner asks for it.
- **The engine owns the maths.** Use its functions; do not write numerics, models or training loops in a widget. When
  the engine lacks something, say what it needs (an API sketch) and wait: the owner adds it to the engine. Never edit
  or read the engine repository unless asked; read `node_modules/aifn-*` types instead.
- **Python, when used:** `uv` only (`uv add`, `uv run`), never pip.
- Never hand-edit `site/src/components/ui/` (shadcn CLI) or `site/src/generated/` (`make contracts`).

## Commands

```bash
make dev                                   # http://localhost:5173/ai-field-notes/
npm run check:content                      # validate content/ (seconds)
make doctor SCOPE="<taxonomy path | slug> ..."   # compile + check only those notes; unscoped takes ~1 min
make stubs SCOPE="<taxonomy path> ..."         # list placeholder notes (status: stub); ARGS="--count"
make links SCOPE="<slug>"                  # report unmarked glossary/note mentions; ARGS="--apply" to write
make links ARGS="--term <key|slug>"        # scan the whole tree for one new entry or note
make wrap SCOPE="<slug>"                   # reflow note prose to 120 columns; ARGS="--check" to report
make maths SCOPE="<slug>"                  # put display-maths $$ fences on their own lines (one-line $$…$$ is inline)
make refs ARGS="search --title '<words>' --author <name>"   # fuzzy-search references.yaml (also --key, --year, --url)
make refs ARGS="check --file new.yaml"     # is each new entry already there? exit 1 on a likely duplicate
make refs ARGS="duplicates"                # repeated keys, shared DOIs/URLs, near-identical titles
npm run check:render -- <slug>             # headless render (needs a dev server on 5180: npx vite --port 5180)
make check                                 # doctor + make ci;  make pre-push = check + lockfiles + build
```

Never hand-roll a headless Chrome loop; use `check:render` (`--shot <dir>` for screenshots).

## Notes

- A note is `content/notes/<category path>/<slug>/index.mdx`. The folder path is its category (a path in
  `taxonomy.yaml`); the folder name is its slug, unique across the tree. URLs are `/n/<slug>`, so moving a note
  (`mv`, then `make doctor`) never breaks links. Folders starting with `_` hold shared code.
- Slugs spell ideas out (`kullback-leibler-divergence`, not `kl-divergence`); acronyms go in `aliases`. Proper names
  that are acronyms are the exception (BERT, FAISS).
- Widgets sit next to their note and are imported by it. A note never imports from another note's folder; shared code
  goes in a `_shared/` folder of the category or `site/src/components/widgets/`.
- Frontmatter: `title`, `kind`, `summary`, `tags`, `aliases`, `requires`, `partOf`, `related`, `code`, `references`,
  `status` (`stub | draft | stable`), `updated`. Kinds and their section order are in `docs/templates/`.
- Place a note by subject, not by use. Add a category level only for a chapter of several notes. A category may name
  its introducing note with `index` in `taxonomy.yaml`. A new top-level topic also needs a group in `groups.yaml`.
- **Reading order:** `content/reading-order.yaml` lists each category's notes in reading order (foundations first,
  `requires` before users, examples and comparisons last); categories follow `taxonomy.yaml`. The prev/next arrows and
  every list of notes follow it. Add every new note to it and move its entry when the note moves.
- **Glossary:** `content/glossary.yaml` defines each acronym, recurring concept and proper name once; notes use
  `<Gloss name="key" />` (first use expands, later ones abbreviate; never in headings). Keys and aliases are short
  kebab-case; `long` is fully expanded Title Case; `definition` is one or two sentences (≤ 400 characters); `note` is
  the note about the entry, `see` notes that discuss it. One sense per entry. `npm run check:content` enforces the
  details.
- **New note checklist:** frontmatter, reading-order entry, glossary entries, references with real verified URLs
  (`make refs ARGS="check ..."` first, so an existing key is reused),
  `make links` (read the report, then apply), `make wrap`, `make doctor SCOPE=…`, `check:render` and a look at a
  screenshot.

## Writing

Notes are encyclopedia entries. Every sentence carries information and stands on its own.

- Short, plain, declarative sentences; one claim each. Name the thing rather than "it"/"this" across sentences. Define
  every symbol on first use; give units, shapes and ranges.
- No filler, hedging or salesmanship ("powerful", "crucial", "delve", "simply", "it's worth noting"), no rhetorical
  questions, no em-dash asides, no sign-off summaries.
- Build intuition before formalism: what the idea does and why, then the definition, then the derivation. Worked
  examples go step by step, with each step explained.
- Self-contained: a reader who knows only the `requires` notes understands it. Link related ideas with `<NoteLink>`,
  never rely on them.
- Layers: `summary` (≤ 280 characters, defines the idea), `<Definition>`, the body, optional `<Derivation>` blocks.
  The central result is derived in the main text; secondary proofs go in `<Derivation>`. No circular arguments.
- Maths in `$…$` with the macros in `content/macros.ts` (`\xvec`, `\Xmat`, `\reals`, `\norm{…}`, …); never
  `\mathbf{x}` or Unicode look-alikes. Add a missing macro to `macros.ts`.
- Cite non-obvious claims with `<Cite id="key" />`; sources go in `references.yaml` with real URLs.
- Tag notes with worked examples `worked-example`. Placeholder text only in `status: stub` notes.

## Figures

- Build with the `aifn-render` v2 API: `Figure`, one `useFigureState` per figure (`slider`, `int`, `float`, `choice`,
  `setting`, `toggle`), `Plot` + `useAxis` + layers (`Curve`, `Points`, `Raster`, `Segments`, `Vectors`, …), `Player`,
  `Readout`, `EquationSteps` for stepped derivations, `Diagram` for architecture and graphical-model diagrams.
- Bind any parameter with a natural place on the chart (a point, threshold, centroid, vector tip) to a `Handle`, by
  default. Distribution parameters stay on sliders. Say in the caption what can be dragged.
- Walk-throughs compute their trace up front and scrub it with one `Player` per figure, opening at step 0.
- Training lengths and rates are typed `int`/`float` fields with suggestions, not fixed dropdowns.
- Keep in-browser work light enough for slider drags; heavy work runs in the worker (`useComputed` with
  `mode: 'worker'` and `call('module/export', …)`). Seeded randomness only: `stream(seed)` from
  `aifn-compute/foundation/random`, never `Math.random`.
- Use `useAxis({ equal: x })` whenever shape or angle matters. Plots inside a `Figure` take the frame's height; for a
  grid of panels at a fixed height, wrap them in a `FrameContext` provider (see `ResidualStages.tsx` in the RQ-VAE
  note).
- Colours: data colours from `design/palette.json` via `slot` (fixed per entity, set explicitly when series toggle),
  at most three colour slots in a scatter, one hue for sequential scales, `scale="diverging"` for signed values,
  `emphasis` for highlighted marks. UI: shadcn components (Base UI, compose with `render`), Tailwind, lucide icons
  only.
- Known engine quirks: a `useFigureState` key named after a state member breaks the figure, because it shadows it
  (`values`, `schema`, `set`, `handle`, `bind`, `reset`, `isDefault`, `json`, `setJson`, `dropped`, `attach`); an
  empty `dense` `Points` layer throws (skip empty layers); a `Raster` paints opaquely, so draw cell edges with
  `Segments`; a log axis cannot show zeros or bars starting at 0.

## Python (only when asked)

A note opts in with `code: <id>`, backed by `python/mlc/examples/<name>/` and an entry in `python/runs.toml`
(`uv run mlc runs --only <id>`); the Code and Outputs tabs show it. Heavy figure data can come from a `@figure`
builder in `python/mlc/figures/` (`uv run mlc figures --only <slug>`), loaded with `useFigure<Model>(id)`. Shapes
crossing to the site are pydantic models in `python/mlc/core/contracts.py`; `make contracts` regenerates the
TypeScript types. Python 3.12, modern typing, every CLI a `mlc.core.cli.Command` with described fields. Generated
assets are committed.

## Gotchas

- One copy each of `react`, `aifn-compute` and `aifn-methods` (`npm ls`): compute throws on a second registration.
  Upgrade the three aifn tarball URLs together.
- `katex` is pinned to rehype-katex's version; two copies break sub- and superscripts.
- `*.mdx` is not run through Prettier; `make wrap` reflows prose and `make check` fails on unwrapped paragraphs.
- Tailwind scans `content/` and `node_modules/aifn-render` via `@source` in `site/src/index.css`; a new TSX directory
  outside `site/` needs its own `@source`.
- `base: '/ai-field-notes/'` in `vite.config.ts` is the only place the deploy path is set.
