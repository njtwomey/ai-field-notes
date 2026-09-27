# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AI Field Notes: a personal, interactive reference of machine-learning concepts, published to GitHub Pages at
`<user>.github.io/ai-field-notes/`. Each page is one **atomic note**: a single idea that reads on its own. It has
interactive figures, runnable Python and citations. Notes link to each other, but a note must never depend on another to
be understood.

The repository separates three concerns. Keep them separate.

| Concern           | Where      | What it owns                                                                 |
| ----------------- | ---------- | ---------------------------------------------------------------------------- |
| Content           | `content/` | MDX notes, `references.yaml`, `taxonomy.yaml`, reusable static assets        |
| Presentation (UX) | `site/`    | React app: layout, navigation, search, content components, charts            |
| Code and data     | `python/`  | Runnable examples, figure-data builders, the build tool `mlc`, the contracts |

`design/palette.json` is the single source of data colours for both TypeScript and Python.

## Commands

`make help` lists every target. The main ones:

```bash
make install     # npm install + uv sync
make dev         # Vite dev server → http://localhost:5173/ai-field-notes/
make assets      # contracts + run examples + build figure data (cached) + manifest
make contracts   # pydantic → JSON Schema → site/src/generated/contracts.ts
make check       # doctor, lint, typecheck, tests, registry check, contract drift
make doctor      # content tree: slugs, folders vs taxonomy, figure/example ids, cross-note imports
make build       # assets + production build into dist/
make format      # prettier + ruff
```

Targeted commands:

```bash
uv run pytest python/tests/mlc/core/test_contracts.py::test_table_must_be_rectangular   # one test
uv run mlc runs --only k-means --force          # re-run one example's runs, ignoring the cache
uv run mlc figures --only logistic-regression   # rebuild one note's figure data
uv run mlc list                                 # registered examples, runs and figures
uv run python -m mlc.examples.kmeans fit --help # every CLI documents itself
npm run check:content                           # validate content/ without starting Vite
```

## Architecture

### Content pipeline

- A note is `content/notes/<category path>/<slug>/index.mdx`, e.g.
  `content/notes/probability/theory/bayes-theorem/index.mdx`. The folder path is the note's category (there is no
  `category` frontmatter) and the folder name is its slug. Depth varies by branch. URLs use the slug only
  (`/n/<slug>`), so moving a note between categories never changes its URL: the taxonomy is plastic, links are not.
  Slugs are one flat namespace and must be unique across the whole tree.
- A folder holding `index.mdx` is a note and contains no other notes. Any other folder is a category and must exist in
  `content/taxonomy.yaml`. Folders starting with `_` hold shared code and are skipped.
- Note-specific widgets sit next to the note (`<note folder>/Widget.tsx`) and are imported by the MDX file. A note
  never imports from another note's folder: code shared by several notes goes in `site/src/components/widgets/`,
  `site/src/lib/`, or a `_shared/` folder in the category.
- `make doctor` (part of `make check`) checks what the build cannot: unique slugs, folders against the taxonomy,
  `code:` ids against `runs.toml`, `useFigure` ids against notes and generated data, and imports across notes. It also
  warns about empty or single-note categories and names (slugs, titles, aliases) that collide across notes. Move a
  note with a plain `mv` into another category folder, then run `make doctor`.
- `plugins/content-index.ts` (Vite plugin) reads every note's frontmatter and validates it with zod schemas from
  `site/src/lib/content-schema.ts`. It checks categories, relation slugs, `<NoteLink to>` targets and `<Cite id>` keys.
  Any error fails dev and build with the offending file named. It exposes two virtual modules:
  - `virtual:content`: metadata, references and taxonomy. Loaded eagerly.
  - `virtual:search`: plain-text bodies. Loaded only when search first opens.
- MDX is compiled by `@mdx-js/rollup` with remark-gfm, remark-math + rehype-katex, and rehype-slug. Components in
  `site/src/components/content/mdx-components.tsx` are available in every note without an import.
- Note bodies load lazily via `import.meta.glob` in `site/src/lib/content.ts`.

### Code and output pipeline

```
python/runs.toml ──uv run mlc runs──▶ site/public/generated/runs/<example>/<run>/run.json (+ assets)
@figure builders ──uv run mlc figures──▶ site/public/generated/figures/<id>.json
                        both ──▶ site/public/generated/manifest.json  (the site's single entry point)
python/mlc/core/contracts.py ──mlc schema──▶ contracts.schema.json ──scripts/gen-types.ts──▶ contracts.ts
```

- **`python/runs.toml`** is the central record of which example backs which note. A note opts in with `code: <id>` in
  its frontmatter, which enables its Code and Outputs tabs.
- An **example** is a package `python/mlc/examples/<name>/` with `__main__.py` (the CLI) and plain modules (the
  algorithm). Every non-empty `.py` file in it appears in the Code tab. Each `[[examples.runs]]` entry is one CLI
  invocation. Its outputs appear in the Outputs tab.
- Examples record outputs with `mlc.core.emit` (`metrics`, `table`, `chart`, `heatmap`, `text`, `file`). Each call
  prints normally. Under `mlc runs` it also appends a typed record to `outputs.jsonl`. The site renders each record kind
  in `site/src/components/code/OutputView.tsx`. That switch is exhaustive, so a new kind fails `tsc` until it is
  rendered.
- **Figure builders** (`python/mlc/figures/<topic>.py`, decorated `@figure("<note-slug>/<name>", title=...)`) compute
  data that an interactive figure needs but should not compute in the browser. They return pydantic models. The site
  loads them with `useFigure<ModelName>(id)`.
- **Caching:** a run or figure rebuilds only when the hash of its sources, `mlc.core`, `uv.lock` and its arguments
  changes. Generated assets are **committed**. CI does not re-run examples; it only checks that they are current.

### Contracts: generated, never mirrored by hand

`python/mlc/core/contracts.py` is the only definition of every shape that crosses from Python to the site. TypeScript
types in `site/src/generated/contracts.ts` are **compiled** from it (`make contracts`). Never edit that file. Never
write a TypeScript type that restates a Python model. Hand-maintained parity between languages is an anti-pattern here.
To change a shape, edit the pydantic model and run `make contracts`. `make check` fails if the generated files are
stale.

- Figure-data primitives live in `contracts.py`: `PointCloud2d`, `Curve`, `CurveSet`, `Grid2d`, `ContingencyTable`,
  `RocCurve`. Builders return one of these or a small `Strict` model composed of them (see `LossSurface`). Add a new
  primitive to `contracts.py` (and to `ROOTS` in `python/mlc/build/schema.py`) when a shape is reusable. Figure return
  models are added to the schema automatically.
- Do not use tuples in contracts. JSON Schema tuples compile to `unknown[]` in TypeScript. Use a small model such as
  `Point2d`.
- The schema is exported in serialization mode, so fields with defaults are required in TypeScript. When a component
  accepts a contract type as input, loosen it at the prop (`Omit<Series, 'group'> & { group?: ... }`), not in Python.

### Site layout

- `site/src/components/ui/`: shadcn components, generated by the CLI. Do not hand-edit. Add with
  `npx shadcn@latest add <name>`. The style is `base-nova`, built on **Base UI**, not Radix: compose with the `render`
  prop, not `asChild`.
- `site/src/components/viz/`: the visual system (see below). `site/src/components/content/`: MDX components.
  `site/src/components/note/`: the note page. `site/src/components/layout/`: shell, search, navigation.
- The note page opens with `NoteBar`, which sticks under the site header and is the page's only breadcrumb:
  `TaxonomyTrail` (every category crumb a dropdown of its siblings, with a link to browse it; the note's title, bold at
  the same size, a dropdown of the other notes in its category), the status, then "›" and the section being read, a
  reading-progress line, and the Concept / Code / Outputs tabs. Below `md` only the title crumb shows. `NoteHeader`
  follows: the centred title (the page's h1) and the summary set as an abstract, with tags beneath.
  `useReadingPosition` computes the current heading and progress once, for both the bar and the left index. Sticky
  offsets assume a 56 px site header and a 56 px note bar.
- The note page (`site/src/pages/NotePage.tsx`) spans the full window width in three columns. The left index is sticky
  and scrolls on its own: the TOC on Concept, files on Code, runs on Outputs. The centre holds the content. The right
  column holds relations and, on the Concept tab at `xl` and wider, **margin references**. Tabs are routes: `/n/:slug`,
  `/n/:slug/code`, `/n/:slug/outputs/:run`.
- Margin references (`site/src/components/content/sidenotes.tsx`): every `<Cite>` marker registers its element, and
  `MarginNotes` places each reference beside the line that cites it, pushing notes down to avoid overlap. It
  re-lays out when the article resizes. Hovering a marker highlights its note and the reverse. Below `xl`, markers
  show a hover card instead. Every note ends with a full References list (`ReferenceList`), which is also the target
  of each marker's link.
- Navigation: the landing page lists every topic as a tile with its subtopics and note counts, never individual notes.
  `/browse` is the explorer: a topic rail with counts, kind chips, a text filter and a List / Map toggle, all held in
  the URL (`browseUrl({ c, kind, q, view })`). The Map (`ConceptMap`) draws notes as a force graph coloured and shaped
  by topic, with requires / part-of / related edges. Top-level topics carry an `icon` in `taxonomy.yaml`.
- Tags are always shown with `TagPill` (`site/src/components/browse/TagPill.tsx`), never as ad-hoc text or badges.
- Search is a ⌘K / `/` command palette (shadcn `Command`, filtering off) over a MiniSearch index of titles, aliases,
  tags, summaries, headings and body text.

## UI rules

- **Typography.** Article text (`.note-prose`, `font-prose`) is set in KaTeX's Computer Modern face (`KaTeX_Main`) at
  the same size as the maths, as in LaTeX. Headings, definitions, callout bodies, derivations and the note title use
  it too. UI chrome (navigation, controls, widget captions, badges, the margin notes) stays in Geist. Do not raise
  `.katex` above `1em` inside prose. Write maths in `$…$`, not Unicode look-alikes, so that it renders in the maths
  font.

- **Components: shadcn only. Styling: Tailwind only. Icons: lucide-react only. Charts: ECharts only.** Do not add
  another component library, CSS-in-JS, icon set or chart library. Colours come from shadcn CSS variables
  (`bg-muted`, `text-muted-foreground`, …), never raw hex, except data colours from the palette.
- `EChart.tsx` is the only file that touches ECharts. Register new ECharts chart types or components once in
  `viz/echarts.ts` (tree-shaken build). All chart styling defaults live in `viz/theme.ts`.
- Build figures from the existing primitives: `XYChart`, `Heatmap`, `GraphDiagram` (fixed-layout graphical models and
  factor graphs), `Interactive` (the standard frame: title,
  controls, figure, readout, caption), `ParamSlider`, `ParamChoice`, `ParamSwitch`, `ParamButton`, `Readout`. Add a new
  primitive to `viz/` and export it from `viz/index.ts` rather than styling ECharts inside a note.
- Data colours follow `design/palette.json`:
  - Categorical slots are assigned in fixed order by entity, never by rank, never cycled. Pass `slot` explicitly when
    series can be toggled, so that colours do not shift.
  - Scatter-type charts use at most three colour slots unless points also differ by marker shape. Grouped scatter
    series get shapes automatically.
  - Sequential scales use one hue, light to dark. Signed values use `scale="diverging"`, which has a neutral midpoint.
  - Highlighted marks such as centroids use `emphasis` (ink colour), not a palette slot.
- Interactive performance:
  - `ParamSlider` debounces `onChange` (default 60 ms, `debounceMs` to change). The thumb moves immediately and the
    final value always commits on release. Do not add debouncing in widgets.
  - Chart components memoise their ECharts option. Memo dependencies must be values, not inline arrays or objects
    (`range={[0, 3]}` is a new array each render), or every render redraws the whole chart.
  - Put small, fast-changing parts of a chart (a marker, a cursor) in `EChart`'s `patch` prop, keyed by series `id`,
    so they update without redrawing the rest. `Heatmap`'s `marker` works this way.
- Direct manipulation: when a parameter has an obvious place on a chart (a mean, a threshold, a start point, a
  centroid, a vector tip, a rank on a spectrum), bind it to a draggable handle as well as its slider. Do this by
  default, without being asked. The handle must be the thing itself, not a proxy: dragging a distribution's mean line
  to reshape the whole curve feels wrong, so distribution parameters stay on sliders.
  - Pass `handles` (see `site/src/components/viz/handles.ts`: `point`, `x` or `y`) to `XYChart`, `Heatmap` or
    `EChart`. Each handle's `onDrag` writes the same state the slider reads; `useParam` clamps and snaps it to the
    slider's range, and `ParamSlider param={…}` binds the slider.
  - There are no update loops: charts emit only on pointer events, never when their props change. Axes freeze during a
    drag, so a range that depends on the dragged value cannot rescale under the pointer; they refit on release.
  - With one handle, pressing anywhere on the plot moves it. With several, the nearest within `GRAB_RADIUS` wins.
    Prefer handles to `onPlotClick`/`onCellClick`, and say in the caption what can be dragged.
- A slider that walks through a sequence (iterations, updates, sweeps, rounds, steps, frames) gets `withArrows`, so
  the reader can step one at a time.
- In-browser computation must stay light enough for slider drags. Seeded randomness uses `rng(seed)` from
  `site/src/lib/math`, never `Math.random`. Anything heavier becomes a Python `@figure` builder.
- Tailwind generates only the classes it finds. `site/src/index.css` has `@source '../../content'` so that classes used
  in note widgets exist. Any new directory holding TSX outside `site/` needs its own `@source`.
- **Diagrams are not charts.** Architecture diagrams, flow charts and graphical models use `Diagram`
  (`site/src/components/diagram/`): a hand-specified SVG diagram, not auto-layout. Nodes are placed on a grid (centres,
  grid units), groups are drawn around nodes or at rectangles (plates are groups labelled bottom-right), and edges are
  routed at right angles through ports (`id:n|s|e|w`) and waypoints, or drawn `straight`/`curve` for graphical models.
  Labels are KaTeX. Reusable pieces (`op`, `gate`, `projector` encoder/decoder trapezoids, `reparam`) live in
  `components.ts`; add a component there rather than repeating a pattern. `/lab/diagrams` is the test bench. The older
  `GraphDiagram` (ECharts) still backs existing graphical-model figures.
- `XYChart equalAspect` gives equal pixel length per unit on both axes, for any ranges. Use it whenever a shape or
  angle matters: an ellipse, a normal vector against a boundary. Arrows are drawn with the `vectors` prop.
- Reusable static images go in `content/assets/` and are placed with `<Asset name="…" />`. Never copy an image into
  several notes.

## Python rules

- Python 3.12 with modern typing: PEP 695 `type` aliases and generics, `Self`, `X | None`. No `from __future__`.
- Pydantic wherever data has a shape: contracts, figure data, CLI options. `Strict` models forbid extra fields and are
  frozen.
- Every CLI is a `mlc.core.cli.Command` subclass (pydantic-settings). Options are fields. Every field needs
  `Field(description=...)` and every command needs a docstring; both are enforced at class creation. Subcommands use
  `CliSubCommand`. Field names become kebab-case flags; single-letter fields become short flags (`-k`, `-n`).
- Examples import only `mlc.core.emit` and `mlc.core.cli` from `mlc`, plus numpy and similar libraries. They must stay
  readable as standalone code, because readers see them in the Code tab.
- One `pyproject.toml` and one environment for everything. Add dependencies with `uv add`.
- Tests cover `mlc.core` only (the contracts and registration). They mirror the source tree:
  `python/tests/mlc/core/test_<module>.py` tests `python/mlc/core/<module>.py`. Examples and figures are not tested.
- Pyright is strict for `mlc/core`, `mlc/build` and tests, and standard elsewhere. Ruff enforces 120-character lines.

## Code style

- Line length is 120 characters in TypeScript (Prettier) and Python (Ruff).
- Match surrounding code. Comment the non-obvious _why_, not the _what_.

## Writing style for notes

Notes are encyclopedia entries, not blog posts. Every sentence must carry information and stand on its own.

- **Short, plain, declarative sentences.** One claim per sentence. Prefer "X is Y." to clauses stacked with commas.
- **Unambiguous.** Name the thing each time rather than using "it" or "this" across sentences. Define every symbol on
  first use. Give units, shapes and ranges.
- **No filler, hedging or salesmanship.** No "powerful", "crucial", "elegant", "delve", "landscape", "it's worth
  noting", "in essence", "simply", "of course", rhetorical questions, "not just X but Y", or tricolons for rhythm. No
  em-dash asides. No sign-off summaries.
- **Self-contained.** A reader who knows only the prerequisites in `requires` must understand the note. Mention
  related ideas with `<NoteLink>`, but never rely on the linked note to finish an explanation.
- **Layered depth.**
  - The frontmatter `summary` (≤ 280 characters as read) defines the idea completely. It appears in search results, hover
    cards and lists. Write its maths as `$…$`, which is rendered with KaTeX and the macros; never Unicode look-alikes
    such as ∑, ≤, ′, ² or ₁.
  - The `<Definition>` block states it precisely.
  - The body adds detail in the section order of the note's kind (templates in `docs/templates/`).
  - `<Derivation>` holds optional depth. The note must read completely with it collapsed.
- **Derive, don't just state.** The note's central result is derived in the main text, not hidden in a collapsed
  block: why the dot product equals $\norm{\xvec}\norm{\yvec}\cos\theta$, where the Taylor coefficients come from, why a
  statistic has its null distribution. Secondary results (moments, identities, special cases) get a proof in a
  `<Derivation>`. Arguments must not be circular, e.g. Cauchy–Schwarz cannot be proved from $\abs{\cos\theta} \le 1$
  when the angle is defined through it. When a note is mined from a source that proves a result, the note proves it
  too.
- **Tag worked examples.** Every note that contains a worked example carries the tag `worked-example`, including
  every note of kind `example`.
- **Maths uses the shared macros** in `content/macros.ts`, the site's `definitions.sty`: `\xvec`, `\Xmat`, `\muvec`
  (or `\mub`), `\Sigmamat`, `\Dcal`, `\reals`, `\expect`, `\Gauss`, `\norm{…}`, `\argmin`, `\KL`, and so on.
  Never spell out `\mathbf{x}` or `\boldsymbol{\mu}` in a note. Add a missing macro to the right group in
  `macros.ts` rather than using `\newcommand` in a note. A macro that cannot render on its own (such as `\LP`) needs an entry in its
  group's `examples`; the content check renders every macro's example. The `/notation` page lists them all. Every formula is
  rendered with the macros by `npm run check:content`, so an unknown command fails the build with the file name.
- **Cite claims that are not common knowledge** with `<Cite id="key" />`. Every source goes in
  `content/references.yaml` with a real URL. Papers, blog posts, books, docs and videos are all valid. Cite the
  primary source for definitions and results.
- Placeholder text is allowed only in notes with `status: stub`.

## Content model

Frontmatter (validated; see `site/src/lib/content-schema.ts`): `title`, `kind`, `summary`, `tags`,
`aliases`, `requires`, `partOf`, `related`, `code`, `references`, `status` (`stub | draft | stable`), `updated`.

- **Kinds** have their own section structure (`docs/templates/<kind>.mdx`):
  - `concept`: one idea.
  - `technique`: an engineering trick, e.g. KV caching.
  - `test`: a statistical test.
  - `distribution`: a named probability distribution, with the shared explorer.
  - `example`: a worked problem that illustrates several ideas (e.g. the occasionally dishonest casino), filed under an
    `examples` category. Examples tied to one technique stay inside that technique's note.
  - `case-study`: compares specific models, with a `<SpecTable>`.
  - `overview`: a hub. Its components list themselves via `partOf`.
- **Category** is the note's folder path, a path in `content/taxonomy.yaml` of any depth (e.g.
  `probability-distributions/discrete`). Add a level only when a subtopic has several notes that form a chapter;
  `make doctor` flags single-note branches. Top-level topics are subjects, ordered from mathematical prerequisites to
  applications; subtopics are a subject's natural chapters. Place a note by its subject, not by where it is used:
  Bayes' theorem is probability theory even though every model uses it. Categories are for browsing; one note has one
  category, and cross-cutting links are relations and tags. Lists follow taxonomy order (`categoryOrder`), never
  alphabetical order.
- **Relations** are typed: `requires` (prerequisites), `partOf` (component of a larger system), `related` (see also).
  Backlinks and component lists are computed; declare each relation only on one side.
- **Tags** are cross-cutting kebab-case labels.

## Git

- Never stage, commit or push unless the user expressly asks for it in the current message. Permission is atomic: one
  request covers one action (one commit, or one push) and does not carry over to later work. "Commit" does not imply
  "push", and making a repository does not imply pushing to it.
- `docs/field-notes-survey.md` is a volatile local planning file, excluded via `.git/info/exclude`. Never add it.

## Deployment

- `base: '/ai-field-notes/'` in `vite.config.ts` is the only place the path is set. The router and generated-asset URLs
  derive it from `import.meta.env.BASE_URL`.
- The build copies `index.html` to `404.html` so that GitHub Pages serves deep links to the SPA.
- `.github/workflows/deploy.yml` runs `make check` and `npm run build`, then publishes `dist/` to Pages.

## Gotchas

- The npm project lives at the repo root, not in `site/`. Vite's `root` is `site/`, but `content/` files import
  `react` and must resolve the root `node_modules`.
- pytest skips directories named `build` by default. That is why there is no `python/tests/mlc/build/`; the registry
  is checked by `mlc check` instead.
- `json-schema-to-typescript` would rename `PointCloud2d` to `PointCloud2D`; `scripts/gen-types.ts` restores the
  pydantic names.
- `katex` is pinned to the exact version that `rehype-katex` renders with (`npm ls katex` must show one deduped
  copy). The stylesheet comes from the top-level package and the HTML from rehype-katex's; if they differ, class
  names drift and sub- and superscripts render at full size.
- `EChart` calls `setOption` without `lazyUpdate`. A lazy update leaves the chart without coordinate systems until the
  next frame, so converting a pointer position to data coordinates (handles, `onPlotClick`) fails mid-drag.
- Render pages headlessly only with `npm run check:render -- <slug> ...` (dev server on port 5180; `--port` to change).
  It runs one page at a time and kills Chrome when the DOM is dumped or after 30 s. Never hand-roll a headless Chrome
  loop: `--dump-dom` often leaves Chrome running, and parallel browsers overload the dev server.
- Handle dragging uses DOM pointer events with pointer capture, not zrender's events. Listeners are removed on
  unmount; StrictMode mounts twice on the same element.
