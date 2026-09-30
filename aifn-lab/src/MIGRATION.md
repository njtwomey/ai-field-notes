# Migrating specimens and views to lab UX v2

The lab depends only on `aifn/<module>` and third-party packages. Until `src/specimens` and `src/views` are migrated,
their old imports resolve to **lab-owned** stand-ins (nothing comes from `site/`):

| Old path                                | Temporarily served by                           |
| --------------------------------------- | ----------------------------------------------- |
| `@/components/viz`                      | `src/legacy/viz.ts` (+ `legacy/components.tsx`) |
| `@/components/ui/<x>`                   | `src/ui/<x>` (the lab's own shadcn primitives)  |
| `@/lib/utils`                           | `src/lib/utils`                                 |
| `@lab/controls` `LabChart`, `LabSlider` | `src/legacy/lab-controls.tsx`                   |

Migration is mechanical: rewrite the imports below, wrap each figure in `Figure`, drop chart heights. When no file
imports `@/…`, `LabChart` or `LabSlider`, delete `src/legacy/`, the `@/…` aliases in `aifn-lab/vite.config.ts` and
`tsconfig.lab.json`, and the `LabChart`/`LabSlider` line in `src/controls/index.ts`. Then `make lab-check`.

Check your work with `make lab-shots ARGS='--only …'` and read the PNGs (`.scratch/lab-shots/`, see `aifn-lab/README.md`).

## Import map

| Old import                                                                  | v2 import                                                                                  | Notes                                                                                                                         |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `@/components/viz` `XYChart`, `XYSeries`, `Segment`                         | `@lab/viz` same names                                                                      | `x`/`y` are `readonly number[]`; new `type: 'area'`, `xLog`, `showPoints`, `hoverGroup`, `zoom`, `formatX/formatY`.           |
| `@lab/controls` `LabChart`                                                  | `@lab/viz` `XYChart` inside a `@lab/layout` `Figure`                                       | Drop `height`, `width`, `maxWidth`, `maxHeight`: the Figure owns the size. Zoom toolbar is built in (`zoom={false}` to hide). |
| `@/components/viz` `Heatmap`, `HeatmapOverlay`                              | `@lab/viz` same names                                                                      | Zoomable; axes span the grid exactly; `equalAspect` sizes the plot area to the grid's aspect, centred (never pads the axes).  |
| `@/components/viz` `Readout`                                                | `@lab/viz` `Readout` (+ `Readouts` row)                                                    | Optional `color` dot. Put readouts in `Figure readouts={…}`.                                                                  |
| `@/components/viz` `Handle`, `Vec2`, `Vector`, `PlotPointer`                | `@lab/viz` same names                                                                      | Unchanged semantics.                                                                                                          |
| `@/components/viz` `formatNumber`, `useScaleColor`                          | `@lab/viz` same names                                                                      | Also `niceStep`, `snapToStep`, `formatPower`.                                                                                 |
| `@/components/viz` `seriesColor`, `sequential`, `diverging`                 | `@lab/design` (or `@lab/design/palette`)                                                   | Plus `chrome`, `categorical`, `interpolateColors`, `scaleStops`.                                                              |
| `@lab/controls` `LabSlider`, `@/components/viz` `ParamSlider`               | `@lab/controls` `Slider`                                                                   | Steppable by default (`steppable={false}` to opt out; `withArrows` still accepted). Omit `step` for an automatic 1-2-5 step.  |
| `@/components/viz` `useParam`, `Param`, `ParamSpec`                         | `@lab/controls` same names                                                                 | `step` optional (automatic).                                                                                                  |
| `@/components/viz` `ParamChoice` with `options`                             | `@lab/controls` `Select` (≤ 12 options) or `Combobox` (long lists); `Choice` picks for you | Same `label`/`value`/`onChange`/`options`; options may also be plain strings.                                                 |
| `@/components/viz` `ParamSwitch`                                            | `@lab/controls` `Switch`                                                                   | Same props.                                                                                                                   |
| `@/components/viz` `ParamButton`                                            | `@lab/controls` `Button variant="outline" size="sm"`                                       | Group related buttons in `ButtonGroup`.                                                                                       |
| `@/components/viz` `StepControls`                                           | `@lab/controls` `StepControls`                                                             | Same props.                                                                                                                   |
| `@/components/viz` `Interactive`                                            | `@lab/layout` `Figure`                                                                     | `readout` → `readouts`; new `description`, `data`, `defaultSize`.                                                             |
| Hand-rolled first/prev/play/next/last + slider + speed (TraceView scrubber) | `@lab/controls` `Player`                                                                   | `value`, `onChange`, `count`, `format`, `loop`.                                                                               |
| Several stacked charts of one trace                                         | same charts with one `hoverGroup` inside `ChartSize scale={…}`                             | Linked hover and relative panel heights.                                                                                      |
| `@/components/ui/badge`, `button`, `table`, …                               | `@lab/ui/badge`, `@lab/ui/button`, `@lab/ui/table`, …                                      | Same shadcn (base-nova) components, installed into the lab.                                                                   |
| `@/lib/utils` `cn`                                                          | `@lab/lib/utils` `cn`                                                                      |                                                                                                                               |
| `@/components/theme-provider`                                               | `@lab/design` `ThemeProvider`, `useTheme`; app root uses `@lab/layout` `Providers`         |                                                                                                                               |

## What migrated figures get for free

- **Hover on line charts.** Any `XYChart` with a line, area or bar series uses an axis-triggered tooltip: a pointer line
  follows the cursor, each series' nearest point is marked, and the tooltip lists every series' value at that x
  (formatted with `formatNumber`, or `formatX`/`formatY`). Pure scatter charts keep item hover; heatmaps show the cell's
  x, y and value. Handles win: no tooltip while a handle is dragged. Inside a `Figure`, the same values appear as a
  hover readout under the chart. No code needed.
- **Zoom and pan** on every `XYChart` and `Heatmap`: per-axis pan/zoom groups with typed ranges, a both-axes group with
  fit-to-data, and pinch or Ctrl/⌘-scroll about the pointer; log axes zoom in log space.
- **Size** from the `Figure`: presets S/M/L/XL/full width and a drag corner, remembered per specimen. Charts fill the
  height they are given; use `ChartSize scale={0.5}` for a panel at half the frame's height.
- **Data export:** the Figure's copy button puts every chart's data on the clipboard as JSON (or pass `data`).

## Example

```tsx
// Before
import { LabChart, LabSlider } from '@lab/controls'
import { ParamChoice, Readout } from '@/components/viz'
<LabSlider label="n" value={n} onChange={setN} min={1} max={12} step={1} withArrows />
<ParamChoice label="method" value={m} onChange={setM} options={methods.map((v) => ({ value: v, label: v }))} />
<LabChart series={series} xLabel="q" height={280} />
<Readout label="median" value={median} />

// After
import { Select, Slider } from '@lab/controls'
import { Figure } from '@lab/layout'
import { Readout, XYChart } from '@lab/viz'
<Figure
  title="Quantiles by method"
  controls={<><Slider label="n" value={n} onChange={setN} min={1} max={12} step={1} /><Select label="method" value={m} onChange={setM} options={methods} /></>}
  readouts={<Readout label="median" value={median} />}
>
  <XYChart series={series} xLabel="q" />
</Figure>
```

## Laying out controls

Controls go in a `Figure`'s `controls` slot or in `<Controls>` from `@lab/layout`: a responsive grid (columns at least
14rem) where labels and tracks line up. Do not put controls in ad-hoc `flex` rows. Outside a grid, `Slider` still has a
usable intrinsic width (`w-full min-w-48 max-w-sm`), and `NumberField`, `Select` and `Combobox` likewise
(`min-w-40 max-w-xs`); the step arrows are `icon-sm` click targets.

## Declarative and conditional controls

For a figure that offers several functions (models, methods), each with its own parameters, declare them once:

```tsx
import { choice, defineVariants, slider, toggle, useVariants, VariantControls } from '@lab/controls'

const FUNCTIONS = defineVariants(
  {
    quadratic: { label: 'Quadratic', params: { a: slider(-2, 2, 1), b: slider(-2, 2, 0) }, f: (x: Value, p) => … },
    softplus: {
      label: 'Softplus',
      params: {
        beta: slider(0.1, 5, 1),
        shape: choice(['plain', 'shifted']),
        shift: slider(-2, 2, 1, { when: (p) => p.shape === 'shifted' }), // shown only for 'shifted'
      },
      f: (x: Value, p) => …,
    },
  },
  { amplitude: slider(0.2, 2, 1) }, // optional shared params: every variant has them, with one value
)

const v = useVariants(FUNCTIONS) // v.key, v.params (narrowed by v.key), v.f(x) (bound), v.set, v.state
<Figure controls={<VariantControls variants={v} />}>…</Figure>
```

- **Kinds:** `slider(min, max, initial, { step?, label?, when?, steppable? })`, `number(initial, { min?, max?, step? })`,
  `choice(options, initial?, { searchable? })`, `toggle(initial)`. Values are clamped and snapped by their definition.
- **Typed:** `v.params` is a union narrowed by `v.key`; each `f` receives its own variant's values plus the shared ones.
  All variants share one `f` signature: annotate `x` and put `params` before `f`.
- **Remembered:** each variant keeps its own values when the reader switches away and back.
- **Serialisable:** `v.state` is plain JSON `{ key, values, shared }`; `v.setState(json)` restores it (invalid entries
  fall back to their initial values), and `useVariants(FUNCTIONS, { state })` starts from it.
- **Without variants:** `const p = useParams({ … })` gives typed `p.values`, `p.set` and JSON `p.state`;
  `<ParamControls {...p} />` renders the controls, honouring `when`.

## Multi-panel figures: `Subplots`

Figures with several charts use `Subplots` and `Panel` from `@lab/viz`, not stacked charts with `ChartSize`:

```tsx
<Subplots rows={2} sharex heightRatios={[2, 1]} hoverGroup>
  <Panel><XYChart series={f} equalAspect … /></Panel>
  <Panel><XYChart series={derivatives} … /></Panel>
</Subplots>
```

- `rows`, `cols`; `sharex` / `sharey`: `true` (all panels), `'row'` or `'col'`. `heightRatios` and `widthRatios` split
  the Figure's frame, so the size presets and the drag corner still apply. `hoverGroup`: `true` links hover across
  the grid, or a name links it with charts elsewhere.
- Plot areas line up: every panel in a column gets the same left and right margins, measured from the column's widest
  y tick labels, and every panel in a row the same top and bottom margins.
- A shared axis has one range: the union of its panels' data (or explicit) ranges, rounded to whole ticks. Zoom, pan,
  typed ranges and fit on any panel move every panel that shares it, and it holds still while a handle is dragged on
  any panel. With x shared only the bottom row shows x ticks; with y shared only the left column shows y ticks.
- Axes fit the data by default (`aspect="fit"`): each chart fits its own non-live series and vector ends, rounded to
  whole ticks; live series never move the axes. `aspect="equal"` (formerly `equalAspect`) only ever widens an axis,
  so it never hides data. In a panel whose x is shared, x keeps the shared range and y takes the same units per pixel
  from the panel's height, if that range still holds the data; otherwise the panel keeps fitted axes and its toolbar
  says "units not equal". With only y shared, x is derived instead; with both shared, units are not equal.
- Controls: a shared axis has one set of controls (x on the bottom panel of its column, y on the left panel of its
  row); every other axis has its own. Each set has the range fields, pan and ±; with equal units, ± on one axis
  zooms both. The grid's toolbar has one "Fit all". The range fields always show the drawn ranges.
- `XYChart` panels align; a `Heatmap` in a panel takes the row's height but does not join the alignment yet.

### Fixed unit-square plots

A chart with `aspect="equal"` and both `xRange` and `yRange` given in full keeps those ranges exactly: the plot area
takes their aspect and is centred in the chart's box (a Heatmap's `equalAspect` does the same). ROC, PR, gain, cost,
PRG and reliability curves use this on [0, 1]², so their plot areas are square and the diagonal is at 45°. They have
no zoom toolbar (`zoom={false}`): the ranges are fixed. `CurveChart` (from `@lab/views`) is the frameless chart of a
typed curve for dashboards, with an optional operating-point handle.

A column with a strip under a square (the reliability diagram's bin counts):
`<Subplots rows={2} sharex heightRatios={[1, 0.22]} ratiosOf="equal" toolbar={false} tight>`. `ratiosOf="equal"`
sizes the other rows from the equal panel's plot height; `toolbar={false}` drops the fit-all row; `tight` puts panels
sharing x nearly edge to edge.

## Dashboards: mixed panels in one figure

A figure of charts, tables and views that share no axes (a classifier's feature space, score histograms, contingency
table, ROC and PR) splits the `Figure`'s chart area with `Dashboard` from `@lab/layout`:

```tsx
<Dashboard>
  <DashboardRow ratio={1.15} minHeight={300}>
    <DashboardCell ratio={1.2}><Heatmap … /></DashboardCell>
    <DashboardCell><XYChart … /></DashboardCell>
  </DashboardRow>
  <DashboardRow minHeight={300}>
    <DashboardCell><ContingencyTableView table={cm} positive={0} /></DashboardCell>
    <DashboardCell aspect="square"><CurveChart curve={roc} handles={…} /></DashboardCell>
    <DashboardCell aspect="square"><CurveChart curve={pr} handles={…} /></DashboardCell>
  </DashboardRow>
</Dashboard>
```

- Rows share the frame's height by `ratio`, each at least `minHeight` (the figure grows rather than squash a row), so
  the size presets and the drag corner still apply.
- Cells share the row's width by `ratio`. A cell with `aspect` (`"square"` or width / height) takes the row's height
  and a width that follows from it, so a square chart is never stretched; the other cells share the rest.
- Every cell tells the charts inside how tall to be. Tables and views fill the cell (`h-full`).
- Narrower than `stackBelow` (default 640 px, e.g. a phone), every cell stacks at full width: plain cells keep their
  row's height, aspect cells take their height from the width, and so does a cell with `stackAspect` (e.g. a
  feature-space plot that would otherwise be a short strip on a phone).
- Aligned axes are a `Subplots` concern; a `Subplots` grid may sit in a cell.

`ContingencyTableView` (from `@lab/views`) draws an `aifn/metrics` `ConfusionMatrix` (K classes) with full margins
from `confusionMargins`: row totals with recall and miss rate (TPR/FNR, TNR/FPR for two classes with `positive`),
column totals with precision and FDR (PPV/FDR, NPV/FOR), and the grand total with prevalence and accuracy. Hovering a
rate outlines the cells it divides.

## URLs and figure anchors

The lab uses path URLs: `/<module>/<specimen-slug>` for a specimen (e.g.
`http://localhost:5190/autodiff/a-function-and-its-derivatives`), `/ui-kit` for the UI kit. Old `#/…` links redirect.
The hash names a figure: `…/a-function-and-its-derivatives#f-with-its-tangent-and-osculating-circle-and-f-f-from-grad`.
A `Figure`'s id is the github-slugger slug of its title (as on the site), made unique within the page with `-2`, `-3`;
pass `id` to fix it. Hovering a title shows a link icon that sets the hash and copies the URL; opening a URL with a hash
scrolls to that figure and highlights it. `make lab-check` fails on duplicate specimen paths or figure ids in a page.

## Held axes for parameter exploration

A figure that explores a parameter holds its axes, so a change of parameter reads as a change of shape rather than of
scale. By default charts refit whenever their data changes; parameter-exploration figures pass
`rescaleOnChange={false}`:

- `XYChart`, `Heatmap` (its colour scale), `Subplots` (every chart in the grid) and `Panel` (one cell) take
  `rescaleOnChange`, `axisKey` and `holdFit`.
- Held axes are fitted once, on mount, and kept while the data changes. `axisKey` refits when it changes (e.g. the
  family or dataset, not a parameter). The fit button refits on demand; zoom and pan work as usual; handle drags still
  freeze the axes.
- `holdFit="initial"` (the default when held) keeps the first fit; `holdFit="union"` grows the range to include new
  data but never shrinks it, e.g. for a density that grows taller as its scale shrinks.
- To hold chosen ranges instead, pass `xRange`/`yRange` (for distributions, the range at the family's default
  parameters: `distributionRange(family.make(...defaults))` from `@lab/views`).
- `DistributionView` takes `rescaleOnChange` and `axisKey`: x holds `range`, the density axis uses `union`, the cdf
  stays on [0, 1].

### Equal units in a column: `<Panel aspect="equal">`

In a single-column `Subplots`, `<Panel aspect="equal">` sizes that panel so its units are equal with both axes on
their fitted ranges: plot height = plot width × (y span / x span). The other panels share what remains of the frame by
their `heightRatios` (at least 160 px each), and the figure grows or shrinks to the sum. If the total would pass about
85% of the window's height, the column's plot width narrows, centred, rather than changing ranges or units. Zoom and
pan keep equal units (± on one axis zooms both); the size presets and the drag corner recompute the height. The sizing
is `layoutColumn` in `viz/subplot-layout.ts`, tested in `subplot-layout.test.ts`
(`npx vitest run --config aifn-lab/vite.config.ts src/viz/subplot-layout.test.ts` from `aifn-lab/`).

## Players open at step 0

Every `Player` opens at position 0. A non-zero initial value is reset to 0 on mount unless the figure passes
`startReason` (why this walk-through should open later). Anything that walks through steps or time uses a `Player`,
not a plain `Slider` (DESIGN.md §2, rule 8).
