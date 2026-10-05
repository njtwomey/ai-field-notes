# Lab UX v2: migration record

The lab was migrated to UX v2 (DESIGN.md) in phases 5a–5d (2026-09-30 to 2026-10-01). Every specimen and view now uses
`Plot` + layers, `useFigureState`, `Figure` + panels; `XYChart`, `Heatmap`, `Subplots`/`Panel`, `ChartSize`, the
legacy `XView` figure wrappers and the `@/…` stand-ins are deleted, and no figure uses `useParam`, `useVariants` or
`useParams` (state and controls are phase 5d-2's). This file
records what replaced what and the conventions a new figure follows; DESIGN.md has the reasons.

Check work with `make lab-check` and `make lab-shots ARGS='--only …'` and read the PNGs (`.scratch/lab-shots/`, see
`lab/README.md`).

## Plot v2: axis models and layers (`Plot`, `Plots`, `useAxis`)

The chart core lives in `src/viz/plot/` and is exported from `aifn-render/viz`. `EChart.tsx` is the only file that touches
ECharts. Every layer has a review page at `/ui-kit/<layer>`, and `/ui-kit/plots` shows grids.

```tsx
import { Curve, Density, Handle, Histogram, Plot, Plots, SupportBand, supportOf, useAxis } from 'aifn-render/viz'

const x = useAxis({ label: 'x', support: supportOf(X) }) // one model per axis
const y = useAxis({ label: 'density', hold: 'union' })
<Plot x={x} y={y}>
  <Histogram values={draws} />
  <Density dist={X} emphasis />
  <SupportBand dist={X} />
  <Handle kind="x" at={x0} onDrag={setX0} label="x₀" />     // a natural place only; never a distribution parameter
</Plot>
```

### Axes: `useAxis(options)`

One `AxisModel` per axis, created once per component. Plots that receive the same model **share** that axis: one range
(the union of their layers' extents), one zoom and pan, one toolbar button, and the drag freeze.

| Option                       | Meaning                                                                                                         |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------- |
| (none)                       | **fit**: the layers' extents, rounded out to whole ticks, refitted whenever the data change                     |
| `hold: 'initial' \| 'union'` | **hold**: keep the first fit, or grow it to take in new data (never shrink); `key` refits                       |
| `range: [lo, hi]`            | **fixed** (either end may be `undefined` to fit that end); still zoomable                                       |
| `support: interval`          | **support-bound**: bounded ends are hard limits (axis ends there, zoom and pan stay inside); unbounded ends fit |
| `key`                        | refit a held axis when it changes (a family, a dataset; not a parameter)                                        |
| `log`                        | logarithmic                                                                                                     |
| `categories: string[]`       | categorical: category k at position k, labelled by name, range [−0.5, K − 0.5], no zoom                         |
| `equal: otherAxis`           | equal units with that axis in every Plot showing both; zooming one zooms both                                   |
| `label`, `format`, `nice`    | name, tick and tooltip format, whether a fit rounds to ticks (default true)                                     |

Equal units never pad the axes: the plot area takes the ranges' aspect and is centred (a unit square is square). A
raster's extent is tight: an axis it sets ends at the grid's edges, not at the next tick.

### `Plot` and `Plots`

- `<Plot x y title? legend? hoverGroup? toolbar? bare? scale? onPointer? onPlotClick? onBrush?>` holds layers in drawing
  order. Hover (axis tooltip and the Figure's readout), the legend (shown for two or more names, live layers included;
  wrapped onto more rows, never paged, and kept clear of a colour bar's name) and palette slots (in layer order,
  skipping `muted`/`emphasis`; pass `slot` to fix one) belong to the Plot. On its own a Plot draws the compact toolbar
  (↔ ↕ popovers and auto-scale). `scale={0.5}` takes half the frame's height (it replaced `ChartSize`).
- Above the plot area, from the top: the legend rows, the `title`, and a row for labels drawn above the plot (an x
  handle's or probe's, a vertical annotation's). A horizontal guide's label sits inside the plot above its right end,
  so no label is clipped at the plot's edge.
- `<Plots rows cols heights widths hoverGroup ratiosOf tight toolbar>` is a grid of Plots. Sharing comes from passing the
  same axis model, not from `sharex`/`sharey`. Tick labels show on the outer panel of a shared axis. Every panel in a
  column has the same left and right edges and every panel in a row the same top and bottom, whatever it draws (a
  raster's colour bar widens its whole column). An equal-units panel in a single column sizes its row; when every
  panel of a column, or of a row, has equal units, they share one plot width, so panels with matching ranges are the
  same size whatever their tick labels. `ratiosOf="equal"` sizes the other rows from the equal panel's plot height;
  `tight` puts panels sharing x nearly edge to edge. One toolbar lists every axis once.
- A non-Plot child (`<div />`) leaves a cell empty.

### Layers

| Layer                                  | Draws                                                                                      |
| -------------------------------------- | ------------------------------------------------------------------------------------------ |
| `Curve x y`                            | a line (`dashed`, `thin`, `width`, `showPoints`, `silent`)                                 |
| `Points x y`                           | scatter; `group` (class → slot and shape), `shape` per point, `colors`, `dense`, `size`    |
| `Bars x y`                             | data-unit rectangles; `edges` (touching), `width`, `base`, `orient`, `colors`              |
| `Area x y base?`                       | filled region to 0, a constant or a second curve; `orient`                                 |
| `SignedArea x y label?`                | positive and negative parts in the diverging scale, net area written in the plot           |
| `Histogram values bins? normalize?`    | aifn histogram (Freedman–Diaconis by default), outlined bars; `orient`                     |
| `Density dist range? fill?`            | an aifn distribution's density; default range its 0.002–0.998 quantiles inside the support |
| `Mass dist range?`                     | a discrete distribution's mass, a bar per integer                                          |
| `Rug values`                           | ticks along the bottom (or left, `orient="y"`) edge                                        |
| `SupportBand dist \| interval`         | the support along the axis (closed ● / open ○ ends), the outside shaded                    |
| `Vectors vectors`                      | arrows, clipped at the plot edge with a chevron                                            |
| `Segments segments`                    | thin muted segments (residuals)                                                            |
| `Raster x y z scale range? colorAxis?` | the heatmap: one cached canvas image, a hit layer for hover, a colour bar                  |
| `Contours x y z levels`                | marching-squares level sets                                                                |
| `Handle kind at onDrag`                | a draggable point or guide line                                                            |
| `Annotation at \| x \| y text`         | a labelled point, vertical or horizontal line                                              |

Every layer takes `name`, `slot`, `emphasis`, `muted`, `color`, `id`, `stale` and **`live`**: a live layer is sent to
ECharts as a patch (nothing else is redrawn) and never moves the axes; it keeps its legend entry and its hover row. To
keep a chart whose data change on every move (a trace recomputed while dragging) on patches, make its layers live and
fix its axes to whole ticks over the data (`useAxis({ range: niceRange(extent) })`), as `TracePanel` does. A layer is rebuilt only when its props change (shallow), so
keep arrays memoised. `Raster scale="diverging"` is symmetric about zero by default (pale midpoint, saturated ends);
`colorAxis={useAxis({ hold: 'initial' })}` holds the colour scale. New layers: `defineLayer({ kind, extent, build })`
(see `viz/plot/layer.ts`).

### What replaced `XYChart`, `Heatmap`, `Subplots`, `ChartSize`

| Old                                                             | New                                                                                 |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `<XYChart series={[…]} xLabel yLabel />`                        | `<Plot x={useAxis({ label })} y={useAxis({ label })}>` with one layer per series    |
| series `type: 'line'` / `'scatter'` / `'bar'` / `'area'`        | `Curve` / `Points` / `Bars` / `Area`                                                |
| series `histogram: true` (bars from precomputed bins)           | `Histogram values={samples}` (or `Bars edges={…}`)                                  |
| `live={[…]}`                                                    | the same layers with `live`                                                         |
| `xRange`, `yRange`                                              | `useAxis({ range })`                                                                |
| `xLog`, `yLog`                                                  | `useAxis({ log: true })`                                                            |
| `integerX`                                                      | `useAxis({ categories })` (or a fixed `range` at half-integers)                     |
| `aspect="equal"` / `equalAspect`                                | `useAxis({ equal: x })` on y                                                        |
| `rescaleOnChange={false}`, `holdFit`, `axisKey`                 | `useAxis({ hold: 'initial' \| 'union', key })`                                      |
| `segments`, `vectors`, `handles`                                | `Segments`, `Vectors`, `Handle` layers                                              |
| `zoom={false}`                                                  | `toolbar={false}` on the Plot, or `zoom: false` on the axis                         |
| `<Heatmap x y z scale range contours overlay marker vectors />` | `Raster` + `Contours` + `Points`/`Curve` (live for moving ones) + `Vectors`         |
| Heatmap `equalAspect`                                           | `useAxis({ equal: x })`; the raster's extent is tight, so nothing is padded         |
| Heatmap `rescaleOnChange={false}` (held colour scale)           | `Raster colorAxis={useAxis({ hold: 'initial', key })}`                              |
| `onCellClick`                                                   | a point `Handle` (the conditioning query), or `onPlotClick`                         |
| `<Subplots sharex sharey heightRatios widthRatios><Panel>…`     | `<Plots heights widths>` with the same axis model passed to the Plots that share it |
| `Panel aspect="equal"`                                          | an equal-linked axis pair in that Plot                                              |
| `Panel share={{ y: false }}`                                    | give that Plot its own axis model                                                   |
| `hoverGroup`                                                    | `hoverGroup` on `Plots` or `Plot` (unchanged)                                       |
| `<ChartSize scale={0.5}>`                                       | `scale={0.5}` on the `Plot` or `Plots`                                              |
| `XYSeries`, `HeatmapOverlay` (overlay props of views)           | layer elements passed as `children` (or a `ReactNode` prop)                         |

Log axes tick every decade, or every few decades when a label per decade would crowd.

## Dashboards: mixed panels in one figure

A figure of charts, tables and views that share no axes splits the `Figure`'s chart area with `Dashboard`,
`DashboardRow` and `DashboardCell` from `@lab/layout`. Rows share the frame's height by `ratio` (each at least
`minHeight`); cells share the row's width by `ratio`. A cell with `aspect` (`"square"` or width / height) takes the
row's height and a width that follows from it; when aspect cells would leave a flexible cell narrower than its
`minWidth` (default 160 px), the whole row shrinks, so nothing is letterboxed. Below `stackBelow` (640 px) cells stack;
`stackAspect` gives a stacked plain cell a shape. Aligned axes are a `Plots` concern; a `Plots` grid may sit in a cell.

## Laying out controls

Controls go in a `Figure`'s `controls` slot or in `<Controls>` from `@lab/layout`: a responsive grid (columns at least
14rem) where labels and tracks line up. Do not put controls in ad-hoc `flex` rows. Outside a grid, `Slider` still has a
usable intrinsic width (`w-full min-w-48 max-w-sm`), and `NumberField`, `Select` and `Combobox` likewise
(`min-w-40 max-w-xs`); the step arrows are `icon-sm` click targets.

## Players open at step 0

Every `Player` opens at position 0. A non-zero initial value is reset to 0 on mount unless the figure passes
`startReason` (why this walk-through should open later). Anything that walks through steps or time uses a `Player`.

## URLs and figure anchors

The lab uses path URLs: `/<module>/<specimen-slug>` for a specimen (e.g.
`http://localhost:5190/autodiff/a-function-and-its-derivatives`), `/ui-kit` for the UI kit. Old `#/…` links redirect.
The hash names a figure: `…/a-function-and-its-derivatives#f-with-its-tangent-and-osculating-circle-and-f-f-from-grad`.
A `Figure`'s id is the github-slugger slug of its title (as on the site), made unique within the page with `-2`, `-3`;
pass `id` to fix it. Hovering a title shows a link icon that sets the hash and copies the URL; opening a URL with a hash
scrolls to that figure and highlights it. `make lab-check` fails on duplicate specimen paths or figure ids in a page.

## Figure state: `useFigureState` (DESIGN.md §4)

One declaration per figure gives the control rows, typed values, handles, reset and URL state. Import from
`aifn-render/state`. The data half of every field is an aifn `Space` dimension (`toSpace(schema)`); `fromSpace(space)` turns a
registry's `Space` into fields.

```tsx
const state = useFigureState({
  input: variants({ normal: { mu: slider(-3, 3, 0), sd: slider(0.1, 3, 1) }, gamma: { shape: slider(0.5, 8, 2) } },
                  { label: '1 · input', choiceLabel: 'family' }),
  reveal: row('2 · reveal', { jacobian: toggle(false, 'Jacobian scaling'), draws: choice([1000, 5000]) }),
  x0: slider(-4, 4, 0.5, { onChart: true }),         // no row: moved by its handle
  logY: setting(false, 'log density'),
})
state.input.key; state.input.values.mu              // narrowed by key
state.reveal.jacobian; state.x0
<Figure title=… purpose=… state={state}>             // rows, reset button, URL
  <Plot …><Handle {...state.handle('x0')} /></Plot>  // or state.handle(['sx', 'sy']) for a point
</Figure>
```

- **Rows:** each `row` and each `variants` field is one labelled row, in declaration order; consecutive plain fields share
  an unlabelled row; `onChart` fields draw none. Hand-placed `controls` still render after the rows.
- **Setting:** `state.set(path, v)` with `x0`, `reveal.jacobian`, `input` (a case name) or `input.mu` (the chosen case's
  parameter); clamped and snapped. `state.bind(path)` gives a `Param` for a `Slider`, `useProbe` or anything else.
- **Conditions:** `when: when('shape', 'shifted')` (data, serialisable, honoured by the URL decoder) or a closure
  (presentation only).
- **Reset and JSON:** `state.reset()`, `state.isDefault`, `state.json` / `state.setJson(json)`,
  `useFigureState(schema, { initial: json })`.
- **URL:** the Figure attaches the state to its anchor: non-default values become `?<figure-id>.<path>=<value>` (several
  figures per page keep their own), decoded through `clampReport` on load. Unknown, inactive or unparseable keys are in
  `state.dropped` and warned about in dev. A link therefore reproduces a view.
- **Inline schemas are fine:** values are re-validated against the latest definitions every render (a range that depends
  on the data stays valid), and unchanged fields keep their identity, so memos on `state.input.values` do not rerun when
  `x0` moves.
- Reserved field names: `values schema set handle bind reset isDefault json setJson dropped attach`.

## `Figure`: purpose, equation band, grouped readouts

- `purpose` (one line under the title) is required, in the type and by `make lab-check`.
- `equation={<Equation>…</Equation>}` draws the equation band between the controls and the charts, only when given.
- `readouts={{ 'at x₀': <>…</>, totals: <>…</> }}` draws labelled groups; a plain node is still one row.
- `make lab-check` also fails a Figure inside another Figure's chart area (a view must be a panel).

## Probes (DESIGN.md §6)

```tsx
const probe = useProbe({ x: state.bind('x0'), label: 'x₀' })            // { x, y, yLabel } for a point
<Plot …><Curve … /><Probe probe={probe} at={f(probe.x!)} /></Plot>          // in every Plot showing x
<Plot …><Probe probe={probe} orient="y" /></Plot>                           // x along this plot's y axis
<ProbeReadout probe={probe} values={{ 'f(x₀)': f(probe.x!) }} />
```

The `Probe` layer is live (a patch, never widens the axes) and draws the guide (or point) as a handle: dragging it on any
plot moves it on all. On a raster a point probe is click-to-set (pressing anywhere moves the only handle). Replace per-page
handle and `onPlotClick` wiring with a probe.

## Derived computation: `useComputed` (DESIGN.md §8a)

```tsx
const path = useComputed(() => ascent(start, steps), [start, steps], { mode: 'frame' }) // or 'release'
<Curve x={path.value.x} y={path.value.y} stale={path.stale} live />
```

- Latest wins, at most once per frame; the first value is computed on mount (server render included).
- Under the 8 ms budget it runs in the frame and renders before paint. Over it, during a drag, it runs after the paint
  with as many free frames as it took, and `stale` is true meanwhile: pass it to the derived layers (`stale` on any layer
  fades it). It always runs on release.
- `mode: 'release'` waits for release while a pointer is held. `mode: 'worker'` is a stub (behaves as `release`) until
  aifn's algorithm registry can name the computation for a worker.
- `Slider`'s `debounceMs` is gone: sliders and handles feed the same scheduler. Do not throttle in widgets.
- Measured on `ui-kit/scheduler` (100 000-step path, 17 ms a run; `--profile`, drag across the raster): useMemo
  input-to-paint median 85.5 ms, p95 122, 35/95 frames dropped; `frame` 38.8 / 83, 7/67; `release` 13.4 / 32, 0/60.

## Equations (DESIGN.md §7)

```tsx
import { Equation, EquationSteps, live, tex } from '@lab/layout'
<Equation>{tex`p(${x0}) = \frac{1}{${sigma}\sqrt{2\pi}} \cdots = ${live(p, { digits: 4, strong: true })}`}</Equation>
<EquationSteps steps={[{ tex: tex`u = x^2 = ${u}`, note: 'Forward' }, …]} step={step} onStep={setStep} />
```

Slots are numbers (lab number format), TeX strings or `live(value, { digits, strong })`, set off by a tint.
`EquationSteps` shows the steps up to `step` (earlier ones faded); with `onStep` it draws its own `Player`.

## Views are panels; the registry; `Show` and `QuickFigure` (design S §4)

Every view now has a panel form that renders no Figure: `DistributionPanel`, `TracePanel`, `SamplesPanel`,
`CurvePanel`, `TensorModePanel`, `MatrixDecompositionPanel`, `ComputationGraphPanel`, `FitPanel`,
`CrossValidationPanel`, `ElboPanel`, `ChainPanel`, `ParamsPanel`, `DecisionRegionPanel`, `DendrogramPanel`,
`DatasetPanel`, `PairPlotPanel`, `ParallelCoordinatesPanel`, `AndrewsCurvesPanel`. A panel's own controls, readouts and
a line about its object go to the enclosing figure through `<PanelSlot slot="controls" | "readouts" | "about">` (drawn
in place when there is no figure). The old `XView` wrappers (the panel inside `ViewFigure`) are gone: a figure is `<Figure …><XPanel … /></Figure>`, so the
figure owns purpose, state and equation. `TracePanel` and `ComputationGraphPanel` take `startReason` (open at the
finished run, saying why) in place of `startAtFirst` and `initialStep`.

```tsx
registerView({ key: 'distribution/density', kind: 'distribution', description: '…', title: (d) => d.name,
               options: { cdf: setting(true, 'cdf') }, render: (d, opts, { step }) => <DistributionPanel distribution={d} /> })
<Show value={dist} />                                  // the default view for its kind, as a panel
<QuickFigure value={trace} purpose="…" />               // in a standard Figure; options become its state
quickFigure(curve, { title: 'ROC' })
```

`kindOf(o)` reads the object's `kind` field, or a test registered with `registerKind` (distribution, tensor, curve,
dataset, …). Registered: distribution/density, trace/series, tensor/values, curve/unit-square,
computation-graph/backprop, cross-validation/folds, dataset/{scatter,pairs,parallel,andrews}, chains/diagnostics,
params/table, linkage/dendrogram (the last three by key only).
