# aifn lab: design (UX v2, draft 1)

The lab is the second version of the site's interactive layer. It depends only on `aifn` and third-party packages. This
document fixes what a lab page is for (§1–2) and the components that make one (§3–8). It replaces the current practice
of patching each page as feedback arrives. Points for the owner are marked **[decide]** and collected in §10.

## 1. Why a redesign

Every piece of feedback on the lab so far falls into one of six gaps, and each gap was patched in place:

| Feedback                                                                                                               | Gap                                                                                                          |
| ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Sliders invisible, unclickable, tiny, not typeable; controls crammed together; "one row per thing"                     | Controls are placed by hand on every page                                                                    |
| Panels misaligned, one toolbar incomplete, circle not round, axes rescaling on every change, arrows vanishing off-axis | Each chart owns its axes; sharing, aspect, holding, zoom and clipping are separate features added one by one |
| Click the heatmap to query, drag x₀, the osculating circle, supports shown on the axes                                 | The same pattern (a point the reader moves, with numbers computed there) is rebuilt on every page            |
| Histograms faint or missing, support bands, a signed KL area, a rotated output density                                 | Probability pictures are assembled from raw lines and bars each time                                         |
| "Terrible, I don't know its purpose" (KL), "quite poor" (transformed), the backprop table                              | There is no standard for what a figure must show                                                             |
| Every visual bug found by the owner, not by the builder                                                                | Builders cannot see the page: the render check proves it renders, not that it reads                          |

The symptom in the code: `XYChart` has 30 props, most added for one page, and `Heatmap` repeats the axis logic.
Specimens hold 197 `useState` calls, each wired by hand to a slider, sometimes a handle, and a readout.

## 2. The standard for a lab page

A **specimen** is a lab page about one aifn object or algorithm. It holds one or more **figures**. Every figure:

1. **Has one purpose, stated in one line** under the title (`purpose`, required; the check fails without it). Example:
   "KL divergence is the expected log-ratio, so its value is the net signed area under p log(p/q)."
2. **Shows its point with the defaults.** The first render already shows the idea: a bimodal p for mode seeking, a
   non-linear map for the Jacobian. No "move the slider to see anything".
3. **Orders its controls as the story unfolds**, one row per step (input, then map, then what to reveal), each row
   labelled.
4. **Makes the thing itself draggable** where it has a natural place on a chart (a point, a threshold, a start value),
   never a proxy. Distribution parameters stay on sliders.
5. **Reports numbers where the reader is looking**: at the probe (§6), in the equation band (§7) or in grouped readouts.
6. **Holds its axes** while parameters move; changing an entity (a family, a map, a dataset) refits.
7. **Reveals, not only displays**: a toggle that turns on the ingredient being taught (the Jacobian, the prior) beats
   showing everything at once.
8. **Plays when it evolves.** Anything that unfolds over steps or time (an ODE or SDE solution, a PDE, a map's orbit,
   a sampler, an optimiser, a filter, a training run, an algorithm's steps) has a `Player`: play and pause, step, a
   scrubber and a speed. The figure shows the state at the current step (a moving point, a partial path, the field at
   time t), not only the finished result.
9. **Is computed by aifn.** The lab draws; it never implements maths.
10. **Has been looked at**: `make lab-shots ARGS="--only <page>"` in light and dark, read before reporting.

A **specimen review** applies this checklist to every existing page from its screenshots, the way the prose review
applies the writing standard to notes.

## 3. Anatomy of a figure

```
┌ Figure ─────────────────────────────────────────────────────────── [size ▾] [copy] ┐
│ Title                                                                       #anchor │
│ Purpose line                                                                        │
│ ── 1 · input ───────  [family ▾]  μ ━━●━━  σ ━●━━━                                  │
│ ── 2 · map ─────────  [map ▾]     T ━●━━━                    (rows from the schema) │
│ ── 3 · reveal ──────  [Jacobian scaling ◐]  draws [1000 ▾]                          │
│ ┌ Equation band (large, live numbers) ─────────────────────────────────────────────┐ │
│ │  p_Y(0.47) = p_X(−0.07) · |1/g′(−0.07)| = 0.036 · 2.81 = 0.101                   │ │
│ └──────────────────────────────────────────────────────────────────────────────────┘ │
│ ┌ Plots (a grid of Plot, shared axes) ─────────────────────────────────────────────┐ │
│ │  …                                                                               │ │
│ └──────────────────────────────────────────────────────────────────────────────────┘ │
│ Readouts, in labelled groups                                                        │
│ Caption: what can be dragged, what to notice                                        │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

`Figure` keeps its current frame (sizes, drag corner, copy data, anchor). It gains `purpose`, `equation` and grouped
`readouts`, and takes its controls from the figure's state (§4) instead of hand-placed children.

## 4. Figure state: one declaration, every binding

```ts
const state = useFigureState({
  input: variants({ normal: { mu: slider(-3, 3, 0), sd: slider(0.1, 3, 1) }, gamma: { … } }, { label: '1 · input' }),
  map:   variants({ sigmoid: { T: slider(0.1, 3, 1) }, exp: {}, … },                        { label: '2 · map' }),
  reveal: row('3 · reveal', { jacobian: toggle(false, 'Jacobian scaling'), draws: choice([1000, 5000, 20000]) }),
  x0: slider(-4, 4, 0.5, { onChart: true }),   // bound to a handle too, but not shown as a row
})
state.input.key; state.input.values.mu; state.reveal.jacobian; state.x0
<Figure state={state} …>                       // renders the rows
```

- One declaration yields the control rows, the parameter types (variants narrow by key, as `useVariants` does now), the
  bindings for handles (`state.handle('x0')` returns a handle that writes the same value, clamped and snapped), and a
  reset button.
- **URL state** **[decide]**: the figure's values live in the URL next to its anchor
  (`…/transformed#change-of-variables?input=gamma&shape=2&map=sigmoid`), so a link reproduces a view. Only non-default
  values are written.
- Toggles come in two kinds: `toggle` (a revealing button, as the Jacobian) and `switch` (a setting).
- Absorbs today's `useParam`, `useVariants`, `useParams`, `ParamControls` and `ControlRow`.

## 5. Plots: one axis model, charts built from layers

`XYChart` and `Heatmap` are split into a `Plot` that owns two axes and **layers** drawn inside it.

```tsx
<Plots rows={2} cols={2} share={{ x: 'col', y: 'row' }} heights={[3, 2]} widths={[2, 1]}>
  <Plot x={xAxis} y={yAxis}>
    {' '}
    {/* the map */}
    <Curve x={xs} y={gx} emphasis />
    <Probe probe={x0} /> {/* §6: guide lines, tangent, dx → dy */}
  </Plot>
  <Plot x={densityAxis} y={yAxis}>
    {' '}
    {/* the output density, rotated */}
    <Histogram values={gxDraws} orient="y" />
    <Density dist={Y} orient="y" />
    <SupportBand interval={Y.support} orient="y" />
  </Plot>
  <Plot x={xAxis} y={densityAxis}>
    <Histogram values={xDraws} /> <Density dist={X} /> <SupportBand interval={X.support} />
  </Plot>
</Plots>
```

- **Axis model** (`useAxis({ fit, hold, key, aspect, log, support })`): one object per axis, shared by every Plot that
  uses it. Fit policy: `fit` (data), `hold: 'initial' | 'union'`, `fixed [a, b]`, `support` (a distribution's interval,
  with bounded ends as hard limits and infinite ends at quantiles). `key` refits. Aspect `equal` links two axes, and the
  grid sizes the panel for it (as `Panel aspect="equal"` does now). Zoom, pan and the toolbar belong to the axis, so
  every panel sharing it moves together and the controls are drawn once.
- **The toolbar is compact by default**: one small button per axis (↔, ↕) opening that axis's full controls in a
  popover (typed range, pan, zoom), and one auto-scale button per chart or grid. Every chart has all of it; none of
  it takes space until asked for.
- **Layers** are small components with one job and a declared extent (which feeds the fit, unless `live`):
  `Curve`, `Points`, `Bars`, `Area`, `SignedArea` (two colours about zero, reports its net area), `Histogram` (from
  samples via `aifn/stats`, with `orient`), `Density` and `Mass` (from an `aifn/distributions` object, with `orient`),
  `Rug`, `SupportBand`, `Vectors` (clipped), `Segments`, `Grid2d` (the heatmap raster), `Contours`, `Handle`, `Probe`,
  `Annotation`. A layer marked `live` updates by patch without redrawing.
- `EChart.tsx` stays the only file that touches ECharts. `Plot` collects its layers' series and builds one option.
- Hover, legends, palette slots and the tooltip are the Plot's, so every chart behaves the same.
- The current `XYChart` becomes a thin wrapper over `Plot` for simple cases and old code, then is retired.

## 6. Probes: a point the reader moves, with numbers computed there

```ts
const probe = useProbe({ x: state.x0 })            // or { x, y } for a point on a heatmap
<Probe probe={probe} />                              // in any Plot sharing that axis: a guide line or marker + handle
<ProbeReadout probe={probe} values={{ 'p_X(x₀)': X.prob(probe.x), 'g′(x₀)': slope(probe.x) }} />
```

- One probe is shared by every plot that shows its coordinate: dragging it on the map moves it on the density.
- Click-to-set on a heatmap is a probe with `{ x, y }` (the conditioning query).
- Probe readouts sit beside the plots or in the equation band.
- Replaces the per-page handle wiring on derivatives, conditioning, change of variables and reliability.

## 7. Equations

- `Equation`: large KaTeX (about 1.6× body), with live values substituted and highlighted, e.g. the Jacobian step
  above. Takes a template with named slots: `tex\`p_Y(${y}) = p_X(${x}) \cdot |1/g'(${x})|\``.
- `EquationSteps`: a sequence of equations for walkthroughs, such as backprop's forward assignments and chain-rule
  steps, driven by a `Player`.

## 8. Stepping and traces

`Player` stays the one scrubber. A figure that steps (optimisers, graph algorithms, backprop, simplex) takes a trace
from `aifn/trace` and a `step` from its state. `TraceView`, `GraphView` and the computation graph view read the same
`{ trace, step }`.

## 8a. Interaction performance: one scheduler, not ad-hoc debouncing

Dragging must feel immediate even when the figure's answer is expensive. Example: in the UI kit's heatmap with
contours and a draggable start, each pointer move re-runs the gradient path and re-renders the figure before the
previous answer has landed. Debouncing trades that for lag, and doing it per widget repeats it everywhere. One
scheduler handles it:

- **Two speeds per interaction.** What the reader holds (the handle, the probe marker, guide lines) moves on the
  pointer event, by a `live` patch, with no React render of the rest. What is derived from it (a path, a posterior, a
  trace) is computed by the scheduler.
- **Latest wins, once per frame.** `useComputed(fn, inputs, { mode })` coalesces input changes to at most one run per
  animation frame, always on the latest inputs, and drops stale results. A run that started for an old input never
  paints over a newer one. Nothing is delayed by a timer.
- **Modes by cost, measured, not guessed.** The scheduler times each run. Under the frame budget (about 8 ms) it runs
  every frame. Over it, it runs as often as it can while dragging and shows the last result marked stale (a subtle
  dimming of the derived layers only), then always runs on release. `mode: 'release'` defers to release outright;
  `mode: 'worker'` runs in a Web Worker with cancellation (aifn is DOM-free, so algorithms move without change).
- **Incremental where aifn allows.** Traces extend instead of recomputing (`extend`), replicates reuse prefixes
  (`replicate`), so a raised iteration count or draw count costs only the new part.
- **Redraw only what changed.** `Plot` sends ECharts a patch of the layers whose data changed. Static layers (a
  heatmap raster, contours, a density) are never resent during a drag.
- **Measured.** `make lab-shots ARGS="--profile …"` scripts a drag and reports input-to-paint latency, dropped frames
  and redraws per move, so a page's interaction cost is a number in its review, not an impression.
- `Slider`'s `debounceMs` and per-widget throttles are removed once the scheduler lands; the slider feeds the same
  scheduler as a handle.

## 9. Components: keep, break up, replace

| Today                                                                                                                      | After                                                                                                  |
| -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `XYChart` (30 props), `Heatmap`                                                                                            | `Plot` + layers; thin compatibility wrappers, then retired                                             |
| `Subplots`/`Panel`, `ChartSize`, per-chart `aspect`, `rescaleOnChange`, `axisKey`, `holdFit`, `ViewportControls` per chart | `Plots` grid + `useAxis` (sharing, aspect, holding and toolbars in one place)                          |
| `useParam`, `useVariants`, `useParams`, `ParamControls`, `ControlRow`                                                      | `useFigureState` + schema builders                                                                     |
| Handles wired per page                                                                                                     | `state.handle(name)` and `Probe`                                                                       |
| `Readout` lists                                                                                                            | `Readouts` groups, `ProbeReadout`, `Equation`                                                          |
| `DistributionView`, `SamplesView`, `CurveView`                                                                             | Rebuilt on layers (`Density`, `Histogram`, `SupportBand`), so the views and pages share one vocabulary |
| `Figure`                                                                                                                   | Same frame + `purpose`, `equation`, `state`                                                            |
| `debounceMs`, per-widget throttles, `useMemo` chains recomputed per pointer move                                           | `useComputed` scheduler (§8a) + live layers                                                            |

## 10. Decisions for the owner

Adopted as recommended (2026-09-30, "carry on"); revisit any on request.

1. **URL state per figure** (§4): recommended yes, non-default values only.
2. **Declarative state for every figure**, or only new ones: recommended every figure, done during the specimen review.
3. **Retire `XYChart` and `Heatmap`** after migration, or keep them as wrappers: recommended retire.
4. **The equation band's default**: shown when a figure declares one; no automatic equations.
5. **Order**: screenshot harness (under way), then axis model and layers, then state and probes, then `Equation`,
   then migrate and review page by page. aifn batch 3 runs alongside, since it does not touch the UI.

## 11. Order of work

1. `make lab-shots` (under way).
2. Axis model, `Plot`, `Plots` and the layers, with a UI-kit page per layer. Port `DistributionView` first as the test.
3. `useFigureState`, schema builders, handles from state, URL state, reset; the `useComputed` scheduler (§8a).
4. `useProbe`/`Probe`, `Equation`/`EquationSteps`, grouped readouts, `purpose` required.
5. Specimen review: each page against §2, from screenshots, migrated as it is reviewed; `XYChart`/`Heatmap` retired
   when the last user goes.
