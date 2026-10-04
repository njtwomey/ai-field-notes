# Migrating note widgets to Plot v2

**Status: done (2026-10-04); this file is now a record.** Note widgets used `aifn-render/compat` (`Interactive`,
`Param*`, `useParam`, `XYChart`, `Heatmap`, `StepControls`, …) and the site's maths (`@/lib/math…`, `@/lib/dsp`,
`@/lib/distributions`). All of it is migrated and deleted: `render/src/compat/`, `site/src/lib/math/`,
`site/src/lib/dsp.ts` and `site/src/lib/distributions/` no longer exist. The site's `DistributionExplorer` draws aifn
distributions (`site/src/components/widgets/distribution-specs.ts` holds each note's parameterisation, sliders and plot
range), and `GradientEstimators` is a site widget. This file maps each construct to its v2 equivalent: `Figure`,
`useFigureState`, `Plot` + layers, `Player` and `aifn`. The lab's record of the same move, with the full layer and axis
reference, is `aifn-js/sandbox/lab/src/MIGRATION.md`; DESIGN.md has the reasons.

**Guard.** `make doctor` (part of `make check`) fails when any file in `content/notes`, `site/src`,
`aifn-js/examples/src` or `aifn-js/sandbox` imports a removed module (`aifn-render/compat`, `@render/compat`,
`@/lib/math`, `@/lib/dsp`, `@/lib/distributions`, or any `…/lib/{math,dsp,distributions}`), imports a legacy name
(`XYChart`, `Heatmap`, `Interactive`, `ImagePlot`, `GlyphPlot`, `MarginalPanels`, `Param{Slider,Choice,Switch,Button}`,
`XYSeries`, `Series`, `HeatmapOverlay`, `useDebouncedCallback`, `GradientEstimators`, …) from `aifn-render`, `@render`
or `@lab`, or uses one of those components as a tag in MDX. The check is in `scripts/doctor.ts` and scans the whole
tree whatever the `SCOPE`. `useParam` and `StepControls` remain: they are v2 controls (`render/src/controls`).

Everything below is imported from `aifn-render` (the root), except the maths, which comes from `aifn/...`.

## Workflow

```bash
node scripts/migrate-notes.ts --dry-run content/notes/<part>      # per file: what it would change, what is left
node scripts/migrate-notes.ts content/notes/<part>                # rewrite and format with prettier
grep -rn "MIGRATE:" content/notes/<part>                           # the hand work, one line per item
npx tsc -p tsconfig.app.json --noEmit && npm run check:content && make doctor SCOPE="<part>"
```

The codemod handles the mechanical part. It leaves anything it cannot do with confidence as it was, and marks it with a
`// MIGRATE: <reason>` line above the statement. It is idempotent. It re-derives its own `MIGRATE` lines on every run.
Output that prettier cannot parse is never written; the report shows it as `ERROR`. The flags are `--print` (show the
result) and `--json <file>` (the report).

A file whose codemod output did not type-check was put back as it was, with a first line
`// MIGRATE: codemod output did not type-check (<first error>); migrate by hand`. The codemod skips any file carrying
that line. Delete the line before running the codemod on the file again.

## `Interactive` → `Figure`

| compat `Interactive` | v2 `Figure`                                                                                                                        |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `title`              | `title` (the figure id and URL anchor are its slug)                                                                                |
| (none)               | `purpose`: optional, and not needed in a note, whose prose already explains the figure (the lab still requires one).               |
| `controls`           | `state={state}` draws the field rows; anything hand-placed (a `Player`, a `Button`) stays in `controls` and renders after the rows |
| `readout`            | `readouts`: a node, or `{ 'group label': <>…</> }` for labelled groups                                                             |
| `equation`           | `equation` (an `<Equation>` band between the controls and the charts)                                                              |
| `caption`            | `caption`                                                                                                                          |
| `className`          | dropped: `Figure` now carries `not-prose`, and the site CSS gives note figures `margin-block: 2rem`                                |

`Readout` is unchanged (from `aifn-render`). One component should hold one `Figure` and one `useFigureState`.
`ParamButton` → `<Button variant="outline" size="sm">` in `controls`.

## `useParam`, `useState` + `Param*` → `useFigureState` fields

One declaration per figure gives the control rows, typed values, handles, reset and URL state
(`?<figure-id>.<field>=<value>`):

```tsx
const state = useFigureState({
  steps: int(10, { min: 1, max: 50, label: 'local steps s' }),
  rate: float(1e-3, { gt: 0, scale: 'log10', suggestions: [1e-4, 1e-3, 1e-2], label: 'learning rate' }),
  angle: slider(0, 90, 30, { step: 1, label: 'angle of w (degrees)' }),
  mode: choice(MODES, 'gauss', { label: 'weight distribution' }),
  logY: setting(false, 'log scale'),
  x0: slider(-3, 3, 0.5, { onChart: true }), // no row: moved by its handle
})
state.steps
state.set('steps', 20)
state.bind('steps') // a Param { value, set, min, max, step }
```

| compat                                                                                                                                                     | v2 field                                                                                                                                                                                        |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `const p = useParam(init, { min, max, step })` + `<ParamSlider param={p} label>`                                                                           | a field keyed `p` (kind below); `p.value` → `state.p`, `p.set(v)` → `state.set('p', v)`, `p` → `state.bind('p')`                                                                                |
| `const [v, setV] = useState(init)` + `<ParamSlider value={v} onChange={setV} min max step label>`                                                          | a field keyed `v`; `v` → `state.v`, `setV(x)` → `state.set('v', x)`                                                                                                                             |
| `ParamSlider` drawn as a slider (`slider`/`variant="slider"`, or a label naming time, angle, threshold, probability, fraction, ratio, correlation, mix, …) | `slider(min, max, init, { step, label, format, steppable })`                                                                                                                                    |
| `ParamSlider` drawn as a number field (every other label)                                                                                                  | `int(init, { min, max, step, label })` for counts (a label naming steps, samples, seed, …, or init, step and bounds that are whole numbers); otherwise `float(init, { min, max, step, label })` |
| number-field props `scale`, `spacing`, `increment`, `points`, `points_per_decade`, `logTransform`, `headerValue`, `suggestions`, `format`                  | the same keys on `int`/`float` (now passed through to `NumberField`)                                                                                                                            |
| `withArrows`, `debounceMs`, `className`                                                                                                                    | dropped: sliders step by default; sliders and handles share the scheduler (`useComputed`)                                                                                                       |
| `<ParamChoice value onChange options label>` with `useState<T>`                                                                                            | `choice<T>(options, init, { label })`; keep `<T>` or `as const` on the options so the value keeps its union type                                                                                |
| `<ParamSwitch checked onChange label>`                                                                                                                     | `setting(init, label)` (a switch). `toggle(init, label)` is a button that reveals an ingredient                                                                                                 |
| `{cond && <ParamSlider …/>}`                                                                                                                               | the field with `when: () => cond`, or `when: (v) => v.mode === 'x'` when it reads other fields                                                                                                  |
| `useParam` moved only by a handle or a click                                                                                                               | `slider(min, max, init, { step, onChart: true })`                                                                                                                                               |
| seed `useState(n)` + `<ParamButton onClick={() => setSeed((s) => s + 1)}>New sample`                                                                       | `seed: int(n, { ge: 0, label: 'seed' })`: the + button draws a new sample, and the URL keeps it                                                                                                 |
| `disabled`                                                                                                                                                 | no equivalent: hide the field with `when`                                                                                                                                                       |

Field values must not read component values declared after `useFigureState`. A field whose range depends on another
field reads it through `when`/closures or moves into a `variants` case. The reserved field names are `values schema set
handle bind reset isDefault json setJson dropped attach`. Rename a local binding called `state` before converting.

**Lab UX rules** (`.scratch/aifn/progress/fun-common.md`) apply to notes too. Counts are `int` with `suggestions`.
Rates and learning rates are `float(…, { gt: 0, scale: 'log10', suggestions })`, never fixed dropdowns. Axes are held.
Class colours are consistent (`group` on `Points`). A value with a natural place on the chart (a threshold, a start
point, a centroid) gets a handle, but distribution parameters stay on sliders. A walk-through uses one `Player` per
figure, opening at step 0.

## `XYChart` → `Plot` + layers

```tsx
const x = useAxis({ label: 'canary share f', range: [0, 0.5] })
const y = useAxis({ label: 'detectable increase (pp)', range: [0, undefined], hold: 'union' })
<Plot x={x} y={y} height={280}>
  <Curve name="detectable increase" x={FRACTIONS} y={curve} slot={0} />
  <Handle {...state.handle('frac', { label: 'canary share' })} />
</Plot>
```

| `XYChart` prop                               | v2                                                                                                                                                                      |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `xLabel`, `yLabel`                           | `useAxis({ label })`, one model per axis, declared at the top level of the component                                                                                    |
| `xRange`, `yRange` (`[0, undefined]`)        | `useAxis({ range })` (an `undefined` end fits)                                                                                                                          |
| no range                                     | `useAxis({ hold: 'union' })`: held, grows to new data and never jumps back (`key` refits for a new dataset)                                                             |
| `yLog`                                       | `useAxis({ log: true })`                                                                                                                                                |
| `equalAspect`                                | `useAxis({ equal: x })` on y                                                                                                                                            |
| `integerX`                                   | `useAxis({ categories })` or `integer: true`                                                                                                                            |
| `height`, `bare`, `onPlotClick`, `ariaLabel` | the same props on `Plot`                                                                                                                                                |
| series `type: 'line'`                        | `<Curve x y name slot dashed thin emphasis muted color />`                                                                                                              |
| series `type: 'line', area: true`            | `<Area x y base? />`                                                                                                                                                    |
| series `type: 'scatter'`                     | `<Points x y group groupNames colors />` (`pointColors` → `colors`; `group` takes slot and shape k)                                                                     |
| series `type: 'bar'`                         | `<Bars x y width? base? />` (precomputed bins: `<Histogram values />` from the raw samples)                                                                             |
| `segments`                                   | `<Segments segments />` (`Segment` is a v2 type now)                                                                                                                    |
| `vectors`                                    | `<Vectors vectors />`                                                                                                                                                   |
| `handles`                                    | one `<Handle …/>` per handle; a state field's handle is `<Handle {...state.handle('k', { label })} />` (`axis: 'y'` for a horizontal guide, `['kx', 'ky']` for a point) |
| two charts sharing x                         | `<Plots rows={2}>` holding two `Plot`s given the same x axis model                                                                                                      |
| a horizontal reference line as a series      | `<Annotation y={…} text="…" dashed muted />`                                                                                                                            |
| `rects`                                      | `<Shapes shapes={[{ contours: [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], tone, opacity }]} />` (labels: `Annotation`)                                                  |

The codemod converts series from an inline array literal (one element per layer) and from a memo that builds an array
literal. A memo series is kept, with its `type` keys dropped and `as const` added, and is spread as
`<Curve {...series[0]} />`. The memo may return the array, or build `const series = […]` and return `{ series, … }`,
read as `result.series` or destructured. A series list built by code (`push`, `.map`, conditional elements, a helper)
goes through `seriesLayers`:

```tsx
const series: SeriesSpec[] = [...]                      // { name, type: 'line' | 'scatter' | 'bar', x, y, area?, group?, colors?, … }
<Plot x={x} y={y}>{seriesLayers(series)}</Plot>         // overlays on a raster: seriesLayers(list, { live: true })
```

The codemod renames `XYSeries`, `Series` and `HeatmapOverlay` to `SeriesSpec` (and `pointColors` to `colors`) once no
compat chart is left in the file. A fixed set of series reads better as layer elements; for a list of curves of one
kind, `{lines.map((l) => <Curve key={l.name} {...l} />)}` is the idiom. Keep layer data memoised: a layer redraws when
its props change by identity.

## `Heatmap` → `Raster` (+ `Contours`, `Points`, `Vectors`, `Handle`)

| `Heatmap` prop                                                               | v2                                                                                       |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `x y z scale range categoryNames scaleTicks fillOpacity valueLabel colorBar` | the same props on `<Raster />` (diverging is symmetric about 0 by default)               |
| held colour scale                                                            | `<Raster colorAxis={useAxis({ hold: 'initial', key })} />`                               |
| `contours={{ levels, field }}`                                               | `<Contours x y z={field ?? z} levels />`                                                 |
| `overlay` series                                                             | layers as for `XYChart`, with `live` (patched, never moves the axes)                     |
| `overlay` scatter `values` (coloured on the raster's scale)                  | `<Points colors={…} />` with `useScaleColor`                                             |
| `marker`                                                                     | `<Points x={[mx]} y={[my]} emphasis live />`, or a `Handle`/`Probe` when it can be moved |
| `onCellClick`                                                                | a point `Handle` (pressing anywhere moves a lone handle) or `onPlotClick`                |
| `equalAspect`, `handles`, `vectors`, `height`, `onPointer`                   | as for `XYChart`                                                                         |

## `StepControls` → `Player`

`StepControls` (`onStep`, `onRun`, `onReset`, `done`) advanced an algorithm by mutation. In v2 the whole trace is
computed up front (in a memo, or with `useComputed` when it is slow), and
`<Player value onChange count label format />` scrubs through its positions. The Player opens at step 0 and runs at
the default 60 positions per second (never pass `defaultSpeed`). When an input changes, restart the walk-through:
store the position together with the input it belongs to, as `RingAllReduce` does.

## Site maths → `aifn`

| site                                                                                     | aifn                                                                                                                                                                 | codemod                                      |
| ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| `rng(seed)`, `g.uniform()`, `g.normal()`, `ReturnType<typeof rng>`                       | `stream(seed)`, `uniform(g)`, `normal(g)`, `Stream` (`aifn/foundation/random`); the same draws                                                                       | yes (by scope; aliases when a name is taken) |
| `linspace(a, b, n)`                                                                      | `toFlat(linspace(a, b, n))` (`aifn/foundation/tensor`; aifn returns a Tensor)                                                                                        | yes                                          |
| `sigmoid`, `normalPdf`, `normalCdf`, `normalQuantile`, `erf`, `logGamma`, `logFactorial` | same names, `aifn/numerics/special` (overloaded ops: pass `(v: number) => f(v)` as a callback)                                                                       | yes                                          |
| `logChoose`, `studentTCdf`, `studentTQuantile`                                           | same names, `aifn/numerics/special`                                                                                                                                  | yes                                          |
| `incompleteBeta(x, a, b)`                                                                | `regularisedBeta(a, b, x)`                                                                                                                                           | yes                                          |
| `incompleteGamma(a, x)`                                                                  | `regularisedGammaP(a, x)`                                                                                                                                            | yes                                          |
| `Vec2`, `Mat2`, `Eig2`, `apply`, `det`, `eig2` (`mat2`)                                  | `Vec2`, `Mat2`, `Eig2`, `apply2`, `det2`, `eig2` (`aifn/numerics/linalg`)                                                                                            | yes                                          |
| `eigSym(p, q, r)`, `cholesky2(p, q, r)`                                                  | `eigh2([[p, q], [q, r]])`, `cholesky2([[p, q], [q, r]])` (eigh2 signs eigenvectors)                                                                                  | yes                                          |
| `svd2(a, b, c, d)`                                                                       | `svd2([[a, b], [c, d]])`: different algorithm and sign convention; check the use                                                                                     | no                                           |
| `hzToMel`, `melToHz`                                                                     | same, `aifn/signal/audio` (HTK by default)                                                                                                                           | yes                                          |
| `isPowerOfTwo`, `nextPowerOfTwo`                                                         | same, `aifn/foundation/fourier`                                                                                                                                      | yes                                          |
| `mean`, `sum` (number arrays)                                                            | `mean`/`sum` of `aifn/foundation/tensor` take a Tensor: `mean(fromData(xs))`, or keep the one-line reduce                                                            | no                                           |
| `fft`, `ifft`, `dft`, `fftInPlace` (`{ re, im }` Float64Arrays)                          | `fft`, `ifft`, `dft` (`aifn/foundation/fourier`) return a complex tensor; no in-place form                                                                           | no                                           |
| `magnitudeSpectrum`, `binFrequencies`, `db`                                              | `magnitude`, `spectrumDecibels` (`aifn/signal`) on a `Spectrum`; **no scalar `db` or bin-frequency helper** (one-liners in place)                                    | no                                           |
| `makeWindow(name, n)`, `kaiser(n, β)`, `WindowName`                                      | `getWindow(spec, n, { periodic })` (`aifn/signal`; returns a Tensor)                                                                                                 | no                                           |
| `lfilter(b, a, x)`, `freqz(b, a, n)`                                                     | `lfilter(filterSpec, x)`, `freqz(system)` (`aifn/signal`; filter objects, not coefficient arrays)                                                                    | no                                           |
| `convolve(x, h)`, `stft`, `unwrap` (phase)                                               | `convolve` (`aifn/foundation/convolution`), `stft` (`aifn/signal/spectral`, a `TimeFrequency` raster); **no phase `unwrap`** (aifn's `unwrap` unwraps traced values) | no                                           |
| `invertCdf(cdf, u, lo, hi)`                                                              | a distribution's `quantile`, or `invertCdf` in `aifn/probability/distributions` (not exported yet)                                                                   | no                                           |
| `binomialPmf/Upper/Lower`, `tPower`, `zPower` (`math/tests`)                             | the binomial distribution's pmf/cdf (`aifn/probability/distributions`); **no power helpers**                                                                         | no                                           |
| `distribution(id)`, `defaults` (`@/lib/distributions`)                                   | the aifn distribution registry (`registry`, `Density`, `Mass` layers)                                                                                                | no                                           |
| `initialCentres`, `CentreInit`, `Point` (`math/cluster`)                                 | `kmeansPlusPlus` (`aifn/numerics/neighbours`)                                                                                                                        | no                                           |
| `olsFit`                                                                                 | `lstsq` (`aifn/numerics/linalg`)                                                                                                                                     | no                                           |

## `ImagePlot`, `GlyphPlot`, `MarginalPanels`

- `ImagePlot` → `<Pixels width height rgb | values scale range />` in a `Plot` with `useAxis({ range: [-0.5, w - 0.5] })`
  on x and `useAxis({ inverse: true, equal: x })` on y, so row 0 is at the top and pixels are square (`xExtent`: a wider
  x range). `lines` → `Curve` layers in pixel coordinates, `handles` → `Handle`, `onPointer` → `Plot onPointer`.
- `GlyphPlot` → `<Shapes shapes={[{ contours, tone, opacity }]} />` in a `bare` `Plot` with equal axes over
  `xRange`/`yRange`; `guides` → `<Annotation y={…} />` per guide.
- `MarginalPanels` → a `Plots` grid, `<Plots rows={2} cols={2} heights={[3, 1]} widths={[3, 1]}>`, with the main
  panel's x model shared by the bottom panel and its y model shared by the right one (an empty `<div />` in the
  corner). The marks map to layers: `line` → `Curve` (for the right panel, x = value and y = position), `bars` →
  `Bars edges orient="y"` on the right, `fill` and `rect` → `Shapes` or `Area`, `segments` → `Segments`, `dots` →
  `Points`, `overlay` → the same layers with `live`.

## Not in v2 yet

- Control `disabled`: no field equivalent (`when` hides the field instead).
