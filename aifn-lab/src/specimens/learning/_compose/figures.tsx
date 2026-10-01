import { useMemo } from 'react'
import {
  log1pTarget,
  logTarget,
  pipeline,
  powerTarget,
  transformTarget,
  type TargetMap,
  type TargetMapEstimator,
} from 'aifn/learning/compose'
import { asTensor, dataset, evaluate } from 'aifn/learning/estimators'
import { accuracy, logLoss } from 'aifn/learning/metrics'
import type { AnyUnivariate } from 'aifn/probability/distributions'
import { linearRegression } from 'aifn-applied/learning/linear'
import { logisticRegression } from 'aifn-applied/learning/generalised/glm'
import { polynomialFeatures, standardScaler } from 'aifn-applied/learning/preprocessing'
import { child, normals, stream } from 'aifn/foundation/random'
import { fromData, linspace, mean, toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { Figure } from '@lab/layout'
import { choice, row, slider, useFigureState } from '@lab/state'
import { Annotation, Area, Curve, Handle, Plot, Plots, Points, Raster, Readout, useAxis } from '@lab/viz'
import { formatValue, StateTree } from '@lab/views'

const grid = (lo: number, hi: number, n: number) => toFlat(linspace(lo, hi, n))
const sdOf = (v: readonly number[]) => {
  const m = v.reduce((a, b) => a + b, 0) / v.length
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length)
}
const sd3 = (v: number) => formatValue(Number(v.toPrecision(3)))

/** The fitted state of a model: its fields that are not functions. */
const stateOf = (m: object) => Object.fromEntries(Object.entries(m).filter(([, v]) => typeof v !== 'function'))

// ---------------------------------------------------------------------------------------------------------------------

const STAGES = [
  { value: 'input', label: 'raw input' },
  { value: 'scaled', label: '1 · standardScaler' },
  { value: 'features', label: '2 · polynomialFeatures' },
  { value: 'rescaled', label: '3 · standardScaler (of the features)' },
  { value: 'model', label: '4 · logisticRegression' },
] as const
const CLASS_NAMES = ['class 0', 'class 1']

/**
 * Two classes on raw scales far from unit (x₀ around 5 with sd 3, x₁ around 10 with sd 0.5), separated by a parabola
 * in standardised units: class 1 when u² + v + noise > 0.8.
 */
function parabolaData(n: number) {
  const s = stream('pipeline-steps')
  const u = toFlat(normals(child(s, 'u'), [n]))
  const v = toFlat(normals(child(s, 'v'), [n]))
  const e = toFlat(normals(child(s, 'e'), [n]))
  const x = fromData(
    Float64Array.from({ length: 2 * n }, (_, k) => (k % 2 === 0 ? 5 + 3 * u[k >> 1] : 10 + 0.5 * v[k >> 1])),
    [n, 2],
  )
  const y = fromData(Float64Array.from(u, (a, i) => (a * a + v[i] + 0.4 * e[i] > 0.8 ? 1 : 0)))
  return dataset(x, y)
}

export function PipelineStepsSpecimen() {
  const state = useFigureState({
    fit: row('1 · pipeline', {
      degree: slider(1, 4, 2, { label: 'polynomial degree', step: 1 }),
      l2: slider(0.001, 10, 0.1, { label: 'l2' }),
    }),
    view: row('2 · stage', {
      stage: choice(STAGES, 'input', { label: 'show the data as this step sees it' }),
    }),
  })
  const { degree, l2 } = state.fit
  const stage = state.view.stage
  const data = useMemo(() => parabolaData(150), [])
  const model = useMemo(
    () =>
      // Standardised again after the expansion: x₀⁴ and x₀ differ by orders of magnitude in spread, and an L2 penalty
      // (and Newton's conditioning) would otherwise treat the features very unequally.
      pipeline(
        standardScaler(),
        polynomialFeatures({ degree, includeBias: false }),
        standardScaler(),
        logisticRegression({ l2 }),
      ).fit(data),
    [data, degree, l2],
  )
  const scores = evaluate(model, data, [accuracy, logLoss])
  const labels = useMemo(() => toFlat(data.y), [data])
  // The probability field covers the data with a margin, so every point sits on it.
  const [xs, ys] = useMemo(() => {
    const rows = toRows(data.x)
    const span = (k: 0 | 1, n: number) => {
      const v = rows.map((r) => r[k])
      const [lo, hi] = [Math.min(...v), Math.max(...v)]
      const pad = 0.08 * (hi - lo)
      return grid(lo - pad, hi + pad, n)
    }
    return [span(0, 90), span(1, 60)]
  }, [data])
  const view = useMemo(() => {
    const stages = model.stages(data.x) as Tensor[]
    if (stage === 'input' || stage === 'scaled') {
      const rows = toRows(stages[stage === 'input' ? 0 : 1])
      const x = rows.map((r) => r[0])
      const y = rows.map((r) => r[1])
      return { kind: 'scatter' as const, x, y, sd: [sdOf(x), sdOf(y)] }
    }
    if (stage === 'features' || stage === 'rescaled') {
      const rows = toRows(stages[stage === 'features' ? 2 : 3])
        .slice(0, 40)
        .reverse()
      return { kind: 'matrix' as const, x: rows[0].map((_, j) => j), y: rows.map((_, i) => i), z: rows }
    }
    // The model: P(y = 1 | x) over a grid of raw inputs, through the whole pipeline.
    const points = new Float64Array(xs.length * ys.length * 2)
    ys.forEach((b, i) =>
      xs.forEach((a, j) => ((points[2 * (i * xs.length + j)] = a), (points[2 * (i * xs.length + j) + 1] = b))),
    )
    const p = toFlat(model.expect(fromData(points, [xs.length * ys.length, 2])))
    const rows = toRows(data.x)
    return {
      kind: 'probability' as const,
      z: ys.map((_, i) => p.slice(i * xs.length, (i + 1) * xs.length)),
      x: rows.map((r) => r[0]),
      y: rows.map((r) => r[1]),
    }
  }, [model, data, stage, xs, ys])
  const stepIndex = { input: -1, scaled: 0, features: 1, rescaled: 2, model: 3 }[stage]
  const names = ['feature', ...(model.steps[1].featureNames as readonly string[])]
  const raw = stage === 'input' || stage === 'model'
  // Each stage is a different space: its axes refit when the stage changes.
  const matrix = view.kind === 'matrix'
  // Each axis fits its own data (equal units flatten the raw cloud into a band, x₀ spanning six times x₁); the label
  // states the axis's sd, so the scales the scaler removes are read off the labels.
  const sdLabel = (k: 0 | 1) => (view.kind === 'scatter' ? ` (sd ${sd3(view.sd[k])})` : '')
  const ax = useAxis({
    label: matrix ? 'feature' : `${raw ? 'x₀' : 'x₀ standardised'}${sdLabel(0)}`,
    key: `${stage}:${degree}`,
    categories: matrix ? names.slice(1) : undefined,
  })
  const ay = useAxis({
    label: matrix ? 'row (first 40, top first)' : `${raw ? 'x₁' : 'x₁ standardised'}${sdLabel(1)}`,
    key: stage,
  })
  return (
    <Figure
      purpose="A pipeline is fitted step by step: each step learns its state from the data as the previous steps left it, and the last step's prediction is a function of the raw input."
      title="A pipeline's fitted state, step by step"
      defaultSize="L"
      state={state}
      readouts={{
        fit: (
          <>
            <Readout label="training accuracy" value={formatValue(scores.accuracy)} />
            <Readout label="training log loss" value={formatValue(scores.logLoss)} />
            <Readout label="features" value={names.length - 1} />
            <Readout label="Newton steps" value={model.steps[3].steps} />
          </>
        ),
      }}
      caption="pipeline(standardScaler(), polynomialFeatures(), standardScaler(), logisticRegression()) fitted once; choose a stage to see the data as that step sees it and the step's fitted state. The raw axes differ by a factor of six in spread, and the first scaler brings them to unit variance. The polynomial features let a linear classifier draw the parabola, but their spreads differ again (at degree 4 the x₀⁴ column dwarfs x₀ on the shared colour scale), so a second scaler standardises each feature before the L2-penalised fit, which would otherwise shrink the features unequally. The final stage is P(y = 1 | x) of the whole pipeline on raw inputs. Degree 1 cannot separate the classes."
    >
      <div className="grid gap-3 md:grid-cols-[3fr_2fr]">
        <div>
          <Plot x={ax} y={ay}>
            {view.kind === 'scatter' && (
              <Points name="rows" x={view.x} y={view.y} group={labels} groupNames={CLASS_NAMES} />
            )}
            {view.kind === 'matrix' && (
              <Raster x={view.x} y={view.y} z={view.z} scale="diverging" valueLabel="feature value" />
            )}
            {view.kind === 'probability' && (
              <Raster x={xs} y={ys} z={view.z} range={[0, 1]} scale="diverging" valueLabel="P(y = 1 | x)" />
            )}
            {view.kind === 'probability' && (
              <Points name="rows" x={view.x} y={view.y} group={labels} groupNames={CLASS_NAMES} />
            )}
          </Plot>
        </div>
        <div className="max-h-96 overflow-auto rounded-md border p-2">
          {stepIndex < 0 ? (
            <p className="text-xs text-muted-foreground">The raw input has no fitted state.</p>
          ) : (
            <StateTree value={stateOf(model.steps[stepIndex] as object)} name={model.names[stepIndex]} />
          )}
        </div>
      </div>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

const MAP_NAMES = ['log', 'log1p', 'box-cox', 'none'] as const
type MapName = (typeof MAP_NAMES)[number]

const MAPS: Record<Exclude<MapName, 'none'>, () => TargetMap | TargetMapEstimator> = {
  log: logTarget,
  log1p: log1pTarget,
  'box-cox': () => powerTarget({ method: 'box-cox' }),
}
const XS = grid(-3, 3, 121)
const YS = grid(0.01, 12, 300)

export function TransformedTargetSpecimen() {
  const state = useFigureState({
    data: row('1 · data', { sigma: slider(0.05, 1, 0.4, { label: 'noise σ on log y' }) }),
    model: row('2 · model', { mapName: choice(MAP_NAMES, 'log', { label: 'target map' }) }),
    at: slider(-3, 3, 1.5, { onChart: true, label: 'x for the density' }),
  })
  const { sigma } = state.data
  const mapName: MapName = state.model.mapName
  const at = state.at
  // log y = 0.3 + 0.7x + σε: a multiplicative, right-skewed target.
  const data = useMemo(() => {
    const s = stream('target')
    const x = toFlat(normals(child(s, 'x'), [120], 0, 1.2))
    const e = toFlat(normals(child(s, 'e'), [120]))
    return dataset(
      fromData(Float64Array.from(x), [x.length, 1]),
      fromData(Float64Array.from(x, (v, i) => Math.exp(0.3 + 0.7 * v + sigma * e[i]))),
    )
  }, [sigma])
  const model = useMemo(
    () =>
      mapName === 'none'
        ? linearRegression().fit(data)
        : transformTarget(linearRegression(), MAPS[mapName]()).fit(data),
    [data, mapName],
  )
  const fit = useMemo(() => {
    const input = fromData(Float64Array.from(XS), [XS.length, 1])
    const d = model.predictive(input) as AnyUnivariate
    const q = (p: number) => toFlat(asTensor(d.quantile(fromData(Float64Array.of(p), []))))
    return {
      x: toFlat(data.x),
      y: toFlat(data.y),
      lo: q(0.05),
      hi: q(0.95),
      decide: toFlat(model.decide(input)),
      expect: toFlat(model.expect(input)),
    }
  }, [model, data])
  const d0 = useMemo(() => model.predictive(fromData(Float64Array.of(at), [1, 1])) as AnyUnivariate, [model, at])
  const density = useMemo(
    () => YS.map((v) => Math.exp(toFlat(asTensor(d0.logProb(fromData(Float64Array.of(v), []))))[0])),
    [d0],
  )
  const median = toFlat(model.decide(fromData(Float64Array.of(at), [1, 1])))[0]
  const mean0 = toFlat(asTensor(d0.mean()))[0]
  const x = useAxis({ label: 'x', range: [-3, 3] })
  const y = useAxis({ label: 'y', range: [0, 12] })
  const dens = useAxis({ label: 'density', hold: 'union', key: mapName })
  return (
    <Figure
      purpose="Fitting a regression to g(y) and pushing its Gaussian predictive back through g⁻¹ gives a skewed predictive: for log y a log-normal, whose median (the inverted point prediction) sits below its mean."
      title="A transformed target's predictive distribution"
      defaultSize="L"
      state={state}
      readouts={{
        [`at x = ${formatValue(at)}`]: (
          <>
            <Readout label="predictive" value={d0.name} />
            <Readout label="median (decide)" value={formatValue(median)} />
            <Readout label="mean" value={formatValue(mean0)} />
          </>
        ),
        fit: (
          <Readout
            label="training log score"
            value={formatValue(-mean(model.predictive(data.x).logProb(data.y) as Tensor))}
          />
        ),
      }}
      caption="transformTarget fits linearRegression on g(y) and pushes its Gaussian predictive through g⁻¹: log gives a log-normal, other maps a general transformed distribution. decide(x) is the inverted point prediction (the median), below the mean E[y | x] for a right-skewed law. Drag the vertical line to choose x; the right panel is the predictive density there, on the same y axis. Without a map the Gaussian predictive puts mass on negative y and the log score is worse."
    >
      <Plots cols={2} widths={[3, 1.1]} hoverGroup>
        <Plot x={x} y={y}>
          <Points name="data" x={fit.x} y={fit.y} muted />
          <Area name="90% predictive band" x={XS} y={fit.hi} base={fit.lo} slot={2} line={false} opacity={0.18} />
          <Curve name="decide(x): median" x={XS} y={fit.decide} slot={0} />
          <Curve name="E[y | x]" x={XS} y={fit.expect} slot={1} />
          <Handle {...state.handle('at', { label: 'x' })} />
        </Plot>
        <Plot x={dens} y={y}>
          <Area name={`p(y | x = ${formatValue(at)})`} x={YS} y={density} orient="y" slot={1} />
          <Annotation y={median} slot={0} />
        </Plot>
      </Plots>
    </Figure>
  )
}
