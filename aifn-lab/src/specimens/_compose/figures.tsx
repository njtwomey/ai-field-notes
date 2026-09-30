import { useMemo, useState } from 'react'
import {
  log1pTarget,
  logTarget,
  pipeline,
  powerTarget,
  transformTarget,
  type TargetMap,
  type TargetMapEstimator,
} from 'aifn/compose'
import {
  accuracy,
  asTensor,
  evaluate,
  linearRegression,
  logisticRegression,
  logLoss,
  type UnivariateDistribution,
} from 'aifn/estimators'
import { polynomialFeatures, standardScaler } from 'aifn/preprocess'
import { normals, stream } from 'aifn/random'
import { fromData, linspace, toFlat, toRows, type Tensor } from 'aifn/tensor'
import { Select, Slider } from '@lab/controls'
import { Figure } from '@lab/layout'
import { ChartSize, Heatmap, Readout, XYChart, type HeatmapOverlay, type XYSeries } from '@lab/viz'
import { formatValue, StateTree } from '@lab/views'

const grid = (lo: number, hi: number, n: number) => toFlat(linspace(lo, hi, n))

/** The fitted state of a model: its fields that are not functions. */
const stateOf = (m: object) => Object.fromEntries(Object.entries(m).filter(([, v]) => typeof v !== 'function'))

// ---------------------------------------------------------------------------------------------------------------------

type Stage = 'input' | 'scaled' | 'features' | 'model'

/**
 * Two classes on raw scales far from unit (x₀ around 5 with sd 3, x₁ around 10 with sd 0.5), separated by a parabola
 * in standardised units: class 1 when u² + v + noise > 0.8.
 */
function parabolaData(n: number) {
  const s = stream('pipeline-steps')
  const u = toFlat(normals(s.child('u'), [n]))
  const v = toFlat(normals(s.child('v'), [n]))
  const e = toFlat(normals(s.child('e'), [n]))
  const x = fromData(
    Float64Array.from({ length: 2 * n }, (_, k) => (k % 2 === 0 ? 5 + 3 * u[k >> 1] : 10 + 0.5 * v[k >> 1])),
    [n, 2],
  )
  const y = fromData(Float64Array.from(u, (a, i) => (a * a + v[i] + 0.4 * e[i] > 0.8 ? 1 : 0)))
  return { x, y }
}

export function PipelineStepsSpecimen() {
  const [degree, setDegree] = useState(2)
  const [l2, setL2] = useState(0.1)
  const [stage, setStage] = useState<Stage>('model')
  const data = useMemo(() => parabolaData(150), [])
  const model = useMemo(
    () =>
      pipeline(standardScaler(), polynomialFeatures({ degree, includeBias: false }), logisticRegression({ l2 })).fit(
        data,
      ),
    [data, degree, l2],
  )
  const scores = evaluate(model, data, [accuracy, logLoss])
  const labels = toFlat(data.y)
  const [xs, ys] = useMemo(() => [grid(-4, 14, 90), grid(8, 12, 60)], [])
  const view = useMemo(() => {
    const stages = model.stages(data.x) as Tensor[]
    if (stage === 'input' || stage === 'scaled') {
      const rows = toRows(stages[stage === 'input' ? 0 : 1])
      const series: XYSeries[] = [
        {
          name: 'rows',
          type: 'scatter',
          x: rows.map((r) => r[0]),
          y: rows.map((r) => r[1]),
          group: labels,
          groupNames: ['class 0', 'class 1'],
        },
      ]
      return { kind: 'scatter' as const, series }
    }
    if (stage === 'features') {
      const rows = toRows(stages[2]).slice(0, 40)
      return {
        kind: 'matrix' as const,
        x: rows[0].map((_, j) => j),
        y: rows.map((_, i) => -i),
        z: rows,
      }
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
      overlay: [
        {
          name: 'rows',
          type: 'scatter',
          x: rows.map((r) => r[0]),
          y: rows.map((r) => r[1]),
          group: labels,
          groupNames: ['class 0', 'class 1'],
        },
      ] as HeatmapOverlay[],
    }
  }, [model, data, stage, labels, xs, ys])
  const stepIndex = { input: -1, scaled: 0, features: 1, model: 2 }[stage]
  const names = ['feature', ...(model.steps[1].featureNames as readonly string[])]
  return (
    <Figure
      title="A pipeline's fitted state, step by step"
      defaultSize="L"
      controls={
        <>
          <Select
            label="stage"
            value={stage}
            onChange={setStage}
            options={[
              { value: 'input', label: 'raw input' },
              { value: 'scaled', label: '1 · standardScaler' },
              { value: 'features', label: '2 · polynomialFeatures' },
              { value: 'model', label: '3 · logisticRegression' },
            ]}
          />
          <Slider label="degree" value={degree} min={1} max={4} step={1} onChange={setDegree} />
          <Slider label="l2" value={l2} min={0.001} max={10} onChange={setL2} />
        </>
      }
      readouts={
        <>
          <Readout label="training accuracy" value={formatValue(scores.accuracy)} />
          <Readout label="training log loss" value={formatValue(scores['log-loss'])} />
          <Readout label="features" value={names.length - 1} />
          <Readout label="Newton steps" value={model.steps[2].iterations} />
        </>
      }
      caption="pipeline(standardScaler(), polynomialFeatures(), logisticRegression()) fitted once; each stage shows the data as that step sees it and the step's fitted state. The raw axes differ by a factor of six in spread, the scaler brings them to unit variance, the polynomial features let a linear classifier draw the parabola, and the final panel is P(y = 1 | x) of the whole pipeline on raw inputs. Degree 1 cannot separate the classes."
    >
      <div className="grid gap-3 md:grid-cols-[3fr_2fr]">
        <div>
          {view.kind === 'scatter' && (
            <XYChart
              series={view.series}
              xLabel={stage === 'input' ? 'x₀' : 'x₀ standardised'}
              yLabel={stage === 'input' ? 'x₁' : 'x₁ standardised'}
            />
          )}
          {view.kind === 'matrix' && (
            <Heatmap x={view.x} y={view.y} z={view.z} scale="diverging" xLabel="feature" yLabel="−row (first 40)" />
          )}
          {view.kind === 'probability' && (
            <Heatmap x={xs} y={ys} z={view.z} range={[0, 1]} overlay={view.overlay} xLabel="x₀" yLabel="x₁" />
          )}
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

type MapName = 'log' | 'log1p' | 'box-cox' | 'none'

const MAPS: Record<Exclude<MapName, 'none'>, () => TargetMap | TargetMapEstimator> = {
  log: logTarget,
  log1p: log1pTarget,
  'box-cox': () => powerTarget({ method: 'box-cox' }),
}

export function TransformedTargetSpecimen() {
  const [mapName, setMapName] = useState<MapName>('log')
  const [sigma, setSigma] = useState(0.4)
  const [at, setAt] = useState(1.5)
  // log y = 0.3 + 0.7x + σε: a multiplicative, right-skewed target.
  const data = useMemo(() => {
    const s = stream('target')
    const x = toFlat(normals(s.child('x'), [120], 0, 1.2))
    const e = toFlat(normals(s.child('e'), [120]))
    return {
      x: fromData(Float64Array.from(x), [x.length, 1]),
      y: fromData(Float64Array.from(x, (v, i) => Math.exp(0.3 + 0.7 * v + sigma * e[i]))),
    }
  }, [sigma])
  const model = useMemo(
    () =>
      mapName === 'none'
        ? linearRegression().fit(data)
        : transformTarget(linearRegression(), MAPS[mapName]()).fit(data),
    [data, mapName],
  )
  const xs = useMemo(() => grid(-3, 3, 121), [])
  const fit = useMemo((): XYSeries[] => {
    const input = fromData(Float64Array.from(xs), [xs.length, 1])
    const d = model.predictive(input) as UnivariateDistribution
    const q = (p: number) => toFlat(asTensor(d.quantile(fromData(Float64Array.of(p), []))))
    const x = toFlat(data.x)
    return [
      { name: 'data', type: 'scatter', x, y: toFlat(data.y), muted: true },
      { name: '5% and 95% quantiles', type: 'line', x: xs, y: q(0.05), slot: 2, dashed: true },
      { name: '5% and 95% quantiles', type: 'line', x: xs, y: q(0.95), slot: 2, dashed: true },
      { name: 'decide(x): inverted point prediction', type: 'line', x: xs, y: toFlat(model.decide(input)), slot: 0 },
      { name: 'E[y | x]', type: 'line', x: xs, y: toFlat(model.expect(input)), slot: 1 },
    ]
  }, [model, data, xs])
  const ys = useMemo(() => grid(0.01, 12, 300), [])
  const density = useMemo((): XYSeries[] => {
    const d = model.predictive(fromData(Float64Array.of(at), [1, 1])) as UnivariateDistribution
    const p = ys.map((v) => Math.exp(toFlat(asTensor(d.logProb(fromData(Float64Array.of(v), []))))[0]))
    return [{ name: `p(y | x = ${formatValue(at)})`, type: 'area', x: ys, y: p, slot: 1 }]
  }, [model, at, ys])
  const d0 = model.predictive(fromData(Float64Array.of(at), [1, 1])) as UnivariateDistribution
  return (
    <Figure
      title="A transformed target's predictive distribution"
      defaultSize="L"
      controls={
        <>
          <Select
            label="target map"
            value={mapName}
            onChange={setMapName}
            options={['log', 'log1p', 'box-cox', 'none']}
          />
          <Slider label="noise σ on log y" value={sigma} min={0.05} max={1} onChange={setSigma} />
          <Slider label="x for the density" value={at} min={-3} max={3} onChange={setAt} />
        </>
      }
      readouts={
        <>
          <Readout label="predictive" value={d0.name} />
          <Readout
            label={`median at x = ${formatValue(at)}`}
            value={formatValue(toFlat(model.decide(fromData(Float64Array.of(at), [1, 1])))[0])}
          />
          <Readout label="mean" value={formatValue(toFlat(asTensor(d0.mean()))[0])} />
          <Readout label="training log loss" value={formatValue(evaluate(model, data, [logLoss])['log-loss'])} />
        </>
      }
      caption="transformTarget fits linearRegression on g(y) and pushes its Gaussian predictive through g⁻¹: log gives a log-normal, other maps a general transformed distribution. decide(x) is the inverted point prediction (the median), below the mean E[y | x] for a right-skewed law. Without a map the Gaussian predictive puts mass on negative y and the log loss is worse."
    >
      <ChartSize scale={0.6}>
        <XYChart series={fit} xLabel="x" yLabel="y" xRange={[-3, 3]} yRange={[0, 12]} />
      </ChartSize>
      <ChartSize scale={0.4}>
        <XYChart series={density} xLabel="y" yLabel="density" xRange={[0, 12]} />
      </ChartSize>
    </Figure>
  )
}
