import { useMemo } from 'react'
import {
  choice,
  Figure,
  float,
  formatNumber,
  Plot,
  Points,
  Raster,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal as drawNormal, stream, uniform as drawUniform } from 'aifn/foundation/random'
import {
  auc,
  embed,
  initNet,
  initialCentre,
  normalData,
  scoreGrid,
  sqDist,
  train,
  uniformAnomalies,
  type Point,
  type Shape,
} from '../_shared/deepOneClass'
import { linspace, toFlat } from 'aifn/foundation/tensor'

/** Uniform and normal draws from one aifn stream, in the shape the shared helpers take. */
const rand = (seed: number) => {
  const s = stream(seed)
  return { uniform: () => drawUniform(s), normal: () => drawNormal(s) }
}

const LIM = 4
const AXIS = toFlat(linspace(-LIM, LIM, 41))
const EPOCHS = 400
const EVERY = 20
const SIZES = [2, 16, 16, 2]
// Colour range of log₁₀ s(x). Fixed, so that a collapsing network visibly fades to a flat map.
const LOG_RANGE: [number, number] = [-5, 1]

type Variant = 'fixed' | 'bias' | 'centre'
const VARIANTS = [
  { value: 'fixed' as const, label: 'no bias, fixed c' },
  { value: 'bias' as const, label: 'bias terms' },
  { value: 'centre' as const, label: 'learned c' },
]
const SHAPES = [
  { value: 'ring' as const, label: 'ring' },
  { value: 'blobs' as const, label: 'two blobs' },
]
type Kind = 'one-class' | 'soft-boundary'
const KINDS = [
  { value: 'one-class' as const, label: 'one-class' },
  { value: 'soft-boundary' as const, label: 'soft-boundary' },
]

const log10 = (v: number) => Math.log10(Math.max(v, 1e-12))
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length

/** Deep SVDD trained in the browser on 2-D data, with the two routes to hypersphere collapse. */
export function DeepSvddFigure() {
  const state = useFigureState({
    shape: choice<Shape>(SHAPES, 'ring', { label: 'data' }),
    variant: choice<Variant>(VARIANTS, 'fixed', { label: 'network' }),
    kind: choice<Kind>(KINDS, 'one-class', { label: 'objective' }),
    epoch: float(EPOCHS, { min: 0, max: EPOCHS, step: EVERY, label: 'epoch', format: (v) => String(v) }),
    nu: float(0.1, {
      min: 0.02,
      max: 0.5,
      step: 0.01,
      label: 'ν',
      format: (v) => v.toFixed(2),
      when: (v) => v.kind === 'soft-boundary',
    }),
  })

  const data = useMemo(() => {
    const r = rand(11)
    const x = normalData(state.shape, 150, r)
    const testNormal = normalData(state.shape, 100, r)
    const testAnomalies = uniformAnomalies(100, LIM, x, 0.5, r)
    return { x, testNormal, testAnomalies }
  }, [state.shape])

  const run = useMemo(() => {
    const net = initNet(SIZES, state.variant === 'bias', rand(7))
    const c = initialCentre(net, data.x)
    const snaps = train(net, data.x, c, {
      objective: state.kind === 'one-class' ? { kind: state.kind } : { kind: state.kind, nu: state.nu },
      learnCentre: state.variant === 'centre',
      epochs: EPOCHS,
      lr: 0.01,
      weightDecay: 0.03,
      every: EVERY,
    })
    // Fix the embedding panel's window from the initial embeddings, so shrinking is visible.
    const e0 = data.x.map((p) => embed(net, p))
    const span = Math.max(...e0.map((e) => Math.max(Math.abs(e[0] - c[0]), Math.abs(e[1] - c[1])))) * 1.1
    return { snaps, span }
  }, [data, state.variant, state.kind, state.nu])

  const snap = run.snaps[Math.round(state.epoch / EVERY)]
  const view = useMemo(() => {
    const { net, c, R } = snap
    const score = (p: Point) => sqDist(embed(net, p), c)
    const z = scoreGrid(net, c, AXIS).map((row) => row.map(log10))
    const sn = data.testNormal.map(score)
    const sa = data.testAnomalies.map(score)
    const train = data.x.map(score)
    const eN = data.x.map((p) => embed(net, p))
    const eA = data.testAnomalies.map((p) => embed(net, p))
    return {
      z,
      eN,
      eA,
      auc: auc(sn, sa),
      meanN: mean(sn),
      meanA: mean(sa),
      outside: train.filter((s) => s > R * R).length / train.length,
    }
  }, [snap, data])

  const overlay = useMemo(
    () =>
      [
        {
          name: 'normal training data',
          x: data.x.map((p) => p[0]),
          y: data.x.map((p) => p[1]),
          slot: 1,
        },
        {
          name: 'test anomalies',
          x: data.testAnomalies.map((p) => p[0]),
          y: data.testAnomalies.map((p) => p[1]),
          slot: 2,
        },
      ] as const,
    [data],
  )

  const series = useMemo((): SeriesSpec[] => {
    const { c, R } = snap
    const out: SeriesSpec[] = [
      {
        name: 'normal training data',
        type: 'scatter',
        x: view.eN.map((e) => e[0]),
        y: view.eN.map((e) => e[1]),
        slot: 1,
      },
      { name: 'test anomalies', type: 'scatter', x: view.eA.map((e) => e[0]), y: view.eA.map((e) => e[1]), slot: 2 },
      { name: 'centre c', type: 'scatter', x: [c[0]], y: [c[1]], emphasis: true },
    ]
    if (state.kind === 'soft-boundary' && R > 0) {
      const t = toFlat(linspace(0, 2 * Math.PI, 97))
      out.push({
        name: 'sphere of radius R',
        type: 'line',
        x: t.map((a) => c[0] + R * Math.cos(a)),
        y: t.map((a) => c[1] + R * Math.sin(a)),
        dashed: true,
        slot: 3,
      })
    }
    return out
  }, [snap, view, state.kind])

  const c0 = run.snaps[0].c
  // Round the window to half units so the axis ends read cleanly.
  const half = Math.ceil((run.span + 0.25) * 2) / 2
  const [cx, cy] = [Math.round(c0[0] * 2) / 2, Math.round(c0[1] * 2) / 2]
  const xRange: [number, number] = [cx - half, cx + half]
  const yRange: [number, number] = [cy - half, cy + half]

  const xAxis = useAxis({ label: 'x₁' })
  const yAxis = useAxis({ label: 'x₂' })
  const xAxis2 = useAxis({ label: 'φ₁(x)', range: xRange })
  const yAxis2 = useAxis({ label: 'φ₂(x)', range: yRange, equal: xAxis2 })
  return (
    <Figure
      title="Deep SVDD in two dimensions, and hypersphere collapse"
      state={state}
      caption="A ReLU network 2 → 16 → 16 → 2 is trained by Adam, with weight decay 0.03, to pull the embeddings of 150 normal points towards a centre c fixed at the mean of their initial embeddings. Left: the anomaly score log₁₀ ‖φ(x) − c‖² over input space on a fixed colour scale, with the training data and 100 held-out uniform anomalies. Right: the embeddings, with c as a diamond. Step through the epochs. With bias terms, or with c learned alongside the weights, the network drifts towards a constant map: every embedding moves onto the centre, the scores shrink by several orders of magnitude and the map fades to one colour. On the ring the test AUC falls as well. The soft-boundary objective adds a radius R, the (1 − ν) quantile of the training distances."

      readouts={
        <>
          <Readout label="loss" value={formatNumber(snap.loss)} />
          <Readout
            label="mean score, normal / anomalies"
            value={`${formatNumber(view.meanN)} / ${formatNumber(view.meanA)}`}
          />
          <Readout label="test AUC" value={view.auc.toFixed(3)} />
          {state.kind === 'soft-boundary' && (
            <Readout
              label="R, training fraction outside"
              value={`${formatNumber(snap.R)}, ${view.outside.toFixed(2)}`}
            />
          )}
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={380}>
          <Raster x={AXIS} y={AXIS} z={view.z} range={LOG_RANGE} valueLabel={'log₁₀ s(x)'} />
          <Points {...overlay[0]} live />
          <Points {...overlay[1]} live />
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          {seriesLayers(series)}
        </Plot>
      </div>
    </Figure>
  )
}
