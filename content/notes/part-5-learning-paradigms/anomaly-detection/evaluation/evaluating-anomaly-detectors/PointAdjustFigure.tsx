import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { Binomial } from 'aifn-compute/probability/distributions'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

/** Series length and number of labelled anomalous segments. */
const N = 10000
const SEGMENTS = 5
const RATES = toFlat(linspace(0.0005, 0.2, 400))

const f1 = (precision: number, recall: number) =>
  precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0

/**
 * Expected precision, recall and F1 of a detector that flags each point independently with probability p, scored
 * point-wise, with point adjustment (PA), and with PA%K. Under PA%K a segment counts as fully detected when at least
 * K% of its L points are flagged; otherwise only its flagged points count.
 */
function expectedScores(L: number, p: number, K: number) {
  const positives = SEGMENTS * L
  const fp = p * (N - positives)
  const need = Math.max(1, Math.ceil((K / 100) * L))
  // E[TP per segment] = Σ_d P(D = d)·(L if d ≥ need else d), with D ~ Binom(L, p) flagged points.
  let tpK = 0
  const flagged = Binomial(L, p)
  for (let d = 0; d <= L; d++) tpK += flagged.prob(d) * (d >= need ? L : d)
  const detected = 1 - Math.pow(1 - p, L)
  const tpPa = L * detected
  const point = f1(positives / N, p)
  const pa = f1((SEGMENTS * tpPa) / (SEGMENTS * tpPa + fp), tpPa / L)
  const paK = f1((SEGMENTS * tpK) / (SEGMENTS * tpK + fp), tpK / L)
  return { point, pa, paK }
}

export function PointAdjustFigure() {
  const state = useFigureState({
    L: int(100, {
      min: 1,
      max: 400,
      step: 1,
      suggestions: [10, 50, 100, 400],
      label: 'segment length L',
      format: (v) => String(v),
    }),
    p: float(0.05, {
      gt: 0,
      max: 0.2,
      scale: 'log10',
      suggestions: [0.001, 0.01, 0.05, 0.1],
      label: 'flag rate p',
      format: (v) => v.toFixed(4),
    }),
    K: slider(0, 100, 20, { step: 5, label: 'PA%K threshold K (%)', format: (v) => String(v) }),
  })

  const curves = useMemo(() => RATES.map((r) => expectedScores(state.L, r, state.K)), [state.L, state.K])
  const series = useMemo(
    () =>
      [
        { name: 'point-wise F1', x: RATES, y: curves.map((c) => c.point), slot: 0 },
        { name: 'point-adjusted F1', x: RATES, y: curves.map((c) => c.pa), slot: 1 },
        { name: `PA%K F1 (K = ${state.K}%)`, x: RATES, y: curves.map((c) => c.paK), slot: 2 },
      ] as const,
    [curves, state.K],
  )
  const at = expectedScores(state.L, state.p, state.K)

  const xAxis = useAxis({ label: 'flag rate p', range: [0, 0.2] })
  const yAxis = useAxis({ label: 'expected F1', range: [0, 1] })
  return (
    <Figure
      title="A random detector under point adjustment"
      state={state}
      caption={`A series of ${N.toLocaleString()} points holds ${SEGMENTS} anomalous segments of L points each. The "detector" ignores the data and flags each point independently with probability p. The curves are its F1, computed from expected counts, as p varies. Point-wise, a random detector scores little. Point adjustment credits a whole segment once any point in it is flagged, so long segments hand the random detector a high F1. PA%K credits the segment only when at least K% of it is flagged. Drag the vertical line to set p.`}

      readouts={
        <>
          <Readout label="anomalous share" value={formatNumber((SEGMENTS * state.L) / N)} />
          <Readout label="point-wise F1" value={formatNumber(at.point)} />
          <Readout label="point-adjusted F1" value={formatNumber(at.pa)} />
          <Readout label="PA%K F1" value={formatNumber(at.paK)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Handle {...state.handle('p', { label: 'p' })} />
      </Plot>
    </Figure>
  )
}
