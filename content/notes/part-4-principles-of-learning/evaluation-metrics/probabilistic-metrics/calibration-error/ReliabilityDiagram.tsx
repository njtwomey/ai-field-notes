import { useMemo } from 'react'
import {
  Bars,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))

/**
 * A simulated binary classifier with known true probabilities q = σ(z), z ~ N(0, 1.5²), labels y ~ Bernoulli(q), and
 * reported probabilities p = σ(k z + b): k > 1 is overconfident, k < 1 underconfident, b shifts every prediction. The
 * reliability diagram bins p and compares the mean prediction with the observed frequency in each bin.
 */
export function ReliabilityDiagram() {
  const state = useFigureState({
    k: float(2, { min: 0.3, max: 3, step: 0.05, label: 'sharpness k' }),
    b: float(0, { min: -2, max: 2, step: 0.05, label: 'bias b' }),
    bins: int(10, { min: 3, max: 20, step: 1, label: 'bins', format: (v) => String(v) }),
    n: int(2000, { min: 200, max: 5000, step: 100, label: 'cases', format: (v) => String(v) }),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const r = useMemo(() => {
    const g = stream(state.seed)
    const cases = Array.from({ length: state.n }, () => {
      const z = 1.5 * normal(g)
      const y = uniform(g) < sigmoid(z) ? 1 : 0
      return { p: sigmoid(state.k * z + state.b), y }
    })
    const m = state.bins
    const sumP = new Array<number>(m).fill(0)
    const sumY = new Array<number>(m).fill(0)
    const count = new Array<number>(m).fill(0)
    let logLoss = 0
    let brier = 0
    for (const { p, y } of cases) {
      const j = Math.min(m - 1, Math.floor(p * m))
      sumP[j] += p
      sumY[j] += y
      count[j] += 1
      const pc = Math.min(Math.max(p, 1e-12), 1 - 1e-12)
      logLoss -= y * Math.log(pc) + (1 - y) * Math.log(1 - pc)
      brier += (p - y) ** 2
    }
    let ece = 0
    let mce = 0
    const centres: number[] = []
    const observed: number[] = []
    const predicted: number[] = []
    for (let j = 0; j < m; j++) {
      if (!count[j]) continue
      const gap = Math.abs(sumP[j] / count[j] - sumY[j] / count[j])
      ece += (count[j] / cases.length) * gap
      mce = Math.max(mce, gap)
      centres.push((j + 0.5) / m)
      observed.push(sumY[j] / count[j])
      predicted.push(sumP[j] / count[j])
    }
    return { ece, mce, logLoss: logLoss / cases.length, brier: brier / cases.length, centres, observed, predicted }
  }, [state.k, state.b, state.bins, state.n, state.seed])

  const series = [
    { name: 'perfect calibration', x: [0, 1], y: [0, 1], dashed: true, muted: true },
    { name: 'observed frequency', x: r.centres, y: r.observed, slot: 0 },
    { name: 'mean prediction in bin', x: r.centres, y: r.predicted, emphasis: true },
  ] as const

  const xAxis = useAxis({ label: 'predicted probability', range: [0, 1] })
  const yAxis = useAxis({ label: 'frequency of positives', range: [0, 1], equal: xAxis })
  return (
    <Figure
      title="Reliability diagram and expected calibration error"
      state={state}
      caption="A classifier reports p = σ(k·z + b) where the true probability is σ(z). Bars show the observed frequency of positives in each bin of p; diamonds show the mean prediction in the bin. Where they differ the classifier is miscalibrated. k > 1 makes it overconfident (bars flatter than the diagonal), k < 1 underconfident, and b biases every prediction. At k = 1, b = 0 the model is calibrated and the remaining ECE is sampling noise, which grows with more bins and fewer cases."

      readouts={
        <>
          <Readout label="ECE" value={formatNumber(r.ece)} />
          <Readout label="MCE" value={formatNumber(r.mce)} />
          <Readout label="log loss" value={formatNumber(r.logLoss)} />
          <Readout label="Brier score" value={formatNumber(r.brier)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-xl">
        <Plot x={xAxis} y={yAxis}>
          <Curve {...series[0]} />
          <Bars {...series[1]} />
          <Points {...series[2]} />
        </Plot>
      </div>
    </Figure>
  )
}
