import { useMemo } from 'react'
import {
  Bars,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'

const log2 = (x: number) => Math.log(x) / Math.LN2
/** The true distribution: four outcomes with probabilities 1/2, 1/4, 1/8, 1/8, so H(p) = 1.75 bits. */
const P = [0.5, 0.25, 0.125, 0.125]
const OUTCOMES = [1, 2, 3, 4]
const H_P = -P.reduce((s, p) => s + p * log2(p), 0)

/** Model q = p tempered by T: q ∝ p^(1/T). T = 1 is exact, large T flattens towards uniform, small T sharpens. */
function model(t: number): number[] {
  const w = P.map((p) => p ** (1 / t))
  const total = w.reduce((a, b) => a + b, 0)
  return w.map((x) => x / total)
}
const crossEntropy = (q: number[]) => -P.reduce((s, p, i) => s + p * log2(q[i]), 0)

const T_GRID = Array.from({ length: 121 }, (_, i) => 0.25 * Math.pow(16, i / 120))
const CURVE = T_GRID.map((t) => crossEntropy(model(t)))

/** Cross-entropy of a tempered model against the true distribution, split into entropy plus KL divergence. */
export function CrossEntropyGap() {
  const state = useFigureState({
    temperature: float(2, { min: 0.25, max: 4, step: 0.05, label: 'temperature T' }),
  })
  const q = useMemo(() => model(state.temperature), [state.temperature])
  const hpq = crossEntropy(q)

  const bars = [
    { name: 'model q', x: OUTCOMES, y: q, slot: 0 },
    { name: 'true p', x: OUTCOMES, y: P, emphasis: true },
  ] as const
  const curve = [
    { name: 'cross-entropy H(p, q)', x: T_GRID, y: CURVE, slot: 0 },
    { name: 'entropy H(p)', x: [0.25, 4], y: [H_P, H_P], dashed: true, slot: 1 },
    { name: 'current model', x: [state.temperature], y: [hpq], emphasis: true },
  ] as const
  // The temperature is the horizontal position on the curve, so dragging along the axis sets it.

  const xAxis = useAxis({ label: 'outcome', range: [0.5, 4.5] })
  const yAxis = useAxis({ label: 'probability', range: [0, 1] })
  const xAxis2 = useAxis({ label: 'temperature T', range: [0.25, 4] })
  const yAxis2 = useAxis({ label: 'bits', range: [1.5, 3.2] })
  return (
    <Figure
      title="Cross-entropy is entropy plus KL divergence"
      state={state}
      caption="The true distribution p has probabilities 1/2, 1/4, 1/8, 1/8, so its entropy is 1.75 bits. The model q is p tempered by T, q ∝ p^(1/T): T = 1 is exact, larger T flattens q towards uniform and smaller T sharpens it. Left: p (diamonds) against q (bars). Right: the cross-entropy H(p, q) against T. It never falls below H(p), and the gap is KL(p ‖ q). Drag T or use the slider."

      readouts={
        <>
          <Readout label="H(p)" value={`${formatNumber(H_P)} bits`} />
          <Readout label="KL(p ‖ q)" value={`${formatNumber(hpq - H_P)} bits`} />
          <Readout label="H(p, q)" value={`${formatNumber(hpq)} bits`} />
          <Readout label="perplexity of q" value={formatNumber(2 ** hpq)} />
          <Readout label="perplexity of p" value={formatNumber(2 ** H_P)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Bars {...bars[0]} />
          <Points {...bars[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          <Curve {...curve[0]} />
          <Curve {...curve[1]} />
          <Points {...curve[2]} />
          <Handle {...state.handle('temperature', { label: 'T' })} />
        </Plot>
      </div>
    </Figure>
  )
}
