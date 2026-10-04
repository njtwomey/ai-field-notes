import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { sinkhorn } from '../_shared/ot'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream } from 'aifn/foundation/random'

const N = 24
const ITERS = 200
const SCALES = toFlat(linspace(0.2, 1.5, 27))
const W = new Array(N).fill(1 / N)
const ALPHA = (() => {
  const r = stream(5)
  return Array.from({ length: N }, () => normal(r)).sort((a, b) => a - b)
})()
const MEAN_SQ = ALPHA.reduce((s, x) => s + x * x, 0) / N

const cost1d = (x: number[], y: number[]) => x.map((xi) => y.map((yj) => (xi - yj) ** 2))
const otEps = (x: number[], y: number[], eps: number) => sinkhorn(W, W, cost1d(x, y), eps, ITERS, 1e-7).dual

/**
 * A fixed sample α and its rescaled copy β = s·α. The entropic cost OT_ε(α, β) is smallest for a shrunken β; the
 * Sinkhorn divergence, which subtracts the self-transport terms, is smallest at s = 1 for every ε.
 */
export function EntropicBias() {
  const state = useFigureState({
    eps: float(0.3, { min: 0.1, max: 3, scale: 'log10', suggestions: [0.1, 0.3, 1, 3], label: 'ε' }),
    scale: slider(0.2, 1.5, 1, { step: 0.05, label: 'scale s' }),
  })
  const eps = state.eps
  const scale = { value: state.scale }

  const curves = useMemo(() => {
    const selfA = otEps(ALPHA, ALPHA, eps)
    const ot: number[] = []
    const div: number[] = []
    for (const s of SCALES) {
      const beta = ALPHA.map((x) => s * x)
      const v = otEps(ALPHA, beta, eps)
      ot.push(v)
      div.push(v - 0.5 * selfA - 0.5 * otEps(beta, beta, eps))
    }
    const argmin = (ys: number[]) => SCALES[ys.indexOf(Math.min(...ys))]
    return { ot, div, selfA, argOt: argmin(ot), argDiv: argmin(div) }
  }, [eps])

  const series = useMemo(
    () =>
      [
        { name: 'exact W₂²', x: SCALES, y: SCALES.map((s) => (1 - s) ** 2 * MEAN_SQ), slot: 2 },
        { name: 'OT_ε(α, β)', x: SCALES, y: curves.ot, slot: 0 },
        { name: 'Sinkhorn divergence S_ε(α, β)', x: SCALES, y: curves.div, slot: 1 },
      ] as const,
    [curves],
  )

  const samples = useMemo(
    () =>
      [
        { name: 'α', x: ALPHA, y: ALPHA.map(() => 1), slot: 0 },
        { name: 'β = s·α', x: ALPHA.map((x) => scale.value * x), y: ALPHA.map(() => 0), slot: 1 },
      ] as const,
    [scale.value],
  )

  const i = Math.round((scale.value - SCALES[0]) / 0.05)

  const xAxis = useAxis({ label: 'scale s of β', range: [0.2, 1.5] })
  const yAxis = useAxis({ label: 'cost', range: [undefined, undefined], hold: 'union' })
  const xAxis2 = useAxis({ label: 'x', range: [-4, 4] })
  const yAxis2 = useAxis({ range: [-0.5, 1.5] })
  return (
    <Figure
      title="The entropic bias and its correction"
      state={state}
      caption="β is the sample α scaled by s. Change ε, and drag the vertical line (or use the slider) to pick s. The entropic cost OT_ε is not zero at s = 1 and is minimised by a shrunken β, more so for larger ε. The Sinkhorn divergence is zero at s = 1 and minimised there for every ε, like the exact cost W₂²."
      readouts={
        <>
          <Readout label="OT_ε(α, α)" value={formatNumber(curves.selfA)} />
          <Readout label="argmin of OT_ε" value={`s = ${formatNumber(curves.argOt)}`} />
          <Readout label="argmin of S_ε" value={`s = ${formatNumber(curves.argDiv)}`} />
          <Readout label="S_ε at this s" value={formatNumber(curves.div[i] ?? NaN)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[3fr_2fr]">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Curve {...series[0]} />
          <Curve {...series[1]} />
          <Curve {...series[2]} />
          <Handle {...state.handle('scale', { label: 's' })} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300} bare={false} ariaLabel={'The two samples'}>
          <Points {...samples[0]} />
          <Points {...samples[1]} />
        </Plot>
      </div>
    </Figure>
  )
}
