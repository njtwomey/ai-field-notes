import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { sinkhorn } from '../_shared/ot'

const N = 24
const ITERS = 200
const SCALES = linspace(0.2, 1.5, 27)
const W = new Array(N).fill(1 / N)
const ALPHA = (() => {
  const r = rng(5)
  return Array.from({ length: N }, () => r.normal()).sort((a, b) => a - b)
})()
const MEAN_SQ = ALPHA.reduce((s, x) => s + x * x, 0) / N

const cost1d = (x: number[], y: number[]) => x.map((xi) => y.map((yj) => (xi - yj) ** 2))
const otEps = (x: number[], y: number[], eps: number) => sinkhorn(W, W, cost1d(x, y), eps, ITERS, 1e-7).dual

/**
 * A fixed sample α and its rescaled copy β = s·α. The entropic cost OT_ε(α, β) is smallest for a shrunken β; the
 * Sinkhorn divergence, which subtracts the self-transport terms, is smallest at s = 1 for every ε.
 */
export function EntropicBias() {
  const logEps = useParam(-0.5, { min: -1, max: 0.5, step: 0.05 })
  const scale = useParam(1, { min: 0.2, max: 1.5, step: 0.05 })
  const eps = 10 ** logEps.value

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
    (): XYSeries[] => [
      { name: 'exact W₂²', type: 'line', x: SCALES, y: SCALES.map((s) => (1 - s) ** 2 * MEAN_SQ), slot: 2 },
      { name: 'OT_ε(α, β)', type: 'line', x: SCALES, y: curves.ot, slot: 0 },
      { name: 'Sinkhorn divergence S_ε(α, β)', type: 'line', x: SCALES, y: curves.div, slot: 1 },
    ],
    [curves],
  )

  const samples = useMemo(
    (): XYSeries[] => [
      { name: 'α', type: 'scatter', x: ALPHA, y: ALPHA.map(() => 1), slot: 0 },
      { name: 'β = s·α', type: 'scatter', x: ALPHA.map((x) => scale.value * x), y: ALPHA.map(() => 0), slot: 1 },
    ],
    [scale.value],
  )

  const handles: Handle[] = [{ kind: 'x', at: scale.value, onDrag: scale.set, label: 's' }]
  const i = Math.round((scale.value - SCALES[0]) / 0.05)

  return (
    <Interactive
      title="The entropic bias and its correction"
      caption="β is the sample α scaled by s. Change ε, and drag the vertical line (or use the slider) to pick s. The entropic cost OT_ε is not zero at s = 1 and is minimised by a shrunken β, more so for larger ε. The Sinkhorn divergence is zero at s = 1 and minimised there for every ε, like the exact cost W₂²."
      controls={
        <>
          <ParamSlider label={`log₁₀ ε (ε = ${formatNumber(eps)})`} param={logEps} />
          <ParamSlider label="scale s" param={scale} />
        </>
      }
      readout={
        <>
          <Readout label="OT_ε(α, α)" value={formatNumber(curves.selfA)} />
          <Readout label="argmin of OT_ε" value={`s = ${formatNumber(curves.argOt)}`} />
          <Readout label="argmin of S_ε" value={`s = ${formatNumber(curves.argDiv)}`} />
          <Readout label="S_ε at this s" value={formatNumber(curves.div[i] ?? NaN)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[3fr_2fr]">
        <XYChart
          height={300}
          series={series}
          xRange={[0.2, 1.5]}
          yRange={[undefined, undefined]}
          xLabel="scale s of β"
          yLabel="cost"
          handles={handles}
        />
        <XYChart
          height={300}
          series={samples}
          xRange={[-4, 4]}
          yRange={[-0.5, 1.5]}
          xLabel="x"
          bare={false}
          ariaLabel="The two samples"
        />
      </div>
    </Interactive>
  )
}
