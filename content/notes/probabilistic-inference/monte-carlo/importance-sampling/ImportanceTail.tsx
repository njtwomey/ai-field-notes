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
import { normalCdf } from '@/lib/math/special'

const N = 1000
const XS = linspace(-4, 8, 481)
const pdf = (x: number, m: number, s: number) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI))

/**
 * Importance sampling for the tail probability P(X > t), X ~ N(0, 1), with a Gaussian proposal N(m, s²). The exact
 * per-sample variance ∫_t^∞ p²/q dx − P² is integrated on a grid; it is infinite when s² ≤ 1/2, because p²/q then grows
 * in the tail.
 */
export function ImportanceTail() {
  const t = useParam(3, { min: 1, max: 5, step: 0.05 })
  const m = useParam(3, { min: -1, max: 6, step: 0.05 })
  const s = useParam(1, { min: 0.4, max: 3, step: 0.05 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })

  const r = useMemo(() => {
    const truth = 1 - normalCdf(t.value)
    let second = Infinity
    if (s.value ** 2 > 0.5) {
      const grid = linspace(t.value, t.value + 20, 4001)
      const h = grid[1] - grid[0]
      second = 0
      grid.forEach((x, i) => {
        const w = i === 0 || i === grid.length - 1 ? 0.5 : 1
        second += (w * h * pdf(x, 0, 1) ** 2) / pdf(x, m.value, s.value)
      })
    }
    const varIs = second - truth ** 2
    const varNaive = truth * (1 - truth)
    // One simulated run of N draws from the proposal.
    const g = rng(seed.value)
    let sumFw = 0
    let sumW = 0
    let sumW2 = 0
    for (let i = 0; i < N; i++) {
      const x = m.value + s.value * g.normal()
      const w = pdf(x, 0, 1) / pdf(x, m.value, s.value)
      if (x > t.value) sumFw += w
      sumW += w
      sumW2 += w * w
    }
    return {
      truth,
      relNaive: Math.sqrt(varNaive / N) / truth,
      relIs: Math.sqrt(Math.max(varIs, 0) / N) / truth,
      ratio: varNaive / varIs,
      estimate: sumFw / N,
      ess: (sumW * sumW) / sumW2,
    }
  }, [t.value, m.value, s.value, seed.value])

  const series: XYSeries[] = useMemo(() => {
    const tailMass = 1 - normalCdf(t.value)
    return [
      { name: 'target p = N(0, 1)', type: 'line', x: XS, y: XS.map((x) => pdf(x, 0, 1)), slot: 0 },
      { name: 'proposal q', type: 'line', x: XS, y: XS.map((x) => pdf(x, m.value, s.value)), slot: 1, area: true },
      {
        name: 'optimal proposal ∝ f p',
        type: 'line',
        x: XS,
        y: XS.map((x) => (x > t.value ? pdf(x, 0, 1) / tailMass : 0)),
        slot: 2,
        dashed: true,
      },
    ]
  }, [t.value, m.value, s.value])

  const handles: Handle[] = [{ kind: 'x', at: t.value, label: 'threshold t', onDrag: t.set }]
  const infinite = !Number.isFinite(r.ratio) || r.ratio <= 0

  return (
    <Interactive
      title="Importance sampling a tail probability"
      caption={`The target is X ~ N(0, 1) and the quantity is P(X > t); drag the threshold or use its slider. Plain Monte Carlo sees the tail only through the few samples that land there. The proposal q = N(m, s²) puts samples in the tail and each is weighted by p/q. Readouts use N = ${N}. Moving m near t cuts the relative error by an order of magnitude. Shrinking s below 1/√2 ≈ 0.71 makes the variance infinite: the weights p/q then grow without bound in the far tail, although a single run can still look accurate. The effective sample size judges q as an approximation to p, not to the optimal proposal, so it is small exactly when the tail estimate is good.`}
      controls={
        <>
          <ParamSlider label="threshold t" param={t} />
          <ParamSlider label="proposal mean m" param={m} />
          <ParamSlider label="proposal standard deviation s" param={s} />
          <ParamSlider label="random seed" param={seed} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="P(X > t)" value={r.truth.toExponential(3)} />
          <Readout label="IS estimate, one run" value={r.estimate.toExponential(3)} />
          <Readout label="relative error, plain MC" value={formatNumber(r.relNaive)} />
          <Readout label="relative error, IS" value={infinite ? '∞' : formatNumber(r.relIs)} />
          <Readout label="variance ratio" value={infinite ? '0' : formatNumber(r.ratio)} />
          <Readout label="ESS of the weights" value={formatNumber(r.ess)} />
        </>
      }
    >
      <XYChart series={series} xLabel="x" yLabel="density" handles={handles} height={300} />
    </Interactive>
  )
}
