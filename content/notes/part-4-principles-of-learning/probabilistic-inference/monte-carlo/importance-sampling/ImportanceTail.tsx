import { useMemo } from 'react'
import {
  Area,
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
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream } from 'aifn/foundation/random'
import { normalCdf } from 'aifn/numerics/special'

const N = 1000
const XS = toFlat(linspace(-4, 8, 481))
const pdf = (x: number, m: number, s: number) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI))

/**
 * Importance sampling for the tail probability P(X > t), X ~ N(0, 1), with a Gaussian proposal N(m, s²). The exact
 * per-sample variance ∫_t^∞ p²/q dx − P² is integrated on a grid; it is infinite when s² ≤ 1/2, because p²/q then grows
 * in the tail.
 */
export function ImportanceTail() {
  const state = useFigureState({
    t: slider(1, 5, 3, { step: 0.05, label: 'threshold t' }),
    m: float(3, { min: -1, max: 6, step: 0.05, label: 'proposal mean m' }),
    s: float(1, { min: 0.4, max: 3, step: 0.05, label: 'proposal standard deviation s' }),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'random seed' }),
  })

  const r = useMemo(() => {
    const truth = 1 - normalCdf(state.t)
    let second = Infinity
    if (state.s ** 2 > 0.5) {
      const grid = toFlat(linspace(state.t, state.t + 20, 4001))
      const h = grid[1] - grid[0]
      second = 0
      grid.forEach((x, i) => {
        const w = i === 0 || i === grid.length - 1 ? 0.5 : 1
        second += (w * h * pdf(x, 0, 1) ** 2) / pdf(x, state.m, state.s)
      })
    }
    const varIs = second - truth ** 2
    const varNaive = truth * (1 - truth)
    // One simulated run of N draws from the proposal.
    const g = stream(state.seed)
    let sumFw = 0
    let sumW = 0
    let sumW2 = 0
    for (let i = 0; i < N; i++) {
      const x = state.m + state.s * normal(g)
      const w = pdf(x, 0, 1) / pdf(x, state.m, state.s)
      if (x > state.t) sumFw += w
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
  }, [state.t, state.m, state.s, state.seed])

  const series = useMemo(() => {
    const tailMass = 1 - normalCdf(state.t)
    return [
      { name: 'target p = N(0, 1)', x: XS, y: XS.map((x) => pdf(x, 0, 1)), slot: 0 },
      { name: 'proposal q', x: XS, y: XS.map((x) => pdf(x, state.m, state.s)), slot: 1 },
      {
        name: 'optimal proposal ∝ f p',
        x: XS,
        y: XS.map((x) => (x > state.t ? pdf(x, 0, 1) / tailMass : 0)),
        slot: 2,
        dashed: true,
      },
    ] as const
  }, [state.t, state.m, state.s])

  const infinite = !Number.isFinite(r.ratio) || r.ratio <= 0

  const xAxis = useAxis({ label: 'x', hold: 'union' })
  const yAxis = useAxis({ label: 'density', hold: 'union' })
  return (
    <Figure
      title="Importance sampling a tail probability"
      state={state}
      caption={`The target is X ~ N(0, 1) and the quantity is P(X > t); drag the threshold or use its slider. Plain Monte Carlo sees the tail only through the few samples that land there. The proposal q = N(m, s²) puts samples in the tail and each is weighted by p/q. Readouts use N = ${N}. Moving m near t cuts the relative error by an order of magnitude. Shrinking s below 1/√2 ≈ 0.71 makes the variance infinite: the weights p/q then grow without bound in the far tail, although a single run can still look accurate. The effective sample size judges q as an approximation to p, not to the optimal proposal, so it is small exactly when the tail estimate is good.`}

      readouts={
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
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Area {...series[1]} />
        <Curve {...series[2]} />
        <Handle {...state.handle('t', { label: 'threshold t' })} />
      </Plot>
    </Figure>
  )
}
