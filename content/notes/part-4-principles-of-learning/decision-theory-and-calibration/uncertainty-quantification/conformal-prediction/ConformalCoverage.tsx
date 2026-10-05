import { useMemo } from 'react'
import {
  Area,
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
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'
import { logGamma } from 'aifn-compute/numerics/special'

const XS = toFlat(linspace(0, 6, 121))
const TEST = 3000
const mean = (x: number) => Math.sin(x) + 0.3 * x
const noise = (x: number) => 0.15 + 0.12 * x
const COV = toFlat(linspace(0.5, 1, 301))

/**
 * Split conformal prediction with the absolute residual score. The regressor is the true mean function (a stand-in
 * for any fitted model); noise grows with x. The interval is f(x) ± q̂, where q̂ is the ⌈(n + 1)(1 − α)⌉-th smallest
 * calibration residual. The right chart is the Beta(n + 1 − l, l) law of the coverage given the calibration set.
 */
export function ConformalCoverage() {
  const state = useFigureState({
    alpha: float(0.1, { min: 0.02, max: 0.3, step: 0.01, label: 'miscoverage α' }),
    n: int(100, { min: 10, max: 2000, step: 10, label: 'calibration points n', format: (v) => String(v) }),
    seed: int(1, { min: 1, max: 50, step: 1, label: 'calibration seed', format: (v) => String(v) }),
  })

  const test = useMemo(() => {
    const g = stream(999)
    return Array.from({ length: TEST }, () => {
      const x = 6 * uniform(g)
      return { x, y: mean(x) + noise(x) * normal(g) }
    })
  }, [])

  const r = useMemo(() => {
    const g = stream(state.seed)
    const scores: number[] = []
    for (let i = 0; i < state.n; i++) {
      const x = 6 * uniform(g)
      scores.push(Math.abs(noise(x) * normal(g)))
    }
    scores.sort((a, b) => a - b)
    const k = Math.ceil((state.n + 1) * (1 - state.alpha))
    const q = k > state.n ? Infinity : scores[k - 1]
    const inside = (p: { x: number; y: number }) => Math.abs(p.y - mean(p.x)) <= q
    const low = test.filter((p) => p.x < 2)
    const high = test.filter((p) => p.x >= 4)
    const l = Math.floor((state.n + 1) * state.alpha)
    // Beta(n + 1 − l, l) density of the conditional coverage; undefined when l = 0 (the interval is infinite).
    const aB = state.n + 1 - l
    const bB = l
    const logB = logGamma(aB) + logGamma(bB) - logGamma(aB + bB)
    const density =
      l > 0 ? COV.map((c) => Math.exp((aB - 1) * Math.log(c) + (bB - 1) * Math.log(1 - c) - logB)) : COV.map(() => 0)
    return {
      q,
      k,
      l,
      coverage: test.filter(inside).length / TEST,
      coverLow: low.filter(inside).length / low.length,
      coverHigh: high.filter(inside).length / high.length,
      density: density.map((d) => (Number.isFinite(d) ? d : 0)),
    }
  }, [state.alpha, state.n, state.seed, test])

  const band = Number.isFinite(r.q) ? r.q : 10
  const top = Math.max(...r.density, 1) * 1.1

  const xAxis = useAxis({ label: 'x', range: [0, 6] })
  const yAxis = useAxis({ label: 'y', range: [-2.5, 4.5] })
  const xAxis2 = useAxis({ label: 'coverage given the calibration set', range: [0.5, 1] })
  const yAxis2 = useAxis({ label: 'density', range: [0, top] })
  return (
    <Figure
      title="Split conformal intervals and their coverage"
      state={state}
      caption="Left: test points and the conformal interval f(x) ± q̂, where q̂ is the ⌈(n + 1)(1 − α)⌉-th smallest absolute residual on n calibration points. Right: the distribution of the coverage you get for a random calibration set, Beta(n + 1 − l, l) with l = ⌊(n + 1)α⌋; the dashed line is the target 1 − α and the dot the coverage of this calibration set on 3000 test points. Change the seed to redraw the calibration set. Coverage holds on average over x but not in each region: the noise grows with x."

      readouts={
        <>
          <Readout label="q̂" value={Number.isFinite(r.q) ? formatNumber(r.q) : '∞'} />
          <Readout label="rank used" value={`${r.k} of ${state.n}`} />
          <Readout label="test coverage" value={formatNumber(r.coverage)} />
          <Readout label="coverage, x < 2" value={formatNumber(r.coverLow)} />
          <Readout label="coverage, x ≥ 4" value={formatNumber(r.coverHigh)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Points
            name="test points"
            x={test.slice(0, 600).map((p) => p.x)}
            y={test.slice(0, 600).map((p) => p.y)}
            muted
          />
          <Curve name="model f(x)" x={XS} y={XS.map(mean)} emphasis />
          <Curve name="upper f(x) + q̂" x={XS} y={XS.map((x) => mean(x) + band)} slot={0} />
          <Curve name="lower f(x) − q̂" x={XS} y={XS.map((x) => mean(x) - band)} slot={0} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          <Area name="Beta(n + 1 − l, l)" x={COV} y={r.density} slot={1} />
          <Curve name="target 1 − α" x={[1 - state.alpha, 1 - state.alpha]} y={[0, top]} dashed muted />
          <Points name="this calibration set" x={[r.coverage]} y={[0]} emphasis />
        </Plot>
      </div>
    </Figure>
  )
}
