import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { normalQuantile } from 'aifn/numerics/special'
import { tTestPower } from 'aifn/probability/tests'

const N_MIN = 5
const N_MAX = 500

/** Smallest n per group whose exact two-sample t-test power reaches `target` for effect d (binary search on n). */
function exactN(d: number, alpha: number, target: number): number | undefined {
  const power = (n: number) => tTestPower(d * Math.sqrt(n / 2), 2 * n - 2, alpha)
  if (d <= 0 || power(N_MAX * 4) < target) return undefined
  let lo = 2
  let hi = N_MAX * 4
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2)
    if (power(mid) >= target) hi = mid
    else lo = mid + 1
  }
  return lo
}

/**
 * Minimum detectable effect of a two-sided two-sample comparison of means with n per group, in standard deviations:
 * MDE = (z₁₋α/₂ + z₁₋β)·√(2/n). The readouts also invert it for a target effect, by the same approximation and by
 * exact t-test power.
 */
export function Planner() {
  const state = useFigureState({
    n: int(64, { min: N_MIN, max: N_MAX, step: 1, label: 'n per group' }),
    target: float(0.5, { min: 0.05, max: 1.5, step: 0.05, label: 'target effect d' }),
    alpha: float(0.05, { min: 0.005, max: 0.2, step: 0.005, label: 'significance level α' }),
    power: float(0.8, { min: 0.5, max: 0.99, step: 0.01, label: 'power 1 − β' }),
  })

  const result = useMemo(() => {
    const k = normalQuantile(1 - state.alpha / 2) + normalQuantile(state.power)
    const ns = Array.from({ length: N_MAX - N_MIN + 1 }, (_, i) => i + N_MIN)
    const series = [
      { name: 'minimum detectable effect', x: ns, y: ns.map((m) => k * Math.sqrt(2 / m)), slot: 0 },
      { name: 'target effect', x: [N_MIN, N_MAX], y: [state.target, state.target], slot: 1, dashed: true },
    ] as const
    return {
      k,
      series,
      approx: Math.ceil((2 * k * k) / (state.target * state.target)),
      exact: exactN(state.target, state.alpha, state.power),
    }
  }, [state.alpha, state.power, state.target])

  const mde = result.k * Math.sqrt(2 / state.n)

  const xAxis = useAxis({ label: 'n per group', range: [0, N_MAX] })
  const yAxis = useAxis({ label: 'effect d (standard deviations)', range: [0, 2] })
  return (
    <Figure
      title="Minimum detectable effect against sample size"
      state={state}
      caption="The curve is the smallest standardised difference in means that a two-sided two-sample test detects with the chosen power, for n per group. It falls like 1/√n. Where it crosses the dashed target effect is the sample size needed. Drag the line labelled n, or use its slider, to read the effect detectable with that many observations."

      readouts={
        <>
          <Readout label="MDE at this n" value={formatNumber(mde)} />
          <Readout label="n per group, normal approx." value={result.approx} />
          <Readout label="n per group, exact t" value={result.exact ?? `> ${N_MAX * 4}`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...result.series[0]} />
        <Curve {...result.series[1]} />
        <Handle kind="x" at={state.n} label="n" onDrag={(x) => state.set('n', Math.round(x))} />
      </Plot>
    </Figure>
  )
}
