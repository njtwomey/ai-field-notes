import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { normalQuantile, studentTCdf } from 'aifn/numerics/special'

const N_MAX = 60

/**
 * The real Type I error rate of a two-sided "z-test" that plugs the sample standard deviation into the z statistic.
 * With normal data the statistic is exactly t with n − 1 degrees of freedom, so its size is P(|T| > z₁₋α/₂).
 */
export function PlugInSize() {
  const state = useFigureState({
    alpha: float(0.05, { min: 0.01, max: 0.1, step: 0.005, label: 'significance level α' }),
    n: int(10, { min: 2, max: N_MAX, step: 1, label: 'sample size n' }),
  })

  const result = useMemo(() => {
    const crit = normalQuantile(1 - state.alpha / 2)
    const size = (m: number) => 2 * (1 - studentTCdf(crit, m - 1))
    const ns = Array.from({ length: N_MAX - 1 }, (_, i) => i + 2)
    const series = [
      { name: 'z-test with s: actual size', x: ns, y: ns.map(size), slot: 0 },
      { name: 't-test: size α', x: [2, N_MAX], y: [state.alpha, state.alpha], slot: 1, dashed: true },
    ] as const
    return { series, size }
  }, [state.alpha])

  const xAxis = useAxis({ label: 'sample size n', hold: 'union' })
  const yAxis = useAxis({ label: 'P(reject | H₀)', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Plugging s into a z-test"
      state={state}
      caption="The z-test assumes the population standard deviation σ is known. Replacing it with the sample standard deviation s and still using the normal critical value rejects a true null too often, because s is itself uncertain. The t-test uses the t distribution and has exactly size α for normal data."

      readouts={
        <>
          <Readout label="actual size at this n" value={formatNumber(result.size(state.n))} />
          <Readout label="inflation" value={`${formatNumber(result.size(state.n) / state.alpha)}×`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={280}>
        <Curve {...result.series[0]} />
        <Curve {...result.series[1]} />
      </Plot>
    </Figure>
  )
}
