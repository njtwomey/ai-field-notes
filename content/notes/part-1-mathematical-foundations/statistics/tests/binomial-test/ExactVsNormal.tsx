import { useMemo } from 'react'
import {
  Bars,
  Curve,
  Figure,
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
import { normalCdf, normalPdf } from 'aifn-compute/numerics/special'

/**
 * Two-sided binomial test of H₀: p = p₀ for s successes in n trials. The exact p-value sums the probabilities of all
 * outcomes no more likely than s; the normal approximation is shown with and without the continuity correction.
 */
export function ExactVsNormal() {
  const state = useFigureState({
    n: int(20, { min: 5, max: 100, step: 1, label: 'trials n' }),
    p0: slider(0.05, 0.95, 0.5, { step: 0.05, label: 'null proportion p₀' }),
    s: int(15, { min: 0, max: 100, step: 1, label: 'successes s' }),
  })
  const s = Math.min(state.s, state.n)

  const result = useMemo(() => {
    const ks = Array.from({ length: state.n + 1 }, (_, k) => k)
    const law = Binomial(state.n, state.p0)
    const pmf = ks.map((k) => law.prob(k))
    // A small tolerance keeps outcomes with equal probability (by symmetry) on the same side.
    const extreme = ks.filter((k) => pmf[k] <= pmf[s] * (1 + 1e-9))
    const rest = ks.filter((k) => pmf[k] > pmf[s] * (1 + 1e-9))
    const exact = Math.min(
      1,
      extreme.reduce((a, k) => a + pmf[k], 0),
    )
    const mean = state.n * state.p0
    const sd = Math.sqrt(state.n * state.p0 * (1 - state.p0))
    const gap = Math.abs(s - mean)
    const plain = 2 * (1 - normalCdf(gap / sd))
    const corrected = Math.min(1, 2 * (1 - normalCdf(Math.max(gap - 0.5, 0) / sd)))
    const xs = toFlat(linspace(-0.5, state.n + 0.5, 300))
    const series = [
      { name: 'less extreme than s', x: rest, y: rest.map((k) => pmf[k]), muted: true },
      { name: 'at least as extreme as s', x: extreme, y: extreme.map((k) => pmf[k]), slot: 0 },
      { name: 'normal approximation', x: xs, y: xs.map((x) => normalPdf((x - mean) / sd) / sd), slot: 1 },
    ] as const
    return { series, exact, plain, corrected }
  }, [state.n, s, state.p0])

  const xAxis = useAxis({ label: 'successes', hold: 'union' })
  const yAxis = useAxis({ label: 'probability', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Exact binomial p-value and its normal approximation"
      state={state}
      caption="Bars are the binomial distribution of the number of successes under H₀. The exact two-sided p-value adds up the coloured bars, the outcomes no more likely than the one observed. The normal curve approximates the bars; shifting the cut-off by half a bar, the continuity correction, matches the area of the bars much better. Drag the line labelled s to change the observed count."
      readouts={
        <>
          <Readout label="exact p" value={formatNumber(result.exact)} />
          <Readout label="normal p" value={formatNumber(result.plain)} />
          <Readout label="normal p, continuity-corrected" value={formatNumber(result.corrected)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={280}>
        <Bars {...result.series[0]} />
        <Bars {...result.series[1]} />
        <Curve {...result.series[2]} />
        <Handle kind="x" at={s} label="s" onDrag={(x) => state.set('s', Math.min(state.n, Math.round(x)))} />
      </Plot>
    </Figure>
  )
}
