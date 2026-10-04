import { useMemo } from 'react'
import {
  Bars,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream } from 'aifn/foundation/random'
import { normalCdf, normalQuantile } from 'aifn/numerics/special'

const STUDIES = 2000
const BINS = 20

/**
 * Many repeats of a two-sided z-test with effect size δ (in standard deviations) and n observations. The z statistic
 * is drawn from its exact sampling distribution, N(δ√n, 1).
 */
export function PValueHistogram() {
  const state = useFigureState({
    effect: float(0, { min: 0, max: 1, step: 0.01, label: 'true effect δ (standard deviations)' }),
    n: int(30, { min: 5, max: 200, step: 1, label: 'observations per study n' }),
    alphaParam: float(0.05, { min: 0.005, max: 0.2, step: 0.005, label: 'significance level α' }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })
  const alpha = state.alphaParam

  const result = useMemo(() => {
    const r = stream(state.seed)
    const shift = state.effect * Math.sqrt(state.n)
    const counts = new Array(BINS).fill(0)
    let below = 0
    for (let i = 0; i < STUDIES; i++) {
      const z = shift + normal(r)
      const p = 2 * (1 - normalCdf(Math.abs(z)))
      counts[Math.min(BINS - 1, Math.floor(p * BINS))]++
      if (p < alpha) below++
    }
    const crit = normalQuantile(1 - alpha / 2)
    const power = 1 - normalCdf(crit - shift) + normalCdf(-crit - shift)
    const width = 1 / BINS
    const series = [
      {
        name: 'p-values',
        x: counts.map((_, i) => (i + 0.5) * width),
        y: counts.map((c) => c / (STUDIES * width)),
        slot: 0,
      },
      { name: 'uniform density', x: [0, 1], y: [1, 1], slot: 2, dashed: true },
    ] as const
    return { series, share: below / STUDIES, power }
  }, [state.effect, state.n, alpha, state.seed])

  const xAxis = useAxis({ label: 'p-value', range: [0, 1] })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="2,000 studies, one p-value each"
      state={state}
      caption="With no effect (δ = 0) the p-values are uniform: about α of them fall below α by chance. Add an effect and they pile up near zero; the share below α is then the test's power. The dashed line is α: drag it to change the significance level."

      readouts={
        <>
          <Readout label="share with p < α" value={`${(100 * result.share).toFixed(1)}%`} />
          <Readout
            label={state.effect === 0 ? 'expected (α)' : 'theoretical power'}
            value={`${(100 * (state.effect === 0 ? alpha : result.power)).toFixed(1)}%`}
          />
          <Readout label="shift of z, δ√n" value={formatNumber(state.effect * Math.sqrt(state.n))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={280}>
        <Bars {...result.series[0]} />
        <Curve {...result.series[1]} />
        <Handle kind="x" at={alpha} label="α" onDrag={(v: number) => state.set('alphaParam', v)} />
      </Plot>
    </Figure>
  )
}
