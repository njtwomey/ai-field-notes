import { useMemo } from 'react'
import {
  Figure,
  float,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'
import { normalCdf } from 'aifn-compute/numerics/special'

type Test = { p: number; effect: boolean }

/**
 * m one-sided z-tests: a share of them have a real effect (z ~ N(shift, 1)), the rest are true nulls (z ~ N(0, 1)).
 * Sorted p-values are compared with the Benjamini–Hochberg line q·i/m and the Bonferroni threshold q/m.
 */
export function StepUp() {
  const state = useFigureState({
    q: float(0.1, { min: 0.005, max: 0.5, step: 0.005, label: 'target level q' }),
    m: int(200, { min: 20, max: 1000, step: 10, label: 'tests m' }),
    share: float(0.2, { min: 0, max: 1, step: 0.05, label: 'share with a real effect' }),
    shift: float(3, { min: 0, max: 5, step: 0.1, label: 'effect size (z shift)' }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })

  const tests = useMemo((): Test[] => {
    const r = stream(state.seed)
    const effects = Math.round(state.share * state.m)
    return Array.from({ length: state.m }, (_, i) => {
      const effect = i < effects
      const z = (effect ? state.shift : 0) + normal(r)
      return { p: Math.max(1 - normalCdf(z), 1e-12), effect }
    }).sort((a, b) => a.p - b.p)
  }, [state.m, state.share, state.shift, state.seed])

  const result = useMemo(() => {
    // Benjamini–Hochberg: reject the k smallest, where k is the largest rank with p₍ₖ₎ ≤ q·k/m.
    let k = 0
    tests.forEach((t, i) => {
      if (t.p <= (state.q * (i + 1)) / state.m) k = i + 1
    })
    const bhFalse = tests.slice(0, k).filter((t) => !t.effect).length
    const bonf = tests.filter((t) => t.p <= state.q / state.m)
    const ranks = tests.map((_, i) => i + 1)
    const pick = (effect: boolean) => ({
      x: ranks.filter((_, i) => tests[i].effect === effect),
      y: tests.filter((t) => t.effect === effect).map((t) => t.p),
    })
    const series: SeriesSpec[] = [
      { name: 'true null', type: 'scatter', ...pick(false), slot: 0 },
      { name: 'real effect', type: 'scatter', ...pick(true), slot: 1 },
      { name: 'BH line q·i/m', type: 'line', x: ranks, y: ranks.map((i) => (state.q * i) / state.m), slot: 2 },
      {
        name: 'Bonferroni q/m',
        type: 'line',
        x: [1, state.m],
        y: [state.q / state.m, state.q / state.m],
        slot: 3,
        dashed: true,
      },
    ]
    return {
      series,
      k,
      bhFalse,
      bonfCount: bonf.length,
      bonfFalse: bonf.filter((t) => !t.effect).length,
    }
  }, [tests, state.q, state.m])

  const fdp = result.k ? result.bhFalse / result.k : 0

  const xAxis = useAxis({ label: 'rank i', hold: 'union' })
  const yAxis = useAxis({ label: 'p-value', range: [undefined, 1], hold: 'union', log: true })
  return (
    <Figure
      title="Benjamini–Hochberg on sorted p-values"
      state={state}
      caption="The m p-values are sorted and plotted against their rank on a log scale. Benjamini–Hochberg rejects every test up to the last one below the line q·i/m. Bonferroni rejects only those below q/m. Drag the end of the BH line, or use the q slider, to change the target false discovery rate."

      readouts={
        <>
          <Readout label="BH rejections" value={`${result.k} (${result.bhFalse} false)`} />
          <Readout label="false discovery proportion" value={`${(100 * fdp).toFixed(1)}%`} />
          <Readout label="Bonferroni rejections" value={`${result.bonfCount} (${result.bonfFalse} false)`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        {seriesLayers(result.series)}
        <Handle kind="point" at={[state.m, state.q]} label="q" onDrag={([, y]) => state.set('q', y)} />
      </Plot>
    </Figure>
  )
}
