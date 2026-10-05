import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'
import { normalQuantile, studentTCdf, studentTQuantile } from 'aifn-compute/numerics/special'

type Method = 't' | 'z'

const SHOWN = 50
const MU = 0
const SIGMA = 1

/**
 * Repeated samples of size n from N(0, 1), each giving an interval x̄ ± q·s/√n. With q from Student's t the long-run
 * coverage equals the nominal level; with the normal quantile and the estimated s it falls short for small n.
 */
export function Coverage() {
  const state = useFigureState({
    n: int(5, { min: 2, max: 50, step: 1, label: 'sample size n' }),
    level: slider(0.5, 0.99, 0.95, { step: 0.01, label: 'confidence level', format: (v) => `${Math.round(100 * v)}%` }),
    method: choice<Method>(
      [
        { value: 't', label: 't, n − 1 df' },
        { value: 'z', label: 'normal' },
      ],
      't',
      { label: 'quantile' },
    ),
    seed: int(1, { ge: 0, label: 'seed' }),
  })

  const result = useMemo(() => {
    const r = stream(state.seed)
    const q =
      state.method === 't'
        ? studentTQuantile(1 - (1 - state.level) / 2, state.n - 1)
        : normalQuantile(1 - (1 - state.level) / 2)
    const hit: { x: number[]; y: number[] } = { x: [], y: [] }
    const miss: { x: number[]; y: number[] } = { x: [], y: [] }
    let covered = 0
    for (let i = 1; i <= SHOWN; i++) {
      const xs = Array.from({ length: state.n }, () => MU + SIGMA * normal(r))
      const mean = xs.reduce((a, b) => a + b, 0) / state.n
      const s = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / (state.n - 1))
      const half = (q * s) / Math.sqrt(state.n)
      const ok = Math.abs(mean - MU) <= half
      if (ok) covered++
      // NaN breaks the line, so one series draws many separate intervals.
      const target = ok ? hit : miss
      target.x.push(i, i, NaN)
      target.y.push(mean - half, mean + half, NaN)
    }
    // Exact long-run coverage: P(|T| ≤ q) with T ~ t(n − 1), whichever q is used.
    const longRun = 2 * studentTCdf(q, state.n - 1) - 1
    const series: SeriesSpec[] = [
      { name: 'covers μ', type: 'line', ...hit, slot: 0 },
      { name: 'misses μ', type: 'line', ...miss, slot: 1 },
      { name: 'true mean μ', type: 'line', x: [0, SHOWN + 1], y: [MU, MU], slot: 2, dashed: true },
    ]
    return { series, covered, longRun, q }
  }, [state.n, state.level, state.method, state.seed])

  const xAxis = useAxis({ label: 'sample', range: [0, SHOWN + 1] })
  const yAxis = useAxis({ label: 'interval for μ', hold: 'union' })
  return (
    <Figure
      title="Fifty intervals from fifty samples"
      state={state}
      caption="Each vertical line is one interval computed from a fresh sample of size n. The procedure covers the true mean in a fixed fraction of samples; any single interval either covers it or does not. With the normal quantile and an estimated standard deviation, the intervals are too narrow when n is small."

      readouts={
        <>
          <Readout label="covered here" value={`${result.covered} of ${SHOWN}`} />
          <Readout label="long-run coverage" value={`${formatNumber(100 * result.longRun)}%`} />
          <Readout label="quantile q" value={formatNumber(result.q)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        {seriesLayers(result.series)}
      </Plot>
    </Figure>
  )
}
