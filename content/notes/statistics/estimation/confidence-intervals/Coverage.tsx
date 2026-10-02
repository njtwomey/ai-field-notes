import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'
import { normalQuantile, studentTCdf } from '@/lib/math/special'
import { studentTQuantile } from '@/lib/math/tests'

type Method = 't' | 'z'

const SHOWN = 50
const MU = 0
const SIGMA = 1

/**
 * Repeated samples of size n from N(0, 1), each giving an interval x̄ ± q·s/√n. With q from Student's t the long-run
 * coverage equals the nominal level; with the normal quantile and the estimated s it falls short for small n.
 */
export function Coverage() {
  const [n, setN] = useState(5)
  const [level, setLevel] = useState(0.95)
  const [method, setMethod] = useState<Method>('t')
  const [seed, setSeed] = useState(1)

  const result = useMemo(() => {
    const r = rng(seed)
    const q = method === 't' ? studentTQuantile(1 - (1 - level) / 2, n - 1) : normalQuantile(1 - (1 - level) / 2)
    const hit: { x: number[]; y: number[] } = { x: [], y: [] }
    const miss: { x: number[]; y: number[] } = { x: [], y: [] }
    let covered = 0
    for (let i = 1; i <= SHOWN; i++) {
      const xs = Array.from({ length: n }, () => MU + SIGMA * r.normal())
      const mean = xs.reduce((a, b) => a + b, 0) / n
      const s = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1))
      const half = (q * s) / Math.sqrt(n)
      const ok = Math.abs(mean - MU) <= half
      if (ok) covered++
      // NaN breaks the line, so one series draws many separate intervals.
      const target = ok ? hit : miss
      target.x.push(i, i, NaN)
      target.y.push(mean - half, mean + half, NaN)
    }
    // Exact long-run coverage: P(|T| ≤ q) with T ~ t(n − 1), whichever q is used.
    const longRun = 2 * studentTCdf(q, n - 1) - 1
    const series: XYSeries[] = [
      { name: 'covers μ', type: 'line', ...hit, slot: 0 },
      { name: 'misses μ', type: 'line', ...miss, slot: 1 },
      { name: 'true mean μ', type: 'line', x: [0, SHOWN + 1], y: [MU, MU], slot: 2, dashed: true },
    ]
    return { series, covered, longRun, q }
  }, [n, level, method, seed])

  return (
    <Interactive
      title="Fifty intervals from fifty samples"
      caption="Each vertical line is one interval computed from a fresh sample of size n. The procedure covers the true mean in a fixed fraction of samples; any single interval either covers it or does not. With the normal quantile and an estimated standard deviation, the intervals are too narrow when n is small."
      controls={
        <>
          <ParamSlider label="sample size n" value={n} onChange={setN} min={2} max={50} step={1} />
          <ParamSlider
            label="confidence level"
            value={level}
            onChange={setLevel}
            min={0.5}
            max={0.99}
            step={0.01}
            format={(v) => `${Math.round(100 * v)}%`}
          />
          <ParamChoice
            label="quantile"
            value={method}
            onChange={setMethod}
            options={[
              { value: 't', label: 't, n − 1 df' },
              { value: 'z', label: 'normal' },
            ]}
          />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New samples</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="covered here" value={`${result.covered} of ${SHOWN}`} />
          <Readout label="long-run coverage" value={`${formatNumber(100 * result.longRun)}%`} />
          <Readout label="quantile q" value={formatNumber(result.q)} />
        </>
      }
    >
      <XYChart height={300} series={result.series} xRange={[0, SHOWN + 1]} xLabel="sample" yLabel="interval for μ" />
    </Interactive>
  )
}
