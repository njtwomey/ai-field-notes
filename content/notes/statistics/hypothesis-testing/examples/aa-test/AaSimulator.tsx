import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'
import { normalPdf } from '@/lib/math/special'
import { binomialLower } from '@/lib/math/tests'
import { sampleBinomial, twoProportionZ } from '../_shared/ab'

const P_BINS = 20
const D_BINS = 30

/** The central 95% range of Binomial(N, α): how many A/A tests should come out significant. */
function expectedRange(experiments: number, alpha: number): [number, number] {
  let lo = 0
  while (binomialLower(lo, experiments, alpha) < 0.025) lo++
  let hi = lo
  while (binomialLower(hi, experiments, alpha) < 0.975) hi++
  return [lo, hi]
}

/**
 * Many A/A experiments: both arms convert at the same rate, and each is analysed with the two-proportion z-test. The
 * p-values should be uniform, about α of them significant, and the observed differences spread as the formula says.
 */
export function AaSimulator() {
  const [n, setN] = useState(10000)
  const [rate, setRate] = useState(0.05)
  const [experiments, setExperiments] = useState(1000)
  const alpha = useParam(0.05, { min: 0.01, max: 0.2, step: 0.01 })
  const [seed, setSeed] = useState(1)

  const r = useMemo(() => {
    const { uniform } = rng(seed)
    const pValues: number[] = []
    const diffs: number[] = []
    let significant = 0
    let covered = 0
    for (let e = 0; e < experiments; e++) {
      const t = twoProportionZ(sampleBinomial(n, rate, uniform), n, sampleBinomial(n, rate, uniform), n, alpha.value)
      pValues.push(t.p)
      diffs.push(t.diff)
      if (t.p < alpha.value) significant++
      if (t.ci[0] <= 0 && t.ci[1] >= 0) covered++
    }
    const sdTheory = Math.sqrt((2 * rate * (1 - rate)) / n)
    const meanDiff = diffs.reduce((s, d) => s + d, 0) / experiments
    const sdObserved = Math.sqrt(diffs.reduce((s, d) => s + (d - meanDiff) ** 2, 0) / (experiments - 1))

    const pCounts = new Array(P_BINS).fill(0)
    for (const p of pValues) pCounts[Math.min(P_BINS - 1, Math.floor(p * P_BINS))]++
    const pSeries: XYSeries[] = [
      {
        name: 'A/A p-values',
        type: 'bar',
        x: pCounts.map((_, i) => (i + 0.5) / P_BINS),
        y: pCounts.map((c) => (c * P_BINS) / experiments),
        slot: 0,
      },
      { name: 'uniform', type: 'line', x: [0, 1], y: [1, 1], dashed: true, slot: 2 },
    ]

    // Differences in percentage points, against the normal curve the formula predicts.
    const span = 4 * sdTheory
    const width = (2 * span) / D_BINS
    const dCounts = new Array(D_BINS).fill(0)
    for (const d of diffs) {
      const i = Math.floor((d + span) / width)
      if (i >= 0 && i < D_BINS) dCounts[i]++
    }
    const grid = Array.from({ length: 81 }, (_, i) => -span + (2 * span * i) / 80)
    const dSeries: XYSeries[] = [
      {
        name: 'observed B − A',
        type: 'bar',
        x: dCounts.map((_, i) => 100 * (-span + (i + 0.5) * width)),
        y: dCounts.map((c) => c / (experiments * width * 100)),
        slot: 0,
      },
      {
        name: 'N(0, 2p(1 − p)/n)',
        type: 'line',
        x: grid.map((d) => 100 * d),
        y: grid.map((d) => normalPdf(d / sdTheory) / (sdTheory * 100)),
        slot: 1,
      },
    ]
    return { significant, covered, sdTheory, sdObserved, pSeries, dSeries }
  }, [n, rate, experiments, alpha.value, seed])

  const [lo, hi] = useMemo(() => expectedRange(experiments, alpha.value), [experiments, alpha.value])
  const handles: Handle[] = [{ kind: 'x', at: alpha.value, label: 'α', onDrag: alpha.set }]

  return (
    <Interactive
      title="A thousand experiments with nothing to find"
      caption="Each experiment splits n users per arm, both converting at the same rate p, and runs a two-sided two-proportion z-test. Left: the p-values are close to uniform, so the share below α is about α; drag the line labelled α. Right: the observed differences B − A, in percentage points, against the normal curve with standard deviation √(2p(1 − p)/n). Differences of this size are what chance alone produces."
      controls={
        <>
          <ParamSlider label="users per arm n" value={n} onChange={setN} min={500} max={50000} step={500} />
          <ParamSlider label="conversion rate p" value={rate} onChange={setRate} min={0.01} max={0.5} step={0.01} />
          <ParamSlider
            label="experiments"
            value={experiments}
            onChange={setExperiments}
            min={100}
            max={2000}
            step={100}
          />
          <ParamSlider label="significance level α" param={alpha} />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>Rerun</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="significant" value={`${r.significant} of ${experiments}`} />
          <Readout label="expected (95% range)" value={`${Math.round(alpha.value * experiments)} (${lo}–${hi})`} />
          <Readout label="intervals covering 0" value={`${((100 * r.covered) / experiments).toFixed(1)}%`} />
          <Readout label="sd of B − A, observed" value={`${formatNumber(100 * r.sdObserved)} pp`} />
          <Readout label="sd of B − A, formula" value={`${formatNumber(100 * r.sdTheory)} pp`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          height={280}
          series={r.pSeries}
          xRange={[0, 1]}
          yRange={[0, undefined]}
          xLabel="p-value"
          yLabel="density"
          handles={handles}
        />
        <XYChart
          height={280}
          series={r.dSeries}
          yRange={[0, undefined]}
          xLabel="observed B − A (percentage points)"
          yLabel="density"
        />
      </div>
    </Interactive>
  )
}
