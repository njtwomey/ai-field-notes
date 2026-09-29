import { useMemo, useState } from 'react'
import { MathText } from '@/components/content/MathText'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace, sigmoid } from '@/lib/math'
import { useClassColors } from '../_shared/classColor'
import { exceedanceProbs, fitShared1d, logistic1d, nonParallelSample } from '../_shared/ordinal'

type Heads = 'independent' | 'shared'
const HEADS = [
  { value: 'independent' as const, label: 'independent (Frank & Hall)' },
  { value: 'shared' as const, label: 'shared slope (CORAL)' },
]
const X = linspace(-3, 3, 241)
const K = 4

/**
 * Three binary classifiers for y > 1, y > 2 and y > 3 on one feature. Fitted independently, their curves can cross, and
 * the differenced class probabilities go negative there. With one shared slope and separate intercepts (CORAL), the
 * curves are ordered everywhere.
 */
export function RankConsistency() {
  const [heads, setHeads] = useState<Heads>('independent')
  const [n, setN] = useState(60)
  const [delta, setDelta] = useState(0.8)
  const at = useParam(2, { min: -3, max: 3, step: 0.02 })
  const colors = useClassColors(K)

  const fits = useMemo(() => {
    const { x, y } = nonParallelSample(n, delta, 23)
    const separate = [0, 1, 2].map((j) =>
      logistic1d(
        x,
        y.map((v) => (v > j ? 1 : 0)),
      ),
    )
    const shared = fitShared1d(x, y, K, 'all-threshold')
    const exceed = {
      independent: (v: number) => separate.map((f) => sigmoid(f.a + f.b * v)),
      shared: (v: number) => shared.theta.map((t) => sigmoid(shared.w * v - t)),
    }
    const crossed = X.filter((v) => exceedanceProbs(exceed.independent(v)).some((p) => p < 0)).length / X.length
    return { exceed, crossed }
  }, [n, delta])

  const series = useMemo<XYSeries[]>(() => {
    const out: XYSeries[] = []
    for (const kind of ['independent', 'shared'] as const) {
      const rows = X.map(fits.exceed[kind])
      ;[0, 1, 2].forEach((j) =>
        out.push({
          name: `P(y > ${j + 1}), ${kind}`,
          type: 'line',
          x: X,
          y: rows.map((r) => r[j]),
          slot: j,
          dashed: kind !== heads,
          muted: kind !== heads,
        }),
      )
    }
    return out
  }, [fits, heads])

  const probs = exceedanceProbs(fits.exceed[heads](at.value))
  const bars: XYSeries[] = [{ name: 'P(y = k | x)', type: 'bar', x: [1, 2, 3, 4], y: probs, pointColors: colors }]
  const handles: Handle[] = [{ kind: 'x', at: at.value, label: 'x', onDrag: at.set }]
  const lowest = Math.min(0, ...probs)

  return (
    <Interactive
      title="Rank consistency of binary decompositions"
      caption={
        <MathText text="Each curve is one binary classifier's estimate of $P(y > k \mid x)$, and class probabilities are the differences $P(y > k - 1) - P(y > k)$. The data's true splits have slopes $1.2\,(1 + \delta(k - 2))$. Independent fits to a small sample can cross, and where they cross a class probability is negative. A shared slope with ordered intercepts keeps the curves nested. Drag the cursor to read the class probabilities of the selected heads." />
      }
      controls={
        <>
          <ParamChoice label="binary heads" value={heads} onChange={setHeads} options={HEADS} />
          <ParamSlider label="sample size n" value={n} onChange={setN} min={30} max={600} step={10} />
          <ParamSlider label="slope spread δ" value={delta} onChange={setDelta} min={0} max={1.2} step={0.05} />
          <ParamSlider label="query x" param={at} />
        </>
      }
      readout={
        <>
          {probs.map((p, k) => (
            <Readout key={k} label={`P(y = ${k + 1})`} value={formatNumber(p)} />
          ))}
          <Readout label="share of x with a negative class (independent)" value={formatNumber(fits.crossed)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[3fr_2fr]">
        <XYChart
          series={series}
          handles={handles}
          xRange={[-3, 3]}
          yRange={[0, 1]}
          xLabel="feature x"
          yLabel="P(y > k | x)"
          height={300}
          ariaLabel="Binary classifier outputs against the feature"
        />
        <XYChart
          series={bars}
          xRange={[0.5, 4.5]}
          integerX
          yRange={[lowest < 0 ? Math.floor(lowest * 10) / 10 : 0, 1]}
          xLabel="class k"
          yLabel="probability"
          height={300}
          ariaLabel="Class probabilities at the cursor"
        />
      </div>
    </Interactive>
  )
}
