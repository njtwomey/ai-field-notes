import { useMemo, useState } from 'react'
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
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { scores, type Scores } from '../_shared/partitions'

type Mode = 'noise' | 'merge' | 'split'

const N = 90
const CENTRES = [
  [-2, 0],
  [2, 0],
  [0, 3],
]
const LEVELS = linspace(0, 1, 21)
const SHOWN: { key: keyof Scores; name: string; slot: number }[] = [
  { key: 'rand', name: 'Rand index', slot: 0 },
  { key: 'ari', name: 'adjusted Rand', slot: 1 },
  { key: 'ami', name: 'adjusted mutual information', slot: 2 },
  { key: 'vMeasure', name: 'V-measure (= NMI)', slot: 3 },
  { key: 'fmi', name: 'Fowlkes–Mallows', slot: 4 },
]

/**
 * A true clustering of three blobs, and a predicted clustering corrupted step by step: labels randomised, one cluster
 * absorbed by another, or one cluster split in two. Chance-corrected scores fall to about 0 for random labels; the Rand
 * index, V-measure and Fowlkes–Mallows do not.
 */
export function ClusteringCorruption() {
  const [mode, setMode] = useState<Mode>('noise')
  const level = useParam(0.4, { min: 0, max: 1, step: 0.05 })
  const seed = useParam(1, { min: 1, max: 20, step: 1 })

  const data = useMemo(() => {
    const g = rng(seed.value)
    const truth = Array.from({ length: N }, (_, i) => i % 3)
    const points = truth.map((t) => [CENTRES[t][0] + 0.8 * g.normal(), CENTRES[t][1] + 0.8 * g.normal()] as const)
    // A fixed random order and random labels, so the corruption grows smoothly with the level.
    const order = truth.map((_, i) => i)
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(g.uniform() * (i + 1))
      ;[order[i], order[j]] = [order[j], order[i]]
    }
    const randomLabel = truth.map(() => Math.floor(g.uniform() * 3))
    // Baseline: fully random labellings with three clusters, averaged.
    const baselines = Array.from({ length: 30 }, () =>
      scores(
        truth,
        truth.map(() => Math.floor(g.uniform() * 3)),
      ),
    )
    const baseline = Object.fromEntries(
      SHOWN.map((s) => [s.key, baselines.reduce((a, b) => a + b[s.key], 0) / baselines.length]),
    ) as Record<keyof Scores, number>
    return { truth, points, order, randomLabel, baseline }
  }, [seed.value])

  const corrupt = (f: number): number[] => {
    const pred = [...data.truth]
    if (mode === 'noise') {
      data.order.slice(0, Math.round(f * N)).forEach((i) => (pred[i] = data.randomLabel[i]))
    } else {
      const source = mode === 'merge' ? 1 : 0
      const members = data.order.filter((i) => data.truth[i] === source)
      // Merge: move part of cluster 2 into cluster 1. Split: move part of cluster 1 into a new cluster 4.
      members.slice(0, Math.round(f * members.length)).forEach((i) => (pred[i] = mode === 'merge' ? 0 : 3))
    }
    return pred
  }

  const curves = useMemo(() => LEVELS.map((f) => scores(data.truth, corrupt(f))), [data, mode]) // eslint-disable-line react-hooks/exhaustive-deps
  const pred = useMemo(() => corrupt(level.value), [data, mode, level.value]) // eslint-disable-line react-hooks/exhaustive-deps
  const current = useMemo(() => scores(data.truth, pred), [data, pred])

  const lines: XYSeries[] = SHOWN.map((s) => ({
    name: s.name,
    type: 'line',
    x: LEVELS,
    y: curves.map((c) => c[s.key]),
    slot: s.slot,
  }))
  const scatter: XYSeries[] = [
    {
      name: 'points',
      type: 'scatter',
      x: data.points.map((p) => p[0]),
      y: data.points.map((p) => p[1]),
      group: pred,
      groupNames: ['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4'],
    },
  ]
  const handles: Handle[] = [
    { kind: 'x', at: level.value, label: 'corruption', onDrag: (x) => level.set(Math.min(1, Math.max(0, x))) },
  ]

  return (
    <Interactive
      title="Comparing clusterings as one degrades"
      caption="Left: the predicted clustering, coloured by its labels. Right: five scores against the true clustering as the corruption grows; drag the level. Fully random labels drive the adjusted Rand index and adjusted mutual information to about 0, as they are designed to; the Rand index stays near 0.56 and Fowlkes–Mallows near 0.33. V-measure is near 0 here, but its value for random labels grows with the number of clusters. Merging two clusters and splitting one hurt the scores differently: homogeneity falls under merges and completeness under splits. The bracketed values are averages over 30 fully random labellings."
      controls={
        <>
          <ParamChoice
            label="corruption"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'noise', label: 'random labels' },
              { value: 'merge', label: 'merge two clusters' },
              { value: 'split', label: 'split a cluster' },
            ]}
          />
          <ParamSlider label="level" param={level} />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          {SHOWN.map((s) => (
            <Readout
              key={s.key}
              label={`${s.name} (random)`}
              value={`${formatNumber(current[s.key])} (${formatNumber(data.baseline[s.key])})`}
            />
          ))}
          <Readout label="homogeneity" value={formatNumber(current.homogeneity)} />
          <Readout label="completeness" value={formatNumber(current.completeness)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart series={scatter} xLabel="x₁" yLabel="x₂" xRange={[-5, 5]} yRange={[-3, 6]} height={320} />
        <XYChart
          series={lines}
          handles={handles}
          xLabel="corruption level"
          yLabel="score"
          xRange={[0, 1]}
          yRange={[-0.1, 1.02]}
          height={320}
        />
      </div>
    </Interactive>
  )
}
