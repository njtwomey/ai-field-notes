import { useMemo } from 'react'
import {
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { scores, type Scores } from '../_shared/partitions'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, uniform } from 'aifn/foundation/random'

type Mode = 'noise' | 'merge' | 'split'

const N = 90
const CENTRES = [
  [-2, 0],
  [2, 0],
  [0, 3],
]
const LEVELS = toFlat(linspace(0, 1, 21))
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
  const state = useFigureState({
    mode: choice<Mode>(
      [
        { value: 'noise', label: 'random labels' },
        { value: 'merge', label: 'merge two clusters' },
        { value: 'split', label: 'split a cluster' },
      ],
      'noise',
      { label: 'corruption' },
    ),
    level: float(0.4, { min: 0, max: 1, step: 0.05, label: 'level' }),
    seed: int(1, { min: 1, max: 20, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const data = useMemo(() => {
    const g = stream(state.seed)
    const truth = Array.from({ length: N }, (_, i) => i % 3)
    const points = truth.map((t) => [CENTRES[t][0] + 0.8 * normal(g), CENTRES[t][1] + 0.8 * normal(g)] as const)
    // A fixed random order and random labels, so the corruption grows smoothly with the level.
    const order = truth.map((_, i) => i)
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(uniform(g) * (i + 1))
      ;[order[i], order[j]] = [order[j], order[i]]
    }
    const randomLabel = truth.map(() => Math.floor(uniform(g) * 3))
    // Baseline: fully random labellings with three clusters, averaged.
    const baselines = Array.from({ length: 30 }, () =>
      scores(
        truth,
        truth.map(() => Math.floor(uniform(g) * 3)),
      ),
    )
    const baseline = Object.fromEntries(
      SHOWN.map((s) => [s.key, baselines.reduce((a, b) => a + b[s.key], 0) / baselines.length]),
    ) as Record<keyof Scores, number>
    return { truth, points, order, randomLabel, baseline }
  }, [state.seed])

  const corrupt = (f: number): number[] => {
    const pred = [...data.truth]
    if (state.mode === 'noise') {
      data.order.slice(0, Math.round(f * N)).forEach((i) => (pred[i] = data.randomLabel[i]))
    } else {
      const source = state.mode === 'merge' ? 1 : 0
      const members = data.order.filter((i) => data.truth[i] === source)
      // Merge: move part of cluster 2 into cluster 1. Split: move part of cluster 1 into a new cluster 4.
      members.slice(0, Math.round(f * members.length)).forEach((i) => (pred[i] = state.mode === 'merge' ? 0 : 3))
    }
    return pred
  }

  const curves = useMemo(() => LEVELS.map((f) => scores(data.truth, corrupt(f))), [data, state.mode]) // eslint-disable-line react-hooks/exhaustive-deps
  const pred = useMemo(() => corrupt(state.level), [data, state.mode, state.level]) // eslint-disable-line react-hooks/exhaustive-deps
  const current = useMemo(() => scores(data.truth, pred), [data, pred])

  const lines: SeriesSpec[] = SHOWN.map((s) => ({
    name: s.name,
    type: 'line',
    x: LEVELS,
    y: curves.map((c) => c[s.key]),
    slot: s.slot,
  }))
  const scatter = [
    {
      name: 'points',
      x: data.points.map((p) => p[0]),
      y: data.points.map((p) => p[1]),
      group: pred,
      groupNames: ['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4'],
    },
  ] as const

  const xAxis = useAxis({ label: 'x₁', range: [-5, 5] })
  const yAxis = useAxis({ label: 'x₂', range: [-3, 6] })
  const xAxis2 = useAxis({ label: 'corruption level', range: [0, 1] })
  const yAxis2 = useAxis({ label: 'score', range: [-0.1, 1.02] })
  return (
    <Figure
      title="Comparing clusterings as one degrades"
      state={state}
      caption="Left: the predicted clustering, coloured by its labels. Right: five scores against the true clustering as the corruption grows; drag the level. Fully random labels drive the adjusted Rand index and adjusted mutual information to about 0, as they are designed to; the Rand index stays near 0.56 and Fowlkes–Mallows near 0.33. V-measure is near 0 here, but its value for random labels grows with the number of clusters. Merging two clusters and splitting one hurt the scores differently: homogeneity falls under merges and completeness under splits. The bracketed values are averages over 30 fully random labellings."

      readouts={
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
        <Plot x={xAxis} y={yAxis} height={320}>
          <Points {...scatter[0]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          {seriesLayers(lines)}
          <Handle
            kind="x"
            at={state.level}
            label="corruption"
            onDrag={(x) => state.set('level', Math.min(1, Math.max(0, x)))}
          />
        </Plot>
      </div>
    </Figure>
  )
}
