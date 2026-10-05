import { useMemo } from 'react'
import {
  choice,
  Figure,
  int,
  Plot,
  Points,
  Readout,
  type Segment,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

type Pt = [number, number]
type Method = 'smote' | 'borderline' | 'adasyn'

const METHODS = [
  { value: 'smote', label: 'SMOTE' },
  { value: 'borderline', label: 'Borderline-SMOTE' },
  { value: 'adasyn', label: 'ADASYN' },
] as const

/** Neighbourhood size over all real points, used by Borderline-SMOTE (m) and ADASYN (K). */
const M = 7

const d2 = (a: Pt, b: Pt) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

/** Indices of the k nearest points of `pool` to `p`, excluding index `self`. */
function nearest(p: Pt, pool: Pt[], k: number, self = -1): number[] {
  return pool
    .map((q, i) => [d2(p, q), i] as const)
    .filter(([, i]) => i !== self)
    .sort((a, b) => a[0] - b[0])
    .slice(0, k)
    .map(([, i]) => i)
}

/**
 * A toy data set: a Gaussian majority, a minority split into two clusters, and two minority points that sit inside
 * the majority (label noise). Fixed seed, so the data never change; only the synthetic points do.
 */
function makeData() {
  const g = stream(5)
  const majority: Pt[] = Array.from({ length: 200 }, () => [normal(g), normal(g)])
  const minority: Pt[] = [
    ...Array.from({ length: 14 }, (): Pt => [2.1 + 0.5 * normal(g), 1.8 + 0.5 * normal(g)]),
    ...Array.from({ length: 6 }, (): Pt => [-1.8 + 0.35 * normal(g), 2.3 + 0.35 * normal(g)]),
    [0.2, -0.3],
    [-0.5, 0.4],
  ]
  return { majority, minority }
}

/**
 * SMOTE, Borderline-SMOTE and ADASYN on a 2D toy set. Each synthetic point lies on the segment from a minority point
 * to one of its k nearest minority neighbours; the variants differ in which minority points generate, and how many.
 */
export function SmoteExplorer() {
  const state = useFigureState({
    k: int(5, { min: 1, max: 12, step: 1, label: 'neighbours k' }),
    rate: int(2, { min: 1, max: 5, step: 1, label: 'synthetic points per minority point' }),
    method: choice<Method>(METHODS, 'smote', { label: 'method' }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })

  const { majority, minority } = useMemo(() => makeData(), [])
  const all = useMemo(() => [...majority, ...minority], [majority, minority])

  // Majority count among each minority point's M nearest real neighbours.
  const majorityNeighbours = useMemo(
    () => minority.map((p, i) => nearest(p, all, M, majority.length + i).filter((j) => j < majority.length).length),
    [minority, all, majority.length],
  )

  const result = useMemo(() => {
    const g = stream(state.seed)
    const total = state.rate * minority.length
    // How many synthetic points each minority point generates.
    let counts: number[]
    if (state.method === 'smote') {
      counts = minority.map(() => state.rate)
    } else if (state.method === 'borderline') {
      // Danger: at least half, but not all, of the M neighbours are majority. Only danger points generate.
      const danger = majorityNeighbours.map((m) => m >= M / 2 && m < M)
      const nDanger = danger.filter(Boolean).length
      counts = danger.map((isDanger) => (isDanger && nDanger ? Math.round(total / nDanger) : 0))
    } else {
      const r = majorityNeighbours.map((m) => m / M)
      const sum = r.reduce((a, b) => a + b, 0)
      counts = r.map((ri) => (sum ? Math.round((ri / sum) * total) : 0))
    }
    const synthetic: Pt[] = []
    const segments: Segment[] = []
    minority.forEach((p, i) => {
      if (!counts[i]) return
      const neighbours = nearest(p, minority, state.k, i)
      for (let c = 0; c < counts[i]; c++) {
        const q = minority[neighbours[Math.floor(uniform(g) * neighbours.length)]]
        const u = uniform(g)
        synthetic.push([p[0] + u * (q[0] - p[0]), p[1] + u * (q[1] - p[1])])
        segments.push({ from: p, to: q })
      }
    })
    // A synthetic point intrudes when its nearest real point is a majority point.
    const intruding = synthetic.filter((s) => nearest(s, all, 1)[0] < majority.length).length
    return { synthetic, segments, intruding }
  }, [state.seed, state.rate, state.method, state.k, minority, majorityNeighbours, all, majority.length])

  const series = useMemo(
    () =>
      [
        { name: 'majority', x: majority.map((p) => p[0]), y: majority.map((p) => p[1]), slot: 0 },
        { name: 'minority', x: minority.map((p) => p[0]), y: minority.map((p) => p[1]), slot: 1 },
        {
          name: 'synthetic',
          x: result.synthetic.map((p) => p[0]),
          y: result.synthetic.map((p) => p[1]),
          slot: 2,
        },
      ] as const,
    [majority, minority, result.synthetic],
  )

  const noise = majorityNeighbours.filter((m) => m === M).length
  const danger = majorityNeighbours.filter((m) => m >= M / 2 && m < M).length

  const xAxis = useAxis({ label: 'x₁', range: [-3.5, 4] })
  const yAxis = useAxis({ label: 'x₂', range: [-3, 4], equal: xAxis })
  return (
    <Figure
      title="Synthetic minority oversampling"
      state={state}
      caption={`Majority points, a minority class in two clusters, and two mislabelled minority points inside the majority. Each synthetic point is drawn uniformly on the grey segment from a minority point to one of its k nearest minority neighbours. Raise k and SMOTE starts interpolating between the two clusters and from the noisy points, across majority territory. Borderline-SMOTE generates only from minority points with at least half their ${M} nearest neighbours in the majority (but not all, which it treats as noise); ADASYN generates more where the majority share is higher, so it amplifies the noisy points.`}

      readouts={
        <>
          <Readout label="synthetic points" value={result.synthetic.length} />
          <Readout label="nearest real point is majority" value={result.intruding} />
          <Readout
            label="minority points: noise, danger, safe"
            value={`${noise}, ${danger}, ${minority.length - noise - danger}`}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Points {...series[0]} />
        <Points {...series[1]} />
        <Points {...series[2]} />
        <Segments segments={result.segments} />
      </Plot>
    </Figure>
  )
}
