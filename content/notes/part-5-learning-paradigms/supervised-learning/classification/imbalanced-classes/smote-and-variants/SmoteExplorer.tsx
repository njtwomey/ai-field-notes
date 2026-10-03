import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  useParam,
  type Segment,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'

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
  const g = rng(5)
  const majority: Pt[] = Array.from({ length: 200 }, () => [g.normal(), g.normal()])
  const minority: Pt[] = [
    ...Array.from({ length: 14 }, (): Pt => [2.1 + 0.5 * g.normal(), 1.8 + 0.5 * g.normal()]),
    ...Array.from({ length: 6 }, (): Pt => [-1.8 + 0.35 * g.normal(), 2.3 + 0.35 * g.normal()]),
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
  const k = useParam(5, { min: 1, max: 12, step: 1 })
  const rate = useParam(2, { min: 1, max: 5, step: 1 })
  const [method, setMethod] = useState<Method>('smote')
  const [seed, setSeed] = useState(1)

  const { majority, minority } = useMemo(() => makeData(), [])
  const all = useMemo(() => [...majority, ...minority], [majority, minority])

  // Majority count among each minority point's M nearest real neighbours.
  const majorityNeighbours = useMemo(
    () => minority.map((p, i) => nearest(p, all, M, majority.length + i).filter((j) => j < majority.length).length),
    [minority, all, majority.length],
  )

  const result = useMemo(() => {
    const g = rng(seed)
    const total = rate.value * minority.length
    // How many synthetic points each minority point generates.
    let counts: number[]
    if (method === 'smote') {
      counts = minority.map(() => rate.value)
    } else if (method === 'borderline') {
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
      const neighbours = nearest(p, minority, k.value, i)
      for (let c = 0; c < counts[i]; c++) {
        const q = minority[neighbours[Math.floor(g.uniform() * neighbours.length)]]
        const u = g.uniform()
        synthetic.push([p[0] + u * (q[0] - p[0]), p[1] + u * (q[1] - p[1])])
        segments.push({ from: p, to: q })
      }
    })
    // A synthetic point intrudes when its nearest real point is a majority point.
    const intruding = synthetic.filter((s) => nearest(s, all, 1)[0] < majority.length).length
    return { synthetic, segments, intruding }
  }, [seed, rate.value, method, k.value, minority, majorityNeighbours, all, majority.length])

  const series = useMemo<XYSeries[]>(
    () => [
      { name: 'majority', type: 'scatter', x: majority.map((p) => p[0]), y: majority.map((p) => p[1]), slot: 0 },
      { name: 'minority', type: 'scatter', x: minority.map((p) => p[0]), y: minority.map((p) => p[1]), slot: 1 },
      {
        name: 'synthetic',
        type: 'scatter',
        x: result.synthetic.map((p) => p[0]),
        y: result.synthetic.map((p) => p[1]),
        slot: 2,
      },
    ],
    [majority, minority, result.synthetic],
  )

  const noise = majorityNeighbours.filter((m) => m === M).length
  const danger = majorityNeighbours.filter((m) => m >= M / 2 && m < M).length

  return (
    <Interactive
      title="Synthetic minority oversampling"
      caption={`Majority points, a minority class in two clusters, and two mislabelled minority points inside the majority. Each synthetic point is drawn uniformly on the grey segment from a minority point to one of its k nearest minority neighbours. Raise k and SMOTE starts interpolating between the two clusters and from the noisy points, across majority territory. Borderline-SMOTE generates only from minority points with at least half their ${M} nearest neighbours in the majority (but not all, which it treats as noise); ADASYN generates more where the majority share is higher, so it amplifies the noisy points.`}
      controls={
        <>
          <ParamSlider label="neighbours k" param={k} />
          <ParamSlider label="synthetic points per minority point" param={rate} />
          <ParamChoice label="method" value={method} onChange={setMethod} options={METHODS} />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>Resample</ParamButton>
        </>
      }
      readout={
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
      <XYChart
        series={series}
        segments={result.segments}
        xLabel="x₁"
        yLabel="x₂"
        xRange={[-3.5, 4]}
        yRange={[-3, 4]}
        equalAspect
      />
    </Interactive>
  )
}
