import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  MathText,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

type Point = { x: number; y: number; age: number }

/** A user's action pins in a 2-D stand-in for embedding space: three interests of different recency, plus strays. */
function actions(): Point[] {
  const r = stream(20)
  const blob = (cx: number, cy: number, sd: number, n: number, age0: number, age1: number) =>
    Array.from({ length: n }, () => ({
      x: cx + sd * normal(r),
      y: cy + sd * normal(r),
      age: Math.round(age0 + (age1 - age0) * uniform(r)),
    }))
  return [
    ...blob(-2.2, 1.2, 0.35, 9, 0, 15),
    ...blob(2.0, 1.8, 0.4, 8, 30, 85),
    ...blob(0.3, -1.8, 0.35, 6, 5, 60),
    { x: -0.5, y: 0.3, age: 3 },
    { x: 3.0, y: -1.2, age: 40 },
    { x: -2.8, y: -1.6, age: 70 },
  ]
}

const POINTS = actions()
const M = POINTS.length
const MAX_CLUSTERS = 8

type Merge = { a: number[]; b: number[]; d: number }

/** Ward clustering with the Lance–Williams update on squared Euclidean distances. Returns the merge history. */
function wardMerges(points: Point[]): Merge[] {
  const members = new Map<number, number[]>(points.map((_, i) => [i, [i]]))
  const dist = new Map<string, number>()
  const key = (i: number, j: number) => (i < j ? `${i},${j}` : `${j},${i}`)
  for (let i = 0; i < points.length; i++)
    for (let j = i + 1; j < points.length; j++)
      dist.set(key(i, j), (points[i].x - points[j].x) ** 2 + (points[i].y - points[j].y) ** 2)
  const merges: Merge[] = []
  let next = points.length
  while (members.size > 1) {
    const active = [...members.keys()]
    let best: [number, number, number] = [-1, -1, Infinity]
    for (let s = 0; s < active.length; s++)
      for (let t = s + 1; t < active.length; t++) {
        const d = dist.get(key(active[s], active[t]))!
        if (d < best[2]) best = [active[s], active[t], d]
      }
    const [i, j, dij] = best
    const ni = members.get(i)!.length
    const nj = members.get(j)!.length
    for (const k of active) {
      if (k === i || k === j) continue
      const nk = members.get(k)!.length
      const dk = ((ni + nk) * dist.get(key(i, k))! + (nj + nk) * dist.get(key(j, k))! - nk * dij) / (ni + nj + nk)
      dist.set(key(next, k), dk)
    }
    merges.push({ a: members.get(i)!, b: members.get(j)!, d: dij })
    members.set(next, [...members.get(i)!, ...members.get(j)!])
    members.delete(i)
    members.delete(j)
    next++
  }
  return merges
}

const MERGES = wardMerges(POINTS)

/** PinnerSage's cluster extraction: the largest merged clusters whose merge distance is at most alpha. */
function clustersAt(alpha: number): number[][] {
  const chosen: number[][] = []
  const used = new Set<number>()
  for (const m of [...MERGES].sort((p, q) => q.d - p.d)) {
    if (m.d > alpha) continue
    const union = [...m.a, ...m.b]
    if (union.some((i) => used.has(i))) continue
    union.forEach((i) => used.add(i))
    chosen.push(union.sort((p, q) => p - q))
  }
  return chosen.sort((p, q) => p[0] - q[0])
}

const HEIGHTS = MERGES.map((m) => m.d)
// The smallest threshold index that yields at most MAX_CLUSTERS clusters, so every cluster gets its own colour.
const FIRST = HEIGHTS.findIndex((h) => clustersAt(h).length <= MAX_CLUSTERS)
// Start at the first threshold with three clusters covering all but the three strays, one cluster per interest.
const START = HEIGHTS.findIndex((h) => clustersAt(h).length === 3 && clustersAt(h).flat().length >= M - 3)

const sq = (i: number, j: number) => (POINTS[i].x - POINTS[j].x) ** 2 + (POINTS[i].y - POINTS[j].y) ** 2

/** The member minimising the sum of squared distances to the other members. */
const medoidOf = (c: number[]) =>
  c.reduce((best, m) => (c.reduce((s, j) => s + sq(m, j), 0) < c.reduce((s, j) => s + sq(best, j), 0) ? m : best))

type Lambda = '0' | '0.01' | '0.1'

/** Ward clustering of one user's action pins at a merge threshold, with medoids and time-decayed importance. */
export function WardMedoids() {
  const state = useFigureState({
    step: slider(FIRST, HEIGHTS.length - 1, START, {
      step: 1,
      label: 'merge threshold α (index into merge distances)',
      format: (v) => formatNumber(HEIGHTS[v]),
    }),
    lambda: choice<Lambda>(
      [
        { value: '0', label: '0' },
        { value: '0.01', label: '0.01' },
        { value: '0.1', label: '0.1' },
      ],
      '0.01',
      { label: 'decay λ per day' },
    ),
  })
  const alpha = HEIGHTS[state.step]
  const clusters = useMemo(() => clustersAt(alpha), [alpha])

  const { series, ranked, stray } = useMemo(() => {
    const group = new Array<number>(M).fill(-1)
    clusters.forEach((c, g) => c.forEach((i) => (group[i] = g)))
    const inAny = POINTS.map((_, i) => i).filter((i) => group[i] >= 0)
    const out = POINTS.map((_, i) => i).filter((i) => group[i] < 0)
    const medoids = clusters.map(medoidOf)
    const l = Number(state.lambda)
    const importance = clusters.map((c) => c.reduce((s, i) => s + Math.exp(-l * POINTS[i].age), 0))
    const s: SeriesSpec[] = [
      {
        name: 'action pin',
        type: 'scatter',
        x: inAny.map((i) => POINTS[i].x),
        y: inAny.map((i) => POINTS[i].y),
        group: inAny.map((i) => group[i]),
        groupNames: clusters.map((_, g) => `cluster ${g + 1}`),
      },
      {
        name: 'in no cluster',
        type: 'scatter',
        x: out.map((i) => POINTS[i].x),
        y: out.map((i) => POINTS[i].y),
        muted: true,
      },
      {
        name: 'medoid',
        type: 'scatter',
        x: medoids.map((i) => POINTS[i].x),
        y: medoids.map((i) => POINTS[i].y),
        emphasis: true,
      },
    ]
    const order = clusters.map((c, g) => ({ g, n: c.length, w: importance[g] })).sort((a, b) => b.w - a.w)
    return { series: s, ranked: order, stray: out.length }
  }, [clusters, state.lambda])

  const xAxis = useAxis({ label: 'embedding dimension 1', range: [-4, 4] })
  const yAxis = useAxis({ label: 'embedding dimension 2', range: [-3.5, 3.5], equal: xAxis })
  return (
    <Figure
      title="Ward clusters, medoids and importance"
      state={state}
      caption={
        <MathText text="Points are one user's action pins in a two-dimensional stand-in for the embedding space: a recent interest on the left, an old one on the right, a mixed one below and three strays. Step the threshold $\alpha$ through the merge distances with the arrows. The clusters are the largest groups merged at distance at most $\alpha$; large ink points are their medoids. As in the published algorithm, a pin that merges only above $\alpha$ joins no cluster (grey). Importance sums $e^{-\lambda \cdot \mathrm{age}}$ over a cluster's pins, with age in days." />
      }

      readouts={
        <>
          <Readout label="clusters" value={clusters.length} />
          <Readout label="pins in no cluster" value={stray} />
          <Readout
            label="importance, highest first"
            value={ranked
              .slice(0, 4)
              .map((r) => `cluster ${r.g + 1} (${r.n} pins): ${formatNumber(r.w)}`)
              .join('; ')}
          />
        </>
      }
    >
      <Plot
        x={xAxis}
        y={yAxis}
        ariaLabel={"Scatter of a user's action pins coloured by Ward cluster, with medoids marked"}
      >
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
