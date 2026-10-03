import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSlider, XYChart, useParam, type Handle, type Param } from 'aifn-render'
import { rng } from '@/lib/math'
import type { Point } from '../../_shared/datasets'
import { clusterSeries } from '../../_shared/groups'
import { covariance, eigSymmetric } from '../../_shared/linalg'
import { agglomerate, cutK, cut, dendrogram, type Linkage } from './agglomerate'
import { IRIS } from './iris'
import {
  boundingBox,
  copheneticCorrelation,
  gap,
  hopkins,
  jumpRatio,
  referenceTrees,
  stability,
  standardise,
  subsampleTrees,
  uniformInBox,
  type Rows,
} from './validity'

type Structured = 'iris' | 'blobs'
type Null = 'uniform' | 'gaussian'

const LINKAGES = [
  { value: 'single', label: 'single' },
  { value: 'complete', label: 'complete' },
  { value: 'average', label: 'average' },
  { value: 'ward', label: 'Ward' },
] as const satisfies readonly { value: Linkage; label: string }[]

const STRUCTURED = [
  { value: 'iris', label: 'Iris' },
  { value: 'blobs', label: '3 Gaussian clusters' },
] as const satisfies readonly { value: Structured; label: string }[]

const NULLS = [
  { value: 'uniform', label: 'uniform box' },
  { value: 'gaussian', label: 'one Gaussian' },
] as const satisfies readonly { value: Null; label: string }[]

/** Subsamples for stability, reference sets for the gap statistic, draws for Hopkins; kept small for slider drags. */
const B_STABILITY = 20
const B_GAP = 10
const K_MAX = 8

/** A seeded subsample of `n` of the rows, in their original order. */
function subsample(rows: Rows, n: number, seed: number): Rows {
  if (n >= rows.length) return rows
  const r = rng(seed)
  const order = rows.map((_, i) => i)
  for (let i = 0; i < n; i++) {
    const j = i + Math.floor(r.uniform() * (rows.length - i))
    ;[order[i], order[j]] = [order[j], order[i]]
  }
  return order
    .slice(0, n)
    .sort((a, b) => a - b)
    .map((i) => rows[i])
}

/**
 * Iris, or three unit-variance Gaussian clusters: centres on a triangle of side 6 in the first two coordinates and
 * drawn from N(0, 2²) in the others. Standardised either way.
 */
function structuredData(kind: Structured, n: number, d: number, seed: number): Rows {
  if (kind === 'iris') return standardise(subsample(IRIS, n, seed))
  const r = rng(seed)
  const centres = [
    [0, 0],
    [6, 0],
    [3, 5.2],
  ].map((t) => Array.from({ length: d }, (_, j) => (j < 2 ? t[j] : 2 * r.normal())))
  return standardise(Array.from({ length: n }, (_, i) => centres[i % 3].map((c) => c + r.normal())))
}

/** Data with no groups, the same size and dimension: uniform over the structured data's box, or one Gaussian. */
function nullData(kind: Null, like: Rows, seed: number): Rows {
  const r = rng(seed)
  const d = like[0].length
  const rows =
    kind === 'uniform'
      ? uniformInBox(like.length, boundingBox(like), r.uniform)
      : like.map(() => Array.from({ length: d }, () => r.normal()))
  return standardise(rows)
}

/** First two principal components, or the data itself in two dimensions. Signs fixed so the view does not flip. */
function project(x: Rows): Point[] {
  if (x[0].length === 2) return x.map((p) => [p[0], p[1]])
  const { vectors } = eigSymmetric(covariance(x))
  const axes = vectors.slice(0, 2).map((v) => {
    const big = v.reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a), 0)
    return big < 0 ? v.map((c) => -c) : v
  })
  return x.map((p) => [0, 1].map((a) => p.reduce((s, v, c) => s + v * axes[a][c], 0)) as Point)
}

/** Everything that depends on the data and linkage but not on the cut. */
function useTree(x: Rows, linkage: Linkage, seed: number) {
  return useMemo(() => {
    const merges = agglomerate(x, linkage)
    const r = rng(seed)
    const refs = referenceTrees(x, linkage, B_GAP, r.uniform)
    const subs = subsampleTrees(x, linkage, B_STABILITY, 0.8, r.uniform)
    // The gap statistic's rule: the smallest k with Gap(k) ≥ Gap(k + 1) − s(k + 1).
    const gaps = Array.from({ length: K_MAX + 1 }, (_, i) => gap(x, cutK(x.length, merges, i + 1), i + 1, refs))
    const chosen = gaps.findIndex((g, i) => i < K_MAX && g.gap >= gaps[i + 1].gap - gaps[i + 1].s) + 1
    const points = project(x)
    // Whole-unit limits, so the axis ends are round numbers.
    const pad = (v: number[]) => [Math.floor(Math.min(...v) - 0.2), Math.ceil(Math.max(...v) + 0.2)] as [number, number]
    return {
      merges,
      refs,
      subs,
      gapK: chosen || K_MAX,
      tree: dendrogram(x.length, merges),
      top: merges[merges.length - 1].height,
      coph: copheneticCorrelation(x, merges),
      hopkins: hopkins(x, Math.max(5, Math.round(x.length / 10)), 10, r.uniform),
      points,
      xRange: pad(points.map((p) => p[0])),
      yRange: pad(points.map((p) => p[1])),
    }
  }, [x, linkage, seed])
}

type Tree = ReturnType<typeof useTree>

function useCut(x: Rows, t: Tree, cutAt: Param) {
  const height = cutAt.value * t.top
  return useMemo(() => {
    const labels = cut(x.length, t.merges, height)
    const k = new Set(labels).size
    const counts = new Map<number, number>()
    labels.forEach((l) => counts.set(l, (counts.get(l) ?? 0) + 1))
    const sizes = [...counts.values()].sort((a, b) => b - a)
    return {
      sizes: sizes.length > 4 ? `${sizes.slice(0, 4).join(' / ')} / …` : sizes.join(' / '),
      height,
      labels,
      k,
      jump: jumpRatio(t.merges, k),
      gap: k < x.length ? gap(x, labels, k, t.refs) : null,
      stability: stability(labels, k, t.subs),
    }
  }, [x, t, height])
}

function Side({
  title,
  t,
  c,
  cutAt,
  dims,
}: {
  title: string
  t: Tree
  c: ReturnType<typeof useCut>
  cutAt: Param
  dims: number
}) {
  const n = t.points.length
  const xRange = useMemo(() => [-1, n] as [number, number], [n])
  // A round upper limit: the next multiple of a 1-2-5 step above the tallest merge.
  const yRange = useMemo(() => {
    const step = 10 ** Math.floor(Math.log10(t.top / 2))
    const unit = [1, 2, 5, 10].map((m) => m * step).find((u) => t.top / u <= 8)!
    return [0, Math.ceil((t.top * 1.02) / unit) * unit] as [number, number]
  }, [t.top])
  const treeSeries = useMemo(
    () => [{ name: 'dendrogram', type: 'line' as const, x: t.tree.x, y: t.tree.y, emphasis: true }],
    [t.tree],
  )
  const scatter = useMemo(() => clusterSeries(t.points, c.labels, 'singletons and small'), [t.points, c.labels])
  const handles: Handle[] = [{ kind: 'y', at: c.height, label: 'cut', onDrag: (y) => cutAt.set(y / t.top) }]
  const pc = dims > 2
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <div className="text-sm font-medium">{title}</div>
      <XYChart
        height={240}
        xLabel="leaf order"
        yLabel="merge height"
        xRange={xRange}
        yRange={yRange}
        handles={handles}
        series={treeSeries}
      />
      <XYChart
        height={240}
        xLabel={pc ? 'PC 1' : 'x₁'}
        yLabel={pc ? 'PC 2' : 'x₂'}
        xRange={t.xRange}
        yRange={t.yRange}
        series={scatter}
      />
    </div>
  )
}

const fmt = (v: number | undefined | null) => (v == null || !Number.isFinite(v) ? '—' : v.toFixed(2))

export function StructureOrNoise() {
  const [kind, setKind] = useState<Structured>('iris')
  const [nullKind, setNullKind] = useState<Null>('uniform')
  const [linkage, setLinkage] = useState<Linkage>('ward')
  const nPoints = useParam(150, { min: 30, max: 150, step: 10 })
  const dim = useParam(4, { min: 2, max: 10, step: 1 })
  const seed = useParam(1, { min: 1, max: 50, step: 1 })
  const cutA = useParam(0.6, { min: 0, max: 1, step: 0.005 })
  const cutB = useParam(0.6, { min: 0, max: 1, step: 0.005 })

  const d = kind === 'iris' ? 4 : dim.value
  const xa = useMemo(
    () => structuredData(kind, nPoints.value, dim.value, seed.value),
    [kind, nPoints.value, dim.value, seed.value],
  )
  const xb = useMemo(() => nullData(nullKind, xa, seed.value + 1000), [nullKind, xa, seed.value])
  const ta = useTree(xa, linkage, seed.value + 1)
  const tb = useTree(xb, linkage, seed.value + 2)
  const ca = useCut(xa, ta, cutA)
  const cb = useCut(xb, tb, cutB)

  const rows: { label: string; a: string; b: string }[] = [
    { label: 'clusters at the cut', a: String(ca.k), b: String(cb.k) },
    { label: 'cluster sizes', a: ca.sizes, b: cb.sizes },
    { label: 'jump ratio at the cut', a: fmt(ca.jump), b: fmt(cb.jump) },
    { label: 'cophenetic correlation', a: fmt(ta.coph), b: fmt(tb.coph) },
    { label: 'Hopkins statistic (0.5 = uniform)', a: fmt(ta.hopkins), b: fmt(tb.hopkins) },
    {
      label: 'gap at the cut ± s',
      a: ca.gap ? `${fmt(ca.gap.gap)} ± ${fmt(ca.gap.s)}` : '—',
      b: cb.gap ? `${fmt(cb.gap.gap)} ± ${fmt(cb.gap.s)}` : '—',
    },
    { label: `k chosen by the gap rule (1 to ${K_MAX})`, a: String(ta.gapK), b: String(tb.gapK) },
    { label: `stability, mean Jaccard (${B_STABILITY} subsamples)`, a: fmt(ca.stability), b: fmt(cb.stability) },
  ]

  return (
    <Interactive
      title="Structure or noise? The same pipeline on both"
      caption={`Left: data with groups (Iris, or three Gaussian clusters). Right: the same number of points in the same ${d} dimensions with no groups. Both go through one pipeline: standardise, build the tree, cut it. Drag the cut line on either dendrogram, or use the sliders; the scatter plots show the first two principal components when d > 2. The two trees look alike, and a cut always returns clusters. The jump ratio and the cophenetic correlation differ only in degree. The gap rule, which compares with uniform reference data, chooses k = 1 for noise, and cluster stability under subsampling is high only for real groups (check the cluster sizes: single linkage peels off stable outliers from noise too). A single Gaussian is not uniform, so the Hopkins statistic and the gap at the cut rate it as clustered: the null reference decides what counts as structure.`}
      controls={
        <>
          <ParamChoice label="structured data" value={kind} onChange={setKind} options={STRUCTURED} />
          <ParamChoice label="random data" value={nullKind} onChange={setNullKind} options={NULLS} />
          <ParamChoice label="linkage" value={linkage} onChange={setLinkage} options={LINKAGES} />
          <ParamSlider label="points" param={nPoints} />
          <ParamSlider label={kind === 'iris' ? 'dimension (Iris has 4)' : 'dimension'} param={dim} />
          <ParamSlider label="seed" param={seed} />
          <ParamSlider label="cut, structured (fraction of top merge)" param={cutA} />
          <ParamSlider label="cut, random (fraction of top merge)" param={cutB} />
        </>
      }
    >
      <div className="grid gap-6 md:grid-cols-2">
        <Side title="Structured data" t={ta} c={ca} cutAt={cutA} dims={d} />
        <Side title="Random data" t={tb} c={cb} cutAt={cutB} dims={d} />
      </div>
      <table className="mt-4 w-full text-xs">
        <thead className="text-muted-foreground">
          <tr className="border-b">
            <th className="py-1 text-left font-normal">check</th>
            <th className="py-1 text-right font-normal">structured</th>
            <th className="py-1 text-right font-normal">random</th>
          </tr>
        </thead>
        <tbody className="font-mono tabular-nums">
          {rows.map((r) => (
            <tr key={r.label} className="border-b border-border/50">
              <td className="py-1 font-sans text-muted-foreground">{r.label}</td>
              <td className="py-1 text-right">{r.a}</td>
              <td className="py-1 text-right">{r.b}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Interactive>
  )
}
