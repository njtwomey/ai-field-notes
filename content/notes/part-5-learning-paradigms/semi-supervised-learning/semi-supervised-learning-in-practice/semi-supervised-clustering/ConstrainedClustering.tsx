import { useMemo, useState } from 'react'
import {
  Button,
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Readout,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { child, normals, stream, uniform } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import {
  constrainedKMeansSteps,
  constraintClosure,
  relevantComponentMetric,
  type ConstraintPair,
} from 'aifn-methods/unsupervised/clustering'

const PER_BAR = 60
const LENGTH_SD = 2.5
const WIDTH_SD = 0.25
const GAP = 1.6
const MAX_STEPS = 60
const X_RANGE: [number, number] = [-8, 8]
const Y_RANGE: [number, number] = [-2.5, 4]

/** Two long parallel bars: the classes. k-means with k = 2 prefers to cut them across. */
function bars(seed: number) {
  const s = stream(seed)
  const zx = toFlat(normals(child(s, 'along'), 2 * PER_BAR))
  const zy = toFlat(normals(child(s, 'across'), 2 * PER_BAR))
  const x = zx.map((v, i) => [LENGTH_SD * v, (i < PER_BAR ? 0 : GAP) + WIDTH_SD * zy[i]])
  const y = x.map((_, i) => (i < PER_BAR ? 0 : 1))
  return { x, y }
}

/** `count` random pairs, each a must-link when its two points share a bar and a cannot-link otherwise. */
function randomConstraints(seed: number, y: number[], count: number) {
  const u = toFlat(uniform(child(stream(seed), 'pairs'), 0, 1, { shape: [Math.max(1, 2 * count)] }))
  const mustLink: ConstraintPair[] = []
  const cannotLink: ConstraintPair[] = []
  for (let c = 0; c < count; c++) {
    const a = Math.floor(u[2 * c] * y.length)
    const b = Math.floor(u[2 * c + 1] * y.length)
    if (a === b) continue
    ;(y[a] === y[b] ? mustLink : cannotLink).push([a, b])
  }
  return { mustLink, cannotLink }
}

/**
 * Must-link and cannot-link constraints on two long bars: plain k-means, COP-k-means, PCK-means, and k-means in the
 * metric that relevant component analysis learns from the must-links.
 */
export function ConstrainedClustering() {
  const fig = useFigureState({
    method: choice(
      [
        { value: 'kmeans', label: 'k-means (constraints ignored)' },
        { value: 'cop', label: 'COP-k-means (hard constraints)' },
        { value: 'pck', label: 'PCK-means (penalised constraints)' },
        { value: 'rca', label: 'learned metric (RCA), then k-means' },
      ],
      'cop',
      { label: 'method' },
    ),
    constraints: int(30, { min: 0, max: 400, label: 'random constraints', suggestions: [0, 10, 30, 100, 200] }),
    weight: float(2, { gt: 0, le: 100, label: 'PCK-means cost per broken constraint w', suggestions: [0.5, 2, 10] }),
    click: choice(
      [
        { value: 'must', label: 'must-link' },
        { value: 'cannot', label: 'cannot-link' },
      ],
      'must',
      { label: 'two clicks add a' },
    ),
    seed: int(1, { min: 1, max: 999, label: 'data seed' }),
  })
  const { method, constraints, weight, click, seed } = fig
  const data = useMemo(() => bars(seed), [seed])
  const random = useMemo(() => randomConstraints(seed, data.y, constraints), [seed, data, constraints])
  const [added, setAdded] = useState<{ seed: number; mustLink: ConstraintPair[]; cannotLink: ConstraintPair[] }>({
    seed,
    mustLink: [],
    cannotLink: [],
  })
  const extra = added.seed === seed ? added : { seed, mustLink: [], cannotLink: [] }
  const [pending, setPending] = useState<number | null>(null)
  const [starts, setStarts] = useState<[number, number][]>([
    [-2, 0.8],
    [2, 0.8],
  ])
  const mustLink = useMemo(() => [...random.mustLink, ...extra.mustLink], [random, extra])
  const cannotLink = useMemo(() => [...random.cannotLink, ...extra.cannotLink], [random, extra])

  const states = useMemo(() => {
    const groups = constraintClosure(data.x.length, mustLink, cannotLink).members
    const hasChunklet = groups.some((g) => g.length > 1)
    const metric = method === 'rca' && hasChunklet ? relevantComponentMetric(data.x, groups) : null
    const space = metric ? metric.transformed : data.x
    const map = (p: readonly number[]) => (metric ? metric.whitening.map((w) => w[0] * p[0] + w[1] * p[1]) : [...p])
    const alg = constrainedKMeansSteps(space, {
      k: 2,
      rule: method === 'cop' ? 'cop' : 'pck',
      weight: method === 'pck' ? weight : 0,
      mustLink: method === 'cop' || method === 'pck' ? mustLink : [],
      cannotLink: method === 'cop' || method === 'pck' ? cannotLink : [],
    })
    // Run every method on the original constraints for the readout of broken constraints.
    const broken = (labels: Int32Array) =>
      mustLink.filter(([a, b]) => labels[a] >= 0 && labels[b] >= 0 && labels[a] !== labels[b]).length +
      cannotLink.filter(([a, b]) => labels[a] >= 0 && labels[a] === labels[b]).length
    const root = stream(0)
    let st = alg.init({ centroids: starts.map(map) }, root)
    const keep = (s: typeof st) => {
      const centres = [0, 1].map((h) => {
        const rows = data.x.filter((_, i) => s.labels[i] === h)
        return rows.length > 0
          ? [rows.reduce((a, r) => a + r[0], 0) / rows.length, rows.reduce((a, r) => a + r[1], 0) / rows.length]
          : null
      })
      let agree = 0
      let disagree = 0
      s.labels.forEach((l, i) => {
        if (l === data.y[i]) agree++
        else if (l >= 0) disagree++
      })
      return {
        t: s.t,
        labels: s.labels,
        centres,
        failed: s.failed,
        broken: broken(s.labels),
        accuracy: Math.max(agree, disagree) / s.labels.length,
        unplaced: s.labels.filter((l) => l < 0).length,
      }
    }
    const out = [keep(st)]
    for (let t = 0; t < MAX_STEPS && !st.converged; t++) {
      st = alg.step(st, { t, stream: root })
      out.push(keep(st))
    }
    return out
  }, [data, method, mustLink, cannotLink, weight, starts])

  const settings = `${method}/${constraints}/${weight}/${seed}/${mustLink.length}/${cannotLink.length}/${starts.flat().join(',')}`
  const [pos, setPos] = useState({ settings, at: 0 })
  const at = pos.settings === settings ? Math.min(pos.at, states.length - 1) : 0
  const s = states[at]

  const onPlotClick = ([px, py]: [number, number]) => {
    let best = -1
    let bestD = Infinity
    data.x.forEach(([a, b], i) => {
      const dd = (a - px) ** 2 + (b - py) ** 2
      if (dd < bestD) {
        bestD = dd
        best = i
      }
    })
    if (best < 0 || bestD > 0.25) return
    if (pending === null || pending === best) {
      setPending(best)
      return
    }
    const pair: ConstraintPair = [pending, best]
    setAdded({
      seed,
      mustLink: click === 'must' ? [...extra.mustLink, pair] : extra.mustLink,
      cannotLink: click === 'cannot' ? [...extra.cannotLink, pair] : extra.cannotLink,
    })
    setPending(null)
  }

  const xa = useAxis({ label: 'x₁', range: X_RANGE })
  const ya = useAxis({ label: 'x₂', range: Y_RANGE, equal: xa })
  const px = (i: number) => data.x[i][0]
  const py = (i: number) => data.x[i][1]
  const byCluster = [0, 1].map((h) => data.x.map((_, i) => i).filter((i) => s.labels[i] === h))
  const unplaced = data.x.map((_, i) => i).filter((i) => s.labels[i] < 0)
  const seg = (pairs: readonly ConstraintPair[]) =>
    pairs.map(([a, b]) => ({ from: [px(a), py(a)] as [number, number], to: [px(b), py(b)] as [number, number] }))
  const centres = s.centres.filter((c): c is number[] => c !== null)

  return (
    <Figure
      title="Constraints move the constrained points; a learned metric moves them all"
      state={fig}
      caption="Two long parallel bars of 60 points each are the two classes. Plain k-means with k = 2 cuts them across the middle, because that split has the lower inertia. Random pairs become must-links (green, same bar) or cannot-links (red dashed, different bars); click two points to add a constraint of the chosen kind. COP-k-means places each point, in index order, in the nearest cluster that breaks no constraint with the points already placed, and fails when a point has no legal cluster (red points are unplaced). PCK-means pays w per broken constraint instead. RCA learns a Mahalanobis metric from the must-link groups and runs plain k-means in it, so the constraints reshape the distance for every point. Drag the two starting centroids (ink) and step through the iterations with the player. With 30 constraints COP-k-means and PCK-means change little or fail; with 200, PCK-means finds the bars; the learned metric finds them with a handful of must-links."
      controls={
        <>
          <Player value={at} onChange={(v) => setPos({ settings, at: v })} count={states.length} label="iteration" />
          <Button
            variant="outline"
            size="sm"
            disabled={extra.mustLink.length + extra.cannotLink.length === 0}
            onClick={() => {
              setAdded({ seed, mustLink: [], cannotLink: [] })
              setPending(null)
            }}
          >
            Clear clicked constraints
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label="iteration" value={s.t} />
          <Readout label="accuracy against the bars" value={formatNumber(s.accuracy)} />
          <Readout label="constraints broken" value={`${s.broken} of ${mustLink.length + cannotLink.length}`} />
          <Readout label="unplaced points" value={s.failed ? `${s.unplaced} (COP-k-means failed)` : s.unplaced} />
        </>
      }
    >
      <Plot x={xa} y={ya} onPlotClick={onPlotClick}>
        {byCluster.map((rows, h) =>
          rows.length > 0 ? (
            <Points key={h} name={`cluster ${h + 1}`} x={rows.map(px)} y={rows.map(py)} slot={h} thin />
          ) : null,
        )}
        {unplaced.length > 0 && (
          <Points name="unplaced" x={unplaced.map(px)} y={unplaced.map(py)} tone="destructive" size={8} />
        )}
        {mustLink.length > 0 && <Segments name="must-link" segments={seg(mustLink)} tone="success" width={1.5} />}
        {cannotLink.length > 0 && (
          <Segments name="cannot-link" segments={seg(cannotLink)} tone="destructive" width={1.5} dashed />
        )}
        {pending !== null && (
          <Points name="first point of a new constraint" x={[px(pending)]} y={[py(pending)]} emphasis size={12} />
        )}
        {centres.length > 0 && (
          <Points name="cluster means" x={centres.map((c) => c[0])} y={centres.map((c) => c[1])} emphasis size={11} />
        )}
        {starts.map((p, h) => (
          <Handle
            key={h}
            kind="point"
            at={p}
            label={`start ${h + 1}`}
            onDrag={(q) => setStarts((old) => old.map((o, j) => (j === h ? q : o)) as [number, number][])}
          />
        ))}
      </Plot>
    </Figure>
  )
}
