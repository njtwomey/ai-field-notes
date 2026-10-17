import { useMemo } from 'react'
import {
  choice,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  Segments,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { circles, moons, semiSupervisedSplit } from 'aifn-methods/data/synthetic'
import { harmonicLabels, neighbourhoodGraph, type NeighbourhoodRule } from 'aifn-compute/graph/propagation'
import { child, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'

const N = 150
/** Edges drawn at most; the full graph keeps its strongest ones in the picture (all of them in the computation). */
const MAX_DRAWN = 2500
type Rule = NeighbourhoodRule['kind']

/**
 * A similarity graph built on two moons or two circles by a chosen neighbourhood rule and edge weight, with the edges
 * that join the two classes in red and the harmonic label-propagation result on the nodes.
 */
export function GraphBuilder() {
  const fig = useFigureState({
    shape: choice(
      [
        { value: 'moons', label: 'two moons' },
        { value: 'circles', label: 'two circles' },
      ],
      'moons',
      { label: 'data' },
    ),
    rule: choice<Rule>(
      [
        { value: 'knn', label: 'k-nearest neighbours (either)' },
        { value: 'mutual-knn', label: 'mutual k-nearest neighbours' },
        { value: 'epsilon', label: 'ε-ball' },
        { value: 'b-matching', label: 'b-matching (greedy)' },
        { value: 'full', label: 'full graph' },
      ],
      'knn',
      { label: 'neighbourhood rule' },
    ),
    k: int(6, {
      min: 1,
      max: 60,
      label: 'neighbours k',
      suggestions: [2, 6, 15, 40],
      when: (v) => v.rule === 'knn' || v.rule === 'mutual-knn',
    }),
    epsilon: float(0.3, { gt: 0, le: 3, step: 0.05, label: 'radius ε', when: (v) => v.rule === 'epsilon' }),
    b: int(4, { min: 1, max: 30, label: 'edges per node b', when: (v) => v.rule === 'b-matching' }),
    weight: choice(
      [
        { value: 'heat', label: 'Gaussian of the distance' },
        { value: 'binary', label: '1 per edge' },
      ],
      'heat',
      { label: 'edge weight' },
    ),
    sigma: float(0.15, { gt: 0, le: 3, step: 0.05, label: 'kernel width σ', when: (v) => v.weight === 'heat' }),
    perClass: int(3, { min: 1, max: 20, label: 'labels per class', suggestions: [1, 3, 10] }),
    seed: int(1, { min: 1, max: 999, label: 'data seed' }),
  })
  const { shape, rule, k, epsilon, b, weight, sigma, perClass, seed } = fig

  const data = useMemo(() => {
    const d =
      shape === 'moons'
        ? moons(stream(seed), { n: N, noise: 0.1, spacing: 'random' })
        : circles(stream(seed), { n: N, noise: 0.06, factor: 0.5 })
    const y = toFlat(d.y)
    const xy = toFlat(d.x)
    const split = semiSupervisedSplit(child(stream(seed), 'split'), y, perClass)
    return { x: d.x, y, split, px: y.map((_, i) => xy[2 * i]), py: y.map((_, i) => xy[2 * i + 1]) }
  }, [shape, seed, perClass])

  const result = useComputed(() => {
    const r: NeighbourhoodRule =
      rule === 'knn' || rule === 'mutual-knn'
        ? { kind: rule, k }
        : rule === 'epsilon'
          ? { kind: 'epsilon', epsilon }
          : rule === 'b-matching'
            ? { kind: 'b-matching', b }
            : { kind: 'full' }
    const g = neighbourhoodGraph(data.x, r, weight === 'heat' ? { kind: 'heat', sigma } : { kind: 'binary' })
    const pred = toFlat(harmonicLabels(g.affinity, data.split.labels, { classes: 2 }).labels)
    const U = data.split.unlabelled
    const crossing = g.edges.filter((e) => data.y[e.from] !== data.y[e.to])
    const totalWeight = g.edges.reduce((s, e) => s + e.weight, 0)
    const crossingWeight = crossing.reduce((s, e) => s + e.weight, 0)
    const drawn =
      g.edges.length > MAX_DRAWN ? [...g.edges].sort((p, q) => q.weight - p.weight).slice(0, MAX_DRAWN) : g.edges
    return {
      pred,
      edges: g.edges.length,
      crossing: crossing.length,
      crossingShare: totalWeight > 0 ? crossingWeight / totalWeight : 0,
      isolated: g.counts.filter((c) => c === 0).length,
      unreached: U.filter((i) => pred[i] < 0).length,
      accuracy: U.filter((i) => pred[i] === data.y[i]).length / U.length,
      within: drawn.filter((e) => data.y[e.from] === data.y[e.to]),
      across: drawn.filter((e) => data.y[e.from] !== data.y[e.to]),
      truncated: g.edges.length > MAX_DRAWN,
    }
  }, [data, rule, k, epsilon, b, weight, sigma])
  const r = result.value

  const segment = (e: { from: number; to: number }) => ({
    from: [data.px[e.from], data.py[e.from]] as const,
    to: [data.px[e.to], data.py[e.to]] as const,
  })
  const U = data.split.unlabelled
  const L = data.split.labelled
  const xa = useAxis({ label: 'x₁', range: shape === 'moons' ? [-1.5, 2.5] : [-1.4, 1.4] })
  const ya = useAxis({ label: 'x₂', range: shape === 'moons' ? [-1, 1.5] : [-1.4, 1.4], equal: xa })

  return (
    <Figure
      title="The graph decides where labels flow"
      state={fig}
      caption={`${N} points, ${perClass} labelled per class (black). Grey segments join points of the same class; red segments join the two classes, and every red edge is a path for labels to leak. Points are coloured by the harmonic label-propagation result on the graph; points in a component with no label stay unassigned (light grey). Small k, ε or b split the data into pieces, some without a label. Large values add red edges, and binary weights let a long red edge count as much as a short one. A Gaussian weight with small σ makes long edges nearly free to cut, which repairs much of the damage.${r.truncated ? ` The picture shows the ${MAX_DRAWN} strongest edges.` : ''}`}
      readouts={
        <>
          <Readout label="edges" value={r.edges} />
          <Readout label="edges across classes" value={r.crossing} />
          <Readout label="weight across classes" value={formatNumber(r.crossingShare)} />
          <Readout label="unlabelled points not reached" value={r.unreached} />
          <Readout label="accuracy on unlabelled" value={formatNumber(r.accuracy)} />
        </>
      }
    >
      <Plot x={xa} y={ya}>
        {r.within.length > 0 && (
          <Segments
            name="edge within a class"
            segments={r.within.map(segment)}
            muted
            width={0.6}
            stale={result.stale}
          />
        )}
        {r.across.length > 0 && (
          <Segments
            name="edge across classes"
            segments={r.across.map(segment)}
            tone="destructive"
            width={1.2}
            stale={result.stale}
          />
        )}
        {[0, 1].map((c) => {
          const rows = U.filter((i) => r.pred[i] === c)
          return rows.length > 0 ? (
            <Points
              key={c}
              name={`assigned class ${c}`}
              x={rows.map((i) => data.px[i])}
              y={rows.map((i) => data.py[i])}
              slot={c}
            />
          ) : null
        })}
        {U.some((i) => r.pred[i] < 0) && (
          <Points
            name="not reached"
            x={U.filter((i) => r.pred[i] < 0).map((i) => data.px[i])}
            y={U.filter((i) => r.pred[i] < 0).map((i) => data.py[i])}
            muted
          />
        )}
        <Points name="labelled" x={L.map((i) => data.px[i])} y={L.map((i) => data.py[i])} emphasis size={10} />
      </Plot>
    </Figure>
  )
}
