import { useMemo, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  mineAntecedents,
  corels,
  type CorelsNode,
  type NodeStatus,
  nominalConditions,
  ruleText,
} from 'aifn-compute/learning/rules'
import { titanicRows, TITANIC_LEVELS, TITANIC_NAMES } from '../_shared/rule-figures'

const LENGTHS = [
  { value: 1, label: 'one condition (8 antecedents)' },
  { value: 2, label: 'up to two conditions (26 antecedents)' },
] as const
const ORDERS = [
  { value: 'breadth', label: 'breadth first' },
  { value: 'best', label: 'best first (smallest bound)' },
] as const

const PRUNED: NodeStatus[] = ['bound', 'lookahead', 'support', 'accurate-support', 'permutation', 'equivalent-points']
const REASON: Record<NodeStatus, string> = {
  queued: 'queued',
  expanded: 'evaluated and expanded',
  leaf: 'evaluated, no children survive',
  bound: 'pruned: lower bound ≥ best risk',
  lookahead: 'evaluated, not expanded: bound + λ ≥ best risk',
  support: 'pruned: captures < λ of the rows',
  'accurate-support': 'pruned: classifies < λ correctly',
  permutation: 'pruned: a permutation has a smaller bound',
  'equivalent-points': 'pruned: bound plus unavoidable errors ≥ best risk',
}

/** Leaf-order x positions of a tree given by parent links: leaves left to right, parents centred over children. */
function layout(nodes: readonly CorelsNode[]): number[] {
  const children: number[][] = nodes.map(() => [])
  for (const n of nodes) if (n.parent >= 0) children[n.parent].push(n.id)
  const x = new Array<number>(nodes.length).fill(0)
  let next = 0
  const visit = (id: number) => {
    const c = children[id]
    if (c.length === 0) {
      x[id] = next++
      return
    }
    for (const k of c) visit(k)
    x[id] = (x[c[0]] + x[c[c.length - 1]]) / 2
  }
  if (nodes.length > 0) visit(0)
  return x
}

/** CORELS's branch and bound on the Titanic data: the prefix tree as it is generated, pruned and evaluated. */
export function CorelsSearch() {
  const fig = useFigureState({
    lambda: float(0.005, {
      min: 0.001,
      max: 0.05,
      step: 0.001,
      label: 'penalty per rule λ',
      suggestions: [0.002, 0.005, 0.01, 0.02],
    }),
    maxLength: choice(LENGTHS, 1, { label: 'antecedents' }),
    order: choice<'breadth' | 'best'>(ORDERS, 'breadth', { label: 'queue order' }),
  })
  const { lambda, maxLength, order } = fig

  const data = useMemo(() => titanicRows(), [])
  const ants = useMemo(
    () => mineAntecedents(data.rows, nominalConditions([4, 2, 2]), { minSupport: 0.01, maxLength }),
    [data, maxLength],
  )
  const result = useMemo(() => corels(ants, data.y, { lambda, order, maxNodes: 20000 }), [ants, data, lambda, order])
  const xs = useMemo(() => layout(result.nodes), [result])
  const text = (prefix: readonly number[]) =>
    prefix.length === 0
      ? '(empty list)'
      : prefix.map((j) => ruleText(ants[j].conditions, { names: TITANIC_NAMES, levels: TITANIC_LEVELS })).join(' → ')

  const settings = `${lambda}/${maxLength}/${order}`
  const [pos, setPos] = useState({ settings, at: 0 })
  const steps = result.steps
  const at = pos.settings === settings ? Math.min(pos.at, steps.length - 1) : 0
  const s = steps[at]
  const t = at + 1 // search steps are numbered from 1

  const statusAt = (n: CorelsNode): NodeStatus | null => (n.generated > t ? null : n.decided <= t ? n.status : 'queued')
  const view = useMemo(() => {
    const groups: Record<'queued' | 'evaluated' | 'pruned', number[]> = { queued: [], evaluated: [], pruned: [] }
    const counts: Partial<Record<NodeStatus, number>> = {}
    const edges: { from: [number, number]; to: [number, number] }[] = []
    for (const n of result.nodes) {
      const st = statusAt(n)
      if (!st) continue
      counts[st] = (counts[st] ?? 0) + 1
      if (st === 'queued') groups.queued.push(n.id)
      else if (PRUNED.includes(st) && st !== 'lookahead') groups.pruned.push(n.id)
      else groups.evaluated.push(n.id)
      if (n.parent >= 0) edges.push({ from: [xs[n.parent], n.prefix.length - 1], to: [xs[n.id], n.prefix.length] })
    }
    return { groups, counts, edges }
  }, [result, xs, t]) // eslint-disable-line react-hooks/exhaustive-deps

  const bestPath = useMemo(() => {
    if (!s) return []
    const ids: number[] = []
    for (const n of result.nodes) {
      if (n.prefix.length <= s.best.length && n.prefix.every((j, k) => j === s.best[k])) ids.push(n.id)
    }
    return ids
  }, [result, s])

  const depth = Math.max(1, ...result.nodes.map((n) => n.prefix.length))
  const width = Math.max(1, ...xs)
  const xa = useAxis({ label: 'prefixes, in tree order', range: [-1, width + 1], nice: false })
  const ya = useAxis({
    label: 'rules in the prefix',
    range: [-0.4, depth + 0.4],
    nice: false,
    inverse: true,
    integer: true,
  })
  const xb = useAxis({ label: 'search step', range: [1, Math.max(2, steps.length)], integer: true })
  const yb = useAxis({ label: 'regularised risk', hold: 'union' })
  if (!s) return null
  const popped = result.nodes[s.node]
  const pts = (ids: number[]) => ({ x: ids.map((i) => xs[i]), y: ids.map((i) => result.nodes[i].prefix.length) })
  const upTo = steps.slice(0, at + 1)

  return (
    <Figure
      title="Branch and bound over rule lists"
      state={fig}
      caption={`CORELS on the Titanic data: the root is the empty list, and each child appends one more antecedent to its parent's prefix. Each step pops one prefix from the queue (ink), computes its lower bound b (errors of the rows its rules capture, plus λ per rule) and, if b is below the best risk found so far, evaluates the list with that prefix and a default rule, then generates its children. Children that cannot belong to an optimal list are pruned at once (grey): too little support, too few correct rows, a permutation with a smaller bound, or a bound that already reaches the best risk once the unavoidable errors of identical passengers with different fates are added. Coloured points are evaluated or still queued; the ink path is the best list so far. Right: the best risk found and the bound of each popped prefix. When the queue empties the best list is certified optimal over all lists of these antecedents. Step through with the player; lower λ to see the tree grow.`}
      controls={<Player value={at} onChange={(v) => setPos({ settings, at: v })} count={steps.length} label="step" />}
      readouts={{
        [`step ${t}`]: (
          <>
            <Readout label="popped prefix" value={text(popped.prefix)} />
            <Readout
              label="its bound b, its risk R"
              value={`${formatNumber(popped.bound)}, ${Number.isFinite(popped.objective) && popped.decided <= t ? formatNumber(popped.objective) : '—'}`}
            />
            <Readout label="outcome" value={REASON[popped.status]} />
            <Readout label="best list so far" value={`${text(s.best)} (risk ${formatNumber(s.bestObjective)})`} />
            <Readout label="queue" value={s.queue} />
          </>
        ),
        'prefixes so far': (
          <>
            {Object.entries(view.counts).map(([k, v]) => (
              <Readout key={k} label={REASON[k as NodeStatus]} value={v} />
            ))}
            <Readout
              label="certified at the end"
              value={result.certified ? `yes, after ${steps.length} steps` : 'no (node budget reached)'}
            />
          </>
        ),
      }}
    >
      <Plots cols={2} widths={[64, 36]}>
        <Plot x={xa} y={ya}>
          {view.edges.length > 0 && <Segments name="tree" segments={view.edges} muted width={0.8} />}
          {view.groups.pruned.length > 0 && <Points name="pruned" {...pts(view.groups.pruned)} muted thin />}
          {view.groups.evaluated.length > 0 && <Points name="evaluated" {...pts(view.groups.evaluated)} slot={1} />}
          {view.groups.queued.length > 0 && <Points name="queued" {...pts(view.groups.queued)} slot={0} />}
          {bestPath.length > 0 && <Points name="best list so far" {...pts(bestPath)} emphasis size={9} />}
          <Points name="popped" x={[xs[popped.id]]} y={[popped.prefix.length]} emphasis size={14} shape={3} />
        </Plot>
        <Plot x={xb} y={yb}>
          <Curve name="best risk so far" x={upTo.map((_, k) => k + 1)} y={upTo.map((u) => u.bestObjective)} emphasis />
          <Points
            name="bound of the popped prefix"
            x={upTo.map((_, k) => k + 1)}
            y={upTo.map((u) => result.nodes[u.node].bound)}
            slot={2}
            thin
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
