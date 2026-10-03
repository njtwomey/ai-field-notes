import { useState } from 'react'
import { MathText } from 'aifn-render'
import { Diagram } from 'aifn-render'
import { link, variable } from 'aifn-render'
import { Interactive, ParamChoice, Readout, StepControls } from 'aifn-render'

type Vertex = { id: string; label?: string; x: number; y: number }
type Problem = { nodes: Vertex[]; edges: [string, string][]; order: string[] }

const LEAVES = [1, 2, 3, 4].map((i) => `l${i}`)
const STAR_NODES: Vertex[] = [
  { id: 'c', x: 1, y: 1 },
  { id: 'l1', label: 'l_1', x: 0, y: 0 },
  { id: 'l2', label: 'l_2', x: 2, y: 0 },
  { id: 'l3', label: 'l_3', x: 2, y: 2 },
  { id: 'l4', label: 'l_4', x: 0, y: 2 },
]
const STAR_EDGES = LEAVES.map((l): [string, string] => ['c', l])

const PROBLEMS = {
  alarm: {
    // The moral graph: B and E share the factor p(A | B, E), so they are joined. A is the query and stays.
    nodes: [
      { id: 'B', x: 0, y: 0 },
      { id: 'E', x: 2, y: 0 },
      { id: 'A', x: 1, y: 1.2 },
    ],
    edges: [
      ['B', 'A'],
      ['E', 'A'],
      ['B', 'E'],
    ],
    order: ['E', 'B'],
  },
  leaves: { nodes: STAR_NODES, edges: STAR_EDGES, order: LEAVES },
  centre: { nodes: STAR_NODES, edges: STAR_EDGES, order: ['c', ...LEAVES] },
} satisfies Record<string, Problem>

type Key = keyof typeof PROBLEMS
const pairKey = (u: string, v: string) => (u < v ? `${u}|${v}` : `${v}|${u}`)

/** Whether the segment from u to v passes through the centre of some other node. */
function passesNode(p: Problem, u: string, v: string): boolean {
  const [a, b] = [u, v].map((id) => p.nodes.find((n) => n.id === id)!)
  return p.nodes.some((n) => {
    if (n === a || n === b) return false
    const t = ((n.x - a.x) * (b.x - a.x) + (n.y - a.y) * (b.y - a.y)) / ((b.x - a.x) ** 2 + (b.y - a.y) ** 2)
    const d = Math.hypot(a.x + t * (b.x - a.x) - n.x, a.y + t * (b.y - a.y) - n.y)
    return t > 0 && t < 1 && d < 0.3
  })
}

/** The graph after `steps` eliminations: remaining nodes, original and fill edges, and each step's neighbourhood. */
function eliminate(p: Problem, steps: number) {
  const adjacent = new Set(p.edges.map(([u, v]) => pairKey(u, v)))
  const fill = new Set<string>()
  const remaining = new Set(p.nodes.map((n) => n.id))
  const scopes: string[][] = []
  const neighbours = (v: string) => [...remaining].filter((u) => u !== v && adjacent.has(pairKey(u, v)))
  for (const v of p.order.slice(0, steps)) {
    const nb = neighbours(v)
    scopes.push(nb)
    for (const a of nb)
      for (const b of nb)
        if (a < b && !adjacent.has(pairKey(a, b))) {
          adjacent.add(pairKey(a, b))
          fill.add(pairKey(a, b))
        }
    remaining.delete(v)
  }
  const next = p.order[steps]
  return { adjacent, fill, remaining, scopes, next, nextNeighbours: next ? neighbours(next) : [] }
}

/** Step through an elimination order; each step joins the eliminated variable's neighbours with fill edges. */
export function EliminationSteps() {
  const [key, setKey] = useState<Key>('centre')
  const [steps, setSteps] = useState(0)
  const p: Problem = PROBLEMS[key]
  const state = eliminate(p, steps)
  const done = steps >= p.order.length
  const label = (id: string) => `$${p.nodes.find((n) => n.id === id)?.label ?? id}$`
  const set = (ids: string[]) => (ids.length ? `$\\{$${ids.map(label).join(', ')}$\\}$` : 'none')
  const width = Math.max(0, ...state.scopes.map((s) => s.length))
  const edges = [...state.adjacent]
    .map((k) => k.split('|'))
    .filter(([u, v]) => state.remaining.has(u) && state.remaining.has(v))
    .map(([u, v]) => {
      const fill = state.fill.has(pairKey(u, v))
      // A straight line through another node (a diagonal of the star) bows around it instead.
      return link(u, v, false, {
        dashed: fill,
        highlight: fill,
        ...(passesNode(p, u, v) && { route: 'curve' as const, bend: 0.9 }),
      })
    })
  // Eliminated variables stay in place, dashed and greyed, so the layout does not move between steps.
  const spec = {
    unit: 56,
    nodes: p.nodes.map((n) =>
      variable(
        n.id,
        n.x * 1.5,
        n.y * 1.5,
        label(n.id),
        state.remaining.has(n.id) ? { highlight: n.id === state.next } : { dashed: true, tone: 'neutral' },
      ),
    ),
    edges,
  }
  return (
    <Interactive
      title="Elimination order and fill edges"
      caption={
        <MathText text="Step eliminates the coloured variable. Its current neighbours become a clique; the dashed coloured edges are fill edges added by earlier steps. Eliminated variables stay in place, dashed and grey. In the star, eliminating the leaves first adds nothing, while eliminating the centre first joins every pair of leaves." />
      }
      controls={
        <>
          <ParamChoice
            label="graph and order"
            value={key}
            onChange={(k) => {
              setKey(k)
              setSteps(0)
            }}
            options={[
              { value: 'alarm', label: 'alarm: E, B' },
              { value: 'leaves', label: 'star: leaves first' },
              { value: 'centre', label: 'star: centre first' },
            ]}
          />
          <StepControls
            onStep={() => setSteps((s) => s + 1)}
            onRun={() => setSteps(p.order.length)}
            onReset={() => setSteps(0)}
            done={done}
          />
        </>
      }
      readout={
        <>
          <Readout label="order" value={<MathText text={p.order.map(label).join(', ')} />} />
          <Readout
            label="next"
            value={done ? 'done' : <MathText text={`${label(state.next!)}, neighbours ${set(state.nextNeighbours)}`} />}
          />
          <Readout label="fill edges" value={state.fill.size} />
          <Readout label="induced width so far" value={width} />
        </>
      }
    >
      <Diagram spec={spec} ariaLabel="Undirected graph during variable elimination" />
    </Interactive>
  )
}
