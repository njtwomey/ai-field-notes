import { toFactorDiagram, type Bindings, type DiscreteFactorGraph, type Model } from 'aifn/inference/model'
import { type BeliefPropagationState } from 'aifn/inference/message-passing'
import { useMemo } from 'react'
import { Diagram, type DiagramEdge, type DiagramNode, type DiagramSpec } from '@lab/diagram'

export type FactorGraphViewProps = {
  /** A discrete factor graph, or a model description (expanded against `bindings`). */
  graph: DiscreteFactorGraph | Model
  bindings?: Bindings
  /** Fixed centres (grid units) of the variables: by index for a discrete graph, by instance key for a model. */
  positions?: Readonly<Record<string | number, readonly [number, number]>>
  /** Variable labels (TeX allowed), by index or instance key. */
  labels?: Readonly<Record<string | number, string>>
  /** Highlight a variable (index or instance key) and its Markov blanket. */
  highlight?: string | number
  /**
   * A belief-propagation state on the discrete graph: the messages updated by its last step ride on their edges
   * (pointing the way they travel), and each variable shows its belief (shaded by p(x = 1) when binary).
   */
  state?: BeliefPropagationState
  /** Which messages to show as chips: those updated by the last step (default), all, or none. */
  messages?: 'updated' | 'all' | 'none'
  /** Called with a variable's index (as a string) or instance key when it is clicked. */
  onVariableClick?: (variable: string) => void
  height?: number | 'fill'
  ariaLabel?: string
}

const fmt = (x: number) => (Math.abs(x) >= 0.995 || x === 0 ? x.toFixed(1) : x.toFixed(2))
const vec = (v: ArrayLike<number>) => (v.length === 2 ? fmt(v[1]) : `(${Array.from(v, fmt).join(', ')})`)

/** Keys of a record by the diagram's node names: `x<i>` for a discrete graph's variables, instance keys for a model. */
function byName<V>(record: Readonly<Record<string | number, V>> | undefined, discrete: boolean) {
  if (!record || !discrete) return record as Readonly<Record<string, V>> | undefined
  return Object.fromEntries(Object.entries(record).map(([k, v]) => [`x${k}`, v]))
}

/**
 * A factor graph drawn from `toFactorDiagram` (a structured graph's `toDiagram`), optionally with belief propagation's
 * messages and beliefs at one step. Binary messages show μ(x = 1); longer ones the whole vector. Click a variable to
 * choose it (e.g. for a Markov blanket).
 */
export function FactorGraphView({
  graph,
  bindings,
  positions,
  labels,
  highlight,
  state,
  messages = 'updated',
  onVariableClick,
  height,
  ariaLabel = 'A factor graph',
}: FactorGraphViewProps) {
  const discrete = 'cardinalities' in graph
  const spec = useMemo((): DiagramSpec => {
    const d: DiagramSpec = toFactorDiagram(graph, {
      bindings,
      positions: byName(positions, discrete),
      labels: byName(labels, discrete),
      highlight: highlight === undefined ? undefined : discrete ? `x${highlight}` : String(highlight),
    })
    if (!state) return d
    const updated = new Map(state.updated.map((u) => [u.edge, u.to]))
    const nodes = d.nodes.map((n): DiagramNode => {
      const v = /^x(\d+)$/.exec(n.id)
      const b = v ? state.beliefs[Number(v[1])]?.data : undefined
      if (!b) return n
      return { ...n, ...(b.length === 2 ? { shade: b[1] } : {}), notes: { s: `$${vec(b)}$` } }
    })
    const edges = state.edges.map((e, k): DiagramEdge => {
      const to = updated.get(k)
      const show = messages === 'all' || (messages === 'updated' && to !== undefined)
      const toVariable = (to ?? 'factor') === 'variable'
      const m = (toVariable ? state.toVariable : state.toFactor)[k].data
      return {
        from: `x${e.variable}`,
        to: `f${e.factor}`,
        route: 'straight',
        arrow: to !== undefined ? 'mid' : 'none',
        reverse: toVariable,
        state: to !== undefined ? 'active' : 'done',
        ...(show ? { note: { text: `$${vec(m)}$`, at: toVariable ? 0.35 : 0.65 } } : {}),
      }
    })
    return { ...d, nodes, edges }
  }, [graph, discrete, bindings, positions, labels, highlight, state, messages])
  const variables = useMemo(() => new Set(spec.nodes.filter((n) => n.shape !== 'factor').map((n) => n.id)), [spec])
  return (
    <Diagram
      spec={spec}
      ariaLabel={ariaLabel}
      height={height}
      onNodeClick={
        onVariableClick ? (id) => variables.has(id) && onVariableClick(discrete ? id.replace(/^x/, '') : id) : undefined
      }
    />
  )
}
