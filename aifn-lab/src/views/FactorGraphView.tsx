import {
  toFactorDiagram,
  type BeliefPropagationState,
  type Bindings,
  type DiscreteFactorGraph,
  type Model,
} from 'aifn/pgm'
import { useMemo } from 'react'
import { Diagram, type DiagramEdge, type DiagramNode, type DiagramSpec } from '@lab/diagram'

export type FactorGraphViewProps = {
  /** A discrete factor graph, or a model description (expanded against `bindings`). */
  graph: DiscreteFactorGraph | Model
  bindings?: Bindings
  /** Fixed centres (grid units) of the variables: by index for a discrete graph, by key for a model. */
  positions?: Readonly<Record<string | number, readonly [number, number]>>
  /** Variable labels (TeX allowed), by index or key. */
  labels?: Readonly<Record<string | number, string>>
  /** Highlight a variable (index or key) and its Markov blanket. */
  highlight?: string | number
  /**
   * A belief-propagation state on the discrete graph: the messages updated by its last step ride on their edges
   * (pointing the way they travel), and each variable shows its belief (shaded by p(x = 1) when binary).
   */
  state?: BeliefPropagationState
  /** Which messages to show as chips: those updated by the last step (default), all, or none. */
  messages?: 'updated' | 'all' | 'none'
  onVariableClick?: (variable: string) => void
  height?: number | 'fill'
  ariaLabel?: string
}

const fmt = (x: number) => (Math.abs(x) >= 0.995 || x === 0 ? x.toFixed(1) : x.toFixed(2))
const vec = (v: ArrayLike<number>) => (v.length === 2 ? fmt(v[1]) : `(${Array.from(v, fmt).join(', ')})`)

/**
 * A factor graph from `toFactorDiagram`, optionally with belief propagation's messages and beliefs at one step.
 * Binary messages show μ(x = 1); longer ones the whole vector. Click a variable to choose it (e.g. for a Markov
 * blanket); ids passed to `onVariableClick` are the variable's index or key.
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
  const spec = useMemo((): DiagramSpec => {
    const d = toFactorDiagram(graph, { bindings, positions, labels, highlight })
    if (!state) return d
    const updated = new Map(state.updated.map((u) => [u.edge, u.to]))
    const nodes = d.nodes.map((n): DiagramNode => {
      if (!n.id.startsWith('var ')) return n
      const b = state.beliefs[Number(n.id.slice(4))]?.data
      if (!b) return n
      return { ...n, ...(b.length === 2 ? { shade: b[1] } : {}), notes: { s: `$${vec(b)}$` } }
    })
    const edges = state.edges.map((e, k): DiagramEdge => {
      const to = updated.get(k)
      const show = messages === 'all' || (messages === 'updated' && to !== undefined)
      const toVariable = (to ?? 'factor') === 'variable'
      const m = (toVariable ? state.toVariable : state.toFactor)[k].data
      return {
        from: `var ${e.variable}`,
        to: `factor ${e.factor}`,
        route: 'straight',
        arrow: to !== undefined ? 'mid' : 'none',
        reverse: toVariable,
        state: to !== undefined ? 'active' : 'done',
        ...(show ? { note: { text: `$${vec(m)}$`, at: toVariable ? 0.35 : 0.65 } } : {}),
      }
    })
    return { ...d, nodes, edges }
  }, [graph, bindings, positions, labels, highlight, state, messages])
  return (
    <Diagram
      spec={spec}
      ariaLabel={ariaLabel}
      height={height}
      onNodeClick={onVariableClick ? (id) => id.startsWith('var ') && onVariableClick(id.slice(4)) : undefined}
    />
  )
}
