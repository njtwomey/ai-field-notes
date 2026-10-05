/** Graph data helpers for the graph specimens. */
import { fromEdges, type Graph } from 'aifn-compute/graph'
import type { Trace } from 'aifn-compute/foundation/trace'
import type { ElementState, Point } from 'aifn-render/diagram'

/** A graph over lettered nodes from edges written 'ab' (a → b), with optional weights. */
export function lettered(
  nodes: string,
  edges: readonly (string | readonly [string, number])[],
  directed = true,
): Graph {
  const at = (c: string) => nodes.indexOf(c)
  return fromEdges(
    nodes.length,
    edges.map((e): [number, number] | [number, number, number] =>
      typeof e === 'string' ? [at(e[0]), at(e[1])] : [at(e[0][0]), at(e[0][1]), e[1]],
    ),
    { directed, labels: nodes.split('') },
  )
}

/** Positions for lettered nodes from a map of letter → [x, y]. */
export const placed = (nodes: string, at: Record<string, [number, number]>): Point[] =>
  nodes.split('').map((c) => ({ x: at[c][0], y: at[c][1] }))

/** The kept state of a trace at player position i (clamped to its last state). */
export const stateAt = <S>(t: Trace<S>, i: number): S => t.steps[Math.min(i, t.steps.length - 1)]

/** Node state from a CLRS colour: white idle, grey active, black done. */
export const colourState = (c: number): ElementState => (c === 0 ? 'idle' : c === 1 ? 'active' : 'done')
