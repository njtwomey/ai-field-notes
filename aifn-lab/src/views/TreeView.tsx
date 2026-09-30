import type { Tree } from 'aifn/graph'
import { useMemo, useState } from 'react'
import {
  Diagram,
  treeLayout,
  type DiagramEdge,
  type DiagramNode,
  type DiagramSpec,
  type ElementState,
  type Shape,
  type Side,
  type Tone,
} from '@lab/diagram'
import type { PerItem } from './GraphView'

const at = <T,>(p: PerItem<T> | undefined, i: number): T | undefined =>
  p === undefined ? undefined : typeof p === 'function' ? p(i) : p[i]

/** Rough width in grid units of a label at the diagram's label size: maths counts by its visible characters. */
function textWidth(text: string): number {
  const plain = text
    .replace(/\$([^$]*)\$/g, (_, m: string) =>
      m
        .replace(/\\[a-zA-Z]+/g, 'x')
        .replace(/[{}^_\\ ]/g, '')
        .replace(/./g, 'x'),
    )
    .split('\n')
    .reduce((a, b) => (a.length >= b.length ? a : b), '')
  return (plain.length * 13 * 0.55 + 14) / 40
}

export type TreeViewProps = {
  /** An `aifn/graph` `Tree`. Node labels come from `node.label`, else the id. */
  tree: Tree
  ariaLabel?: string
  /** `down` (root at the top, default) or `right` (root at the left). */
  orientation?: 'down' | 'right'
  /**
   * Place nodes by their `height` (a dendrogram: leaves at the bottom, merges at their distance) with elbow edges,
   * instead of by depth with straight edges. A number sets grid units per unit of height.
   */
  heightAxis?: boolean | number
  /** Gap between neighbouring nodes, and between levels, in grid units. */
  siblingGap?: number
  levelGap?: number
  /** Node labels (TeX in `$…$`), overriding `node.label`; an empty string draws an unlabelled node. */
  nodeLabels?: PerItem<string>
  /**
   * A note beside each node, e.g. a probability: a string goes below a leaf and beside an internal node (outside the
   * edges), or give the sides explicitly.
   */
  nodeNotes?: PerItem<string | Partial<Record<Side, string>>>
  /** Edge labels by child id (the edge parent → child), overriding `tree.edges[child].label`. */
  edgeLabels?: PerItem<string>
  /** Step state of each node, and of each edge by child id (`idle` dimmed, `active` emphasised, `done` plain). */
  nodeState?: PerItem<ElementState>
  edgeState?: PerItem<ElementState>
  nodeTone?: PerItem<Tone>
  edgeTone?: PerItem<Tone>
  /** Nodes to highlight; an edge is highlighted when both its ends are, so a path lights up as a path. */
  highlight?: Iterable<number>
  /** Nodes not drawn (with their edges), keeping the layout of the whole tree, e.g. nodes not yet created. */
  hidden?: PerItem<boolean>
  /** Node shape; default circles, or boxes when a label is wider than a circle. Leaves may differ from the rest. */
  shape?: Shape | ((id: number) => Shape)
  /** Clicking an internal node collapses or expands it; collapsed nodes show how many nodes they hide. */
  collapsible?: boolean
  /** Nodes collapsed at first (with `collapsible`), or every node at this depth. */
  defaultCollapsed?: readonly number[] | { depth: number }
  /** Further overrides of each node's or edge's (by child id) diagram spec. */
  node?: (id: number) => Partial<DiagramNode>
  edge?: (child: number) => Partial<DiagramEdge>
  /** Height in pixels, or `fill`; by default the Figure frame's height. */
  height?: number | 'fill'
  onNodeClick?: (id: number) => void
  onNodeHover?: (id: number | null) => void
}

const nid = (v: number) => `t${v}`

/**
 * An `aifn/graph` `Tree` drawn with the lab's diagram system and the tidy `treeLayout`: parents centred over their
 * children, binary trees keeping left and right, or a dendrogram by node height with elbow edges. Nodes carry KaTeX
 * labels and a note (e.g. a probability), edges a label (e.g. 0/1); nodes and edges take step states, tones and a
 * highlighted path; hidden nodes keep their place; internal nodes can collapse. It sizes from the `Figure` frame.
 */
export function TreeView({
  tree,
  ariaLabel = 'A tree',
  orientation = 'down',
  heightAxis = false,
  siblingGap,
  levelGap,
  nodeLabels,
  nodeNotes,
  edgeLabels,
  nodeState,
  edgeState,
  nodeTone,
  edgeTone,
  highlight,
  hidden,
  shape,
  collapsible = false,
  defaultCollapsed,
  node,
  edge,
  height,
  onNodeClick,
  onNodeHover,
}: TreeViewProps) {
  const [collapsed, setCollapsed] = useState<ReadonlySet<number>>(() => {
    if (!collapsible || !defaultCollapsed) return new Set()
    if ('depth' in defaultCollapsed) {
      const out = new Set<number>()
      const walk = (v: number, d: number) => {
        if (d === defaultCollapsed.depth && tree.nodes[v].children.length) out.add(v)
        else tree.nodes[v].children.forEach((c) => walk(c, d + 1))
      }
      walk(tree.root, 0)
      return out
    }
    return new Set(defaultCollapsed)
  })
  // A collapse set from another tree is dropped rather than applied to the wrong nodes.
  const live = useMemo(
    () => new Set([...collapsed].filter((v) => v < tree.nodes.length && tree.nodes[v].children.length)),
    [collapsed, tree],
  )

  const spec = useMemo((): DiagramSpec => {
    const down = orientation === 'down'
    const leaf = (v: number) => tree.nodes[v].children.length === 0 || live.has(v)
    const labelOf = (v: number) => at(nodeLabels, v) ?? tree.nodes[v].label ?? String(v)
    const labels = tree.nodes.map((n) => labelOf(n.id))
    const shapeOf = (v: number): Shape => {
      if (typeof shape === 'function') return shape(v)
      if (shape) return shape
      return textWidth(labels[v]) > 0.95 ? 'box' : 'circle'
    }
    // Hidden descendants of each collapsed node.
    const hiddenCount = new Map<number, number>()
    for (const v of live) {
      let count = 0
      const stack = [...tree.nodes[v].children]
      while (stack.length) {
        const w = stack.pop()!
        count++
        stack.push(...tree.nodes[w].children)
      }
      hiddenCount.set(v, count)
    }
    const notesOf = (v: number): Partial<Record<Side, string>> | undefined => {
      const note = at(nodeNotes, v)
      const extra = hiddenCount.has(v) ? `+${hiddenCount.get(v)}` : undefined
      if (note === undefined) return extra ? { [down ? 's' : 'e']: extra } : undefined
      if (typeof note !== 'string') return extra ? { ...note, [down ? 's' : 'e']: extra } : note
      const side: Side = leaf(v) ? (down ? 's' : 'e') : down ? 'e' : 'n'
      return extra && side === (down ? 's' : 'e')
        ? { [side]: `${note} (${extra})` }
        : { [side]: note, ...(extra && { [down ? 's' : 'e']: extra }) }
    }
    const notes = tree.nodes.map((n) => notesOf(n.id))
    // Sizes: a circle grows to fit its label; a box is as wide as its label. Across the levels a node also needs room
    // for notes above or below it (down) or beside it (right).
    const dims = tree.nodes.map((n) => {
      const s = shapeOf(n.id)
      const w = textWidth(labels[n.id])
      if (s === 'circle') {
        const d = Math.max(0.9, w)
        return [d, d]
      }
      if (s === 'dot') return [0.14, 0.14]
      return [Math.max(0.9, w), 0.7]
    })
    const acrossSize = (v: number) => {
      const [w, h] = dims[v]
      const ns = notes[v] ?? {}
      if (down) {
        const below = Math.max(ns.s ? textWidth(ns.s) : 0, ns.n ? textWidth(ns.n) : 0)
        const beside = (ns.e ? textWidth(ns.e) + 0.1 : 0) + (ns.w ? textWidth(ns.w) + 0.1 : 0)
        return Math.max(w, below) + 2 * beside
      }
      return h + (ns.n ? 0.5 : 0) + (ns.s ? 0.5 : 0)
    }
    const maxAlong = Math.max(
      ...tree.nodes.map((n) => {
        const [w, h] = dims[n.id]
        const ns = notes[n.id] ?? {}
        return down ? h + (ns.s ? 0.5 : 0) + (ns.n ? 0.5 : 0) : w + (ns.e ? textWidth(ns.e) : 0)
      }),
    )
    const labelled = tree.nodes.some((n) => n.parent !== null && (at(edgeLabels, n.id) ?? tree.edges[n.id]?.label))
    const positions = treeLayout(tree, {
      orientation,
      siblingGap: siblingGap ?? 0.35,
      // Labelled edges need room between levels for their labels beside the notes above and below.
      levelGap: levelGap ?? maxAlong + (down ? 0.8 : 1.2) + (labelled ? 0.5 : 0),
      size: acrossSize,
      heightAxis,
      collapsed: live,
    })
    const lit = new Set(highlight ?? [])
    const visible = (v: number) => positions[v] !== null && !at(hidden, v)
    const nodes = tree.nodes
      .filter((n) => visible(n.id))
      .map((n): DiagramNode => {
        const p = positions[n.id]!
        const [w, h] = dims[n.id]
        return {
          id: nid(n.id),
          shape: shapeOf(n.id),
          x: p.x,
          y: p.y,
          w,
          h,
          label: labels[n.id] || undefined,
          tone: at(nodeTone, n.id) ?? 'ink',
          state: at(nodeState, n.id),
          highlight: lit.has(n.id) || undefined,
          notes: notes[n.id],
          dashed: live.has(n.id) || undefined,
          ...node?.(n.id),
        }
      })
    const edges = tree.nodes
      .filter((n) => n.parent !== null && visible(n.id) && visible(n.parent))
      .map((n): DiagramEdge => {
        const c = n.id
        const p = n.parent!
        const pc = positions[c]!
        const pp = positions[p]!
        const label = at(edgeLabels, c) ?? tree.edges[c]?.label
        // A dendrogram's edge leaves its parent sideways and drops to the child at a right angle.
        const elbow = heightAxis !== false
        let from = nid(p)
        if (elbow) {
          const offset = down ? pc.x - pp.x : pc.y - pp.y
          from +=
            Math.abs(offset) < 1e-9 ? (down ? ':s' : ':e') : offset < 0 ? (down ? ':w' : ':n') : down ? ':e' : ':s'
        }
        return {
          from,
          to: elbow ? `${nid(c)}:${down ? 'n' : 'w'}` : nid(c),
          route: elbow ? 'ortho' : 'straight',
          arrow: 'none',
          label,
          labelRotate: false,
          // Labels sit outside the fork: left of a left child's edge, right of a right child's.
          labelSide: (down ? pc.x < pp.x : pc.y > pp.y) ? 'right' : 'left',
          ...(label !== undefined && !elbow && { labelPos: 0.45 }),
          tone: at(edgeTone, c),
          state: at(edgeState, c),
          highlight: (lit.has(c) && lit.has(p)) || undefined,
          ...edge?.(c),
        }
      })
    return { nodes, edges }
  }, [
    tree,
    orientation,
    heightAxis,
    siblingGap,
    levelGap,
    nodeLabels,
    nodeNotes,
    edgeLabels,
    nodeState,
    edgeState,
    nodeTone,
    edgeTone,
    highlight,
    hidden,
    shape,
    live,
    node,
    edge,
  ])

  const toggle = (v: number) =>
    setCollapsed((prev) => {
      const next = new Set(prev)
      if (next.has(v)) next.delete(v)
      else next.add(v)
      return next
    })
  const click =
    onNodeClick || collapsible
      ? (s: string) => {
          const v = Number(s.slice(1))
          if (collapsible && tree.nodes[v].children.length) toggle(v)
          onNodeClick?.(v)
        }
      : undefined
  return (
    <Diagram
      spec={spec}
      ariaLabel={ariaLabel}
      height={height}
      onNodeClick={click}
      onNodeHover={onNodeHover && ((s) => onNodeHover(s === null ? null : Number(s.slice(1))))}
    />
  )
}
