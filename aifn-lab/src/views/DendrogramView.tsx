import { useMemo } from 'react'
import { cutTree, mergeTree } from 'aifn/cluster'
import { toFlat, toRows, type Tensor } from 'aifn/tensor'
import { Slider } from '@lab/controls'
import { Figure } from '@lab/layout'
import { Readout, XYChart } from '@lab/viz'
import type { FrameProps } from './frame'
import { formatValue } from './format'
import { TreeView } from './TreeView'

export type DendrogramViewProps = FrameProps & {
  /** A linkage matrix [n − 1, 4] in SciPy's format (from `aifn/cluster`'s `linkage` or an `agglomerative` model). */
  merges: Tensor
  /** Where the tree is cut (a merge height); merges above it are undone. */
  cut: number
  onCut: (height: number) => void
  /** Point labels (default the point ids). */
  leafLabels?: readonly string[]
  /** The clustered points [n, 2], drawn beside the tree in their clusters' colours. */
  points?: Tensor
}

/**
 * A dendrogram of agglomerative merges, drawn by `TreeView` with nodes at their merge heights, and a cut: every
 * cluster below the cut in its own colour, the merges above it dimmed. The cut height has a slider (with the number
 * of clusters it gives as a readout); with `points`, the points are drawn beside the tree in the same colours.
 */
export function DendrogramView({
  merges,
  cut,
  onCut,
  leafLabels,
  points,
  title,
  controls,
  readouts,
  ...frame
}: DendrogramViewProps) {
  const tree = useMemo(() => mergeTree(merges), [merges])
  const n = merges.shape[0] + 1
  const heights = useMemo(() => toRows(merges).map((r) => r[2]), [merges])
  const top = heights.length ? heights[heights.length - 1] : 1
  const labels = useMemo(() => toFlat(cutTree(merges, { height: cut })), [merges, cut])
  // Each node below the cut takes the cluster of its leaves; nodes above it are dimmed.
  const tone = useMemo(() => {
    const out = new Array<number | 'neutral'>(tree.nodes.length)
    const firstLeaf = (v: number): number => (tree.nodes[v].children.length ? firstLeaf(tree.nodes[v].children[0]) : v)
    for (const node of tree.nodes) out[node.id] = (node.height ?? 0) <= cut ? labels[firstLeaf(node.id)] % 8 : 'neutral'
    return out
  }, [tree, labels, cut])
  const clusters = new Set(labels).size
  const scatter = useMemo(() => {
    if (!points) return null
    const rows = toRows(points)
    return { x: rows.map((r) => r[0]), y: rows.map((r) => r[1]) }
  }, [points])
  return (
    <Figure
      title={title ?? 'Dendrogram'}
      defaultSize="L"
      {...frame}
      controls={
        <>
          {controls}
          <Slider label="cut height" value={cut} min={0} max={top * 1.05} onChange={onCut} />
        </>
      }
      readouts={
        <>
          <Readout label="clusters at the cut" value={clusters} />
          <Readout label="points" value={n} />
          <Readout label="top merge" value={formatValue(top)} />
          {readouts}
        </>
      }
    >
      <div className={scatter ? 'grid grid-cols-[3fr_2fr] gap-2' : undefined}>
        <TreeView
          tree={tree}
          heightAxis
          ariaLabel="A dendrogram"
          nodeLabels={(v) => (v < n ? (leafLabels?.[v] ?? (n <= 40 ? String(v) : '')) : '')}
          nodeTone={(v) => tone[v]}
          nodeState={(v) => ((tree.nodes[v].height ?? 0) <= cut ? 'done' : 'idle')}
          edgeTone={(v) => tone[v]}
          siblingGap={0.2}
        />
        {scatter && (
          <XYChart
            aspect="equal"
            xLabel="x₀"
            yLabel="x₁"
            series={[
              {
                name: 'points',
                type: 'scatter',
                x: scatter.x,
                y: scatter.y,
                group: labels.map((l) => l % 8),
                groupNames: Array.from({ length: 8 }, (_, k) => `cluster ${k}`),
              },
            ]}
            legend={false}
          />
        )}
      </div>
    </Figure>
  )
}
