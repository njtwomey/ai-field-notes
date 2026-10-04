import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Readout,
  seriesLayers,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { blobs, type Point } from '../../_shared/datasets'
import { clusterSeries } from '../../_shared/groups'
import { agglomerate, cut, dendrogram, type Linkage } from './agglomerate'

const LINKAGES = [
  { value: 'single', label: 'single' },
  { value: 'complete', label: 'complete' },
  { value: 'average', label: 'average' },
  { value: 'ward', label: 'Ward' },
] as const satisfies readonly { value: Linkage; label: string }[]

/** Three blobs, two of them joined by a thin bridge of points that single linkage chains along. */
function data(): Point[] {
  const { points } = blobs(
    [
      [0, 0],
      [4, 0.5],
      [2, 3.5],
    ],
    [9, 9, 9],
    [0.45, 0.45, 0.45],
    4,
  )
  const bridge: Point[] = [0.9, 1.6, 2.3, 3.0].map((x, i) => [x, 0.15 + 0.1 * (i % 2)])
  return [...points, ...bridge]
}

export function DendrogramCut() {
  const points = useMemo(() => data(), [])
  const state = useFigureState({
    linkage: choice<Linkage>(LINKAGES, 'single', { label: 'linkage' }),
    cutAt: slider(0, 1, 0.55, { step: 0.005, label: 'cut height (fraction of the tallest merge)' }),
  })
  const merges = useMemo(() => agglomerate(points, state.linkage), [points, state.linkage])
  const top = merges[merges.length - 1].height
  // The cut is a fraction of the tallest merge, so it stays meaningful when the linkage rescales the tree.
  const height = state.cutAt * top
  const labels = useMemo(() => cut(points.length, merges, height), [points, merges, height])
  const tree = useMemo(() => dendrogram(points.length, merges), [points, merges])
  const k = new Set(labels).size
  const n = points.length

  const xAxis = useAxis({ label: 'leaf order', range: [-1, n] })
  const yAxis = useAxis({ label: 'merge height', range: [0, top * 1.05] })
  const xAxis2 = useAxis({ label: 'x₁', range: [-1.5, 5.5] })
  const yAxis2 = useAxis({ label: 'x₂', range: [-1.5, 5] })
  return (
    <Figure
      title="Cutting a dendrogram"
      state={state}
      caption="Each horizontal bar joins two clusters at the height of their linkage distance. Drag the cut line up or down, or use the slider: every bar below the line is applied, and the clusters are what remains. Single linkage chains the two lower blobs together along the bridge of points; complete, average and Ward linkage separate them."

      readouts={
        <>
          <Readout label="cut height" value={formatNumber(height)} />
          <Readout label="clusters" value={k} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={340}>
          <Curve name="dendrogram" x={tree.x} y={tree.y} emphasis />
          <Handle kind="y" at={height} label="cut" onDrag={(y) => state.set('cutAt', y / top)} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          {seriesLayers(clusterSeries(points, labels, 'other points'))}
        </Plot>
      </div>
    </Figure>
  )
}
