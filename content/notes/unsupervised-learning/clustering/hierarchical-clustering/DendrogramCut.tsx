import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
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
  const [linkage, setLinkage] = useState<Linkage>('single')
  const merges = useMemo(() => agglomerate(points, linkage), [points, linkage])
  const top = merges[merges.length - 1].height
  // The cut is a fraction of the tallest merge, so it stays meaningful when the linkage rescales the tree.
  const cutAt = useParam(0.55, { min: 0, max: 1, step: 0.005 })
  const height = cutAt.value * top
  const labels = useMemo(() => cut(points.length, merges, height), [points, merges, height])
  const tree = useMemo(() => dendrogram(points.length, merges), [points, merges])
  const k = new Set(labels).size
  const handles: Handle[] = [{ kind: 'y', at: height, label: 'cut', onDrag: (y) => cutAt.set(y / top) }]
  const n = points.length

  return (
    <Interactive
      title="Cutting a dendrogram"
      caption="Each horizontal bar joins two clusters at the height of their linkage distance. Drag the cut line up or down, or use the slider: every bar below the line is applied, and the clusters are what remains. Single linkage chains the two lower blobs together along the bridge of points; complete, average and Ward linkage separate them."
      controls={
        <>
          <ParamChoice label="linkage" value={linkage} onChange={setLinkage} options={LINKAGES} />
          <ParamSlider label="cut height (fraction of the tallest merge)" param={cutAt} />
        </>
      }
      readout={
        <>
          <Readout label="cut height" value={formatNumber(height)} />
          <Readout label="clusters" value={k} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          height={340}
          xLabel="leaf order"
          yLabel="merge height"
          xRange={[-1, n]}
          yRange={[0, top * 1.05]}
          handles={handles}
          series={[{ name: 'dendrogram', type: 'line', x: tree.x, y: tree.y, emphasis: true }]}
        />
        <XYChart
          height={340}
          xLabel="x₁"
          yLabel="x₂"
          xRange={[-1.5, 5.5]}
          yRange={[-1.5, 5]}
          series={clusterSeries(points, labels, 'other points')}
        />
      </div>
    </Interactive>
  )
}
