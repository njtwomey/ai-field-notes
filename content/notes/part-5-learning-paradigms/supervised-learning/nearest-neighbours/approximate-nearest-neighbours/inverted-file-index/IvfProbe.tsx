import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type HeatmapOverlay,
  type XYSeries,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { blobs, dist2, kmeans2d, nearest, rankBy, type P } from '../_shared/geometry'

const N = 1200
const NLIST = 24
const N_EVAL = 400
const GRID = linspace(-5, 5, 81)
const RANGE: [number, number] = [0, 1]
const RECALL_RANGE: [number, number] = [0, 1]

/** An inverted file in 2-D: k-means cells, a query and the nprobe cells searched for it. */
export function IvfProbe() {
  const nprobe = useParam(2, { min: 1, max: NLIST, step: 1 })
  const [query, setQuery] = useState<P>([0.4, 0.9])

  const world = useMemo(() => {
    const pts = blobs(N, 9, 0.75, 7)
    const { centres, assign } = kmeans2d(pts, NLIST, 3)
    // Each grid cell's coarse cell, for shading the searched region.
    const gridCell = GRID.map((y) => GRID.map((x) => nearest(centres, [x, y])))
    // Held-out queries from the same distribution, for the recall curve: the rank of each query's true
    // nearest neighbour's cell among the query's closest centroids decides the smallest nprobe that finds it.
    const g = rng(99)
    const queries = blobs(N_EVAL, 9, 0.75, 7).map((p): P => [p[0] + 0.3 * g.normal(), p[1] + 0.3 * g.normal()])
    const needed = queries.map((q) => rankBy(centres, q).indexOf(assign[nearest(pts, q)]) + 1)
    const recall = Array.from({ length: NLIST }, (_, w) => needed.filter((r) => r <= w + 1).length / N_EVAL)
    return { pts, centres, assign, gridCell, recall }
  }, [])

  const probed = useMemo(
    () => new Set(rankBy(world.centres, query).slice(0, nprobe.value)),
    [world, query, nprobe.value],
  )
  const z = useMemo(() => world.gridCell.map((row) => row.map((c) => (probed.has(c) ? 0.55 : 0))), [world, probed])

  const truth = nearest(world.pts, query)
  const scanned = world.pts.flatMap((_, i) => (probed.has(world.assign[i]) ? [i] : []))
  const found = scanned.length
    ? scanned.reduce((b, i) => (dist2(world.pts[i], query) < dist2(world.pts[b], query) ? i : b))
    : -1
  const hit = found === truth

  const overlay = useMemo(
    (): HeatmapOverlay[] => [
      {
        name: 'points',
        type: 'scatter',
        x: world.pts.map((p) => p[0]),
        y: world.pts.map((p) => p[1]),
        slot: 2,
      },
      {
        name: 'centroids',
        type: 'scatter',
        x: world.centres.map((c) => c[0]),
        y: world.centres.map((c) => c[1]),
        emphasis: true,
      },
      {
        name: 'true nearest neighbour',
        type: 'scatter',
        x: [world.pts[truth][0]],
        y: [world.pts[truth][1]],
        slot: 3,
      },
    ],
    [world, truth],
  )
  const handles: Handle[] = [{ kind: 'point', at: query, label: 'query', onDrag: (p) => setQuery(p) }]

  const curve: XYSeries[] = useMemo(
    () => [
      {
        name: '1-recall@1',
        type: 'line',
        x: world.recall.map((_, i) => i + 1),
        y: world.recall,
        slot: 0,
      },
    ],
    [world],
  )
  const probeHandle: Handle[] = [
    { kind: 'x', at: nprobe.value, label: 'nprobe', onDrag: (x) => nprobe.set(Math.round(x)) },
  ]

  return (
    <Interactive
      title="Inverted file: which cells a query searches"
      caption={`${N} points are clustered by k-means into ${NLIST} cells (diamonds are the centroids). A query compares itself with the ${NLIST} centroids, then scans only the points in its nprobe nearest cells, shaded. Drag the query towards a cell boundary: its true nearest neighbour (the highlighted point) can sit in a neighbouring cell, and nprobe = 1 misses it. Right: the fraction of ${N_EVAL} random queries whose true nearest neighbour lies in a searched cell. Drag the nprobe line or use the slider.`}
      controls={<ParamSlider label="cells searched, nprobe" param={nprobe} format={(v) => String(v)} withArrows />}
      readout={
        <>
          <Readout
            label="points scanned"
            value={`${scanned.length} of ${N} (${formatNumber((100 * scanned.length) / N)}%)`}
          />
          <Readout label="distance computations" value={String(scanned.length + NLIST)} />
          <Readout label="true nearest neighbour found" value={hit ? 'yes' : 'no'} />
          <Readout label="1-recall@1 at this nprobe" value={formatNumber(world.recall[nprobe.value - 1])} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Heatmap
          x={GRID}
          y={GRID}
          z={z}
          range={RANGE}
          scale="sequential"
          xLabel="x₁"
          yLabel="x₂"
          valueLabel="searched"
          overlay={overlay}
          handles={handles}
          height={380}
        />
        <XYChart
          series={curve}
          xLabel="nprobe"
          yLabel="1-recall@1"
          yRange={RECALL_RANGE}
          handles={probeHandle}
          height={380}
        />
      </div>
    </Interactive>
  )
}
