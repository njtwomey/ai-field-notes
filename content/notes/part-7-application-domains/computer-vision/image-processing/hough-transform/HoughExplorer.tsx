import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Button,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  type PlotPointer,
  Points,
  Raster,
  Readout,
  seriesLayers,
  type SeriesSpec,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  AXIS,
  accumulate,
  clipLine,
  edgePoints,
  findPeaks,
  grid,
  pointCloud,
  rasterise,
  rhoBin,
  sceneImage,
  votingColumns,
  type EdgePoint,
  type Scene,
} from './hough'

const SCENES: { value: Scene; label: string }[] = [
  { value: 'grid', label: 'chessboard' },
  { value: 'corridor', label: 'corridor' },
  { value: 'house', label: 'house' },
  { value: 'circles', label: 'circles + line' },
  { value: 'points', label: 'point cloud' },
]

/** Lines worth keeping in each scene: its true line count, so extra peaks are a choice the reader makes. */
const DEFAULT_LINES: Record<Scene, number> = { grid: 14, corridor: 6, house: 8, circles: 1, points: 3 }

/** A point votes only within this many degrees of its gradient direction when direction voting is on. */
const DIRECTION_WINDOW = 5
/** Hovering within this many pixels of an edge point snaps to it. */
const SNAP = 2.5

type Hover = { kind: 'point'; x: number; y: number; snapped: boolean } | { kind: 'cell'; j: number; k: number } | null

/** Line segments as one overlay path, broken between segments by NaN points. */
function segmentsPath(lines: { theta: number; rho: number }[]): { x: number[]; y: number[] } {
  const x: number[] = []
  const y: number[] = []
  for (const l of lines) {
    const ends = clipLine(l.theta, l.rho)
    if (!ends) continue
    x.push(ends[0][0], ends[1][0], NaN)
    y.push(ends[0][1], ends[1][1], NaN)
  }
  return { x, y }
}

/** Linked views of the Hough transform: an image with its edge points, and the (θ, ρ) accumulator they vote into. */
export function HoughExplorer() {
  const state = useFigureState({
    scene: choice<Scene>(SCENES, 'corridor', { label: 'image' }),
    show: choice<'edges' | 'image'>(
      [
        { value: 'edges', label: 'edge points' },
        { value: 'image', label: 'image' },
      ],
      'edges',
      { label: 'left panel', when: (v) => v.scene !== 'points' },
    ),
    pixelNoise: slider(0, 0.3, 0.04, {
      step: 0.01,
      label: 'pixel noise σ',
      format: (v) => v.toFixed(2),
      when: (v) => v.scene !== 'points',
    }),
    threshold: slider(0.05, 0.8, 0.2, {
      step: 0.05,
      label: 'edge threshold (share of the largest gradient)',
      format: (v) => v.toFixed(2),
      when: (v) => v.scene !== 'points',
    }),
    positionNoise: slider(0, 4, 1, {
      step: 0.25,
      label: 'position noise σ (pixels)',
      when: (v) => v.scene === 'points',
    }),
    clutter: int(80, {
      min: 0,
      max: 300,
      suggestions: [0, 40, 80, 150, 300],
      label: 'clutter points',
      when: (v) => v.scene === 'points',
    }),
    dTheta: slider(1, 6, 1, { step: 1, label: 'θ bin width (degrees)' }),
    dRho: float(1.5, { min: 1, max: 6, step: 0.5, label: 'ρ bin width (pixels)' }),
    lines: int(DEFAULT_LINES.corridor, { min: 1, max: 20, label: 'lines to keep' }),
    directional: setting(false, 'vote along the gradient direction only'),
    logScale: setting(true, 'log colour scale'),
    seed: int(1, { ge: 0, label: 'seed' }),
  })
  const { scene, show, pixelNoise, threshold, positionNoise, clutter } = state
  const [added, setAdded] = useState<EdgePoint[]>([])
  const [hover, setHover] = useState<Hover>(null)

  // A new scene starts from its own line count, without the points added to the last one.
  const lastScene = useRef(scene)
  useEffect(() => {
    if (lastScene.current === scene) return
    lastScene.current = scene
    state.set('lines', DEFAULT_LINES[scene])
    setAdded([])
    setHover(null)
  }, [scene, state])

  const image = useMemo(
    () => (scene === 'points' ? null : sceneImage(scene, pixelNoise, state.seed)),
    [scene, pixelNoise, state.seed],
  )
  const detected = useMemo(
    () => (image ? edgePoints(image, threshold) : pointCloud(positionNoise, clutter, state.seed)),
    [image, threshold, positionNoise, clutter, state.seed],
  )
  const points = useMemo(() => [...detected, ...added], [detected, added])
  const shown = useMemo(() => (image && show === 'image' ? image : rasterise(detected)), [image, show, detected])

  const g = useMemo(() => grid(state.dTheta, state.dRho), [state.dTheta, state.dRho])
  const voteWindow = state.directional ? DIRECTION_WINDOW : null
  const { acc, votes } = useMemo(() => accumulate(g, points, voteWindow), [g, points, voteWindow])
  const z = useMemo(
    () => (state.logScale ? acc.map((row) => row.map((v) => Math.log1p(v))) : acc),
    [acc, state.logScale],
  )
  const peaks = useMemo(() => findPeaks(g, acc, state.lines), [g, acc, state.lines])

  // Everything the hover changes is an overlay or the marker, which the charts patch without redrawing their grids.
  const hoveredCell = hover?.kind === 'cell' ? hover : null
  const hoveredPoint = hover?.kind === 'point' ? hover : null
  const voters = useMemo(() => {
    if (!hoveredCell) return []
    const { j, k } = hoveredCell
    return points.filter(
      (p) =>
        rhoBin(g, p.x * g.cos[j] + p.y * g.sin[j]) === k &&
        (voteWindow === null || votingColumns(g, p, voteWindow).includes(j)),
    )
  }, [hoveredCell, points, g, voteWindow])

  const imageOverlay = useMemo(() => {
    const found = segmentsPath(peaks)
    const probe = hoveredCell ? segmentsPath([{ theta: g.thetas[hoveredCell.j], rho: g.rhos[hoveredCell.k] }]) : null
    const lit = hoveredCell ? voters : hoveredPoint ? [hoveredPoint] : []
    return [
      { name: 'detected lines', x: found.x, y: found.y, emphasis: true },
      { name: 'line of hovered cell', x: probe?.x ?? [], y: probe?.y ?? [], slot: 1 },
      { name: 'hovered point or voters', x: lit.map((p) => p.x), y: lit.map((p) => p.y), slot: 1 },
      { name: 'added points', x: added.map((p) => p.x), y: added.map((p) => p.y), slot: 2 },
    ] as const
  }, [peaks, hoveredCell, hoveredPoint, voters, added, g])

  const houghOverlay = useMemo((): SeriesSpec[] => {
    const curve = hoveredPoint
      ? {
          x: g.thetas,
          y: g.thetas.map((_, j) => hoveredPoint.x * g.cos[j] + hoveredPoint.y * g.sin[j]),
        }
      : { x: [], y: [] }
    return [
      { name: 'peaks', type: 'scatter', x: peaks.map((p) => p.theta), y: peaks.map((p) => p.rho), emphasis: true },
      { name: 'sinusoid of hovered point', type: 'line', ...curve, slot: 1 },
    ]
  }, [peaks, hoveredPoint, g])

  const onImagePointer = (e: PlotPointer) => {
    if (e.type === 'leave') return setHover(null)
    const [x, y] = e.point
    if (e.type === 'click') {
      setAdded((a) => [...a, { x, y }])
      return
    }
    let best: EdgePoint | null = null
    let d = SNAP
    for (const p of points) {
      const dd = Math.hypot(p.x - x, p.y - y)
      if (dd < d) {
        d = dd
        best = p
      }
    }
    setHover(best ? { kind: 'point', x: best.x, y: best.y, snapped: true } : { kind: 'point', x, y, snapped: false })
  }

  const onHoughPointer = (e: PlotPointer) => {
    if (e.type === 'leave') return setHover(null)
    const [theta, rho] = e.point
    const nT = g.thetas.length
    const j = ((Math.round(theta / g.dTheta) % nT) + nT) % nT
    const k = rhoBin(g, rho)
    setHover((h) => (h?.kind === 'cell' && h.j === j && h.k === k ? h : { kind: 'cell', j, k }))
  }

  const marker: [number, number] | undefined = hoveredCell
    ? [g.thetas[hoveredCell.j], g.rhos[hoveredCell.k]]
    : undefined
  const hoverText = hoveredCell
    ? `θ = ${formatNumber(g.thetas[hoveredCell.j])}°, ρ = ${formatNumber(g.rhos[hoveredCell.k])}: ${acc[hoveredCell.k][hoveredCell.j]} votes`
    : hoveredPoint
      ? `(${formatNumber(hoveredPoint.x)}, ${formatNumber(hoveredPoint.y)})${hoveredPoint.snapped ? ', an edge point' : ''}`
      : 'none'

  const xAxis = useAxis({ label: 'x (pixels)' })
  const yAxis = useAxis({ label: 'y (pixels)', equal: xAxis })
  const xAxis2 = useAxis({ label: 'θ (degrees)' })
  const yAxis2 = useAxis({ label: 'ρ (pixels)' })
  return (
    <Figure
      title="Hough voting on edge points"
      state={state}
      caption={
        <>
          Left: an image or its edge points, with the origin at the centre and y up, and the detected lines drawn over
          it. Right: the accumulator. Each edge point adds one vote along its sinusoid ρ = x cos θ + y sin θ, and
          diamonds mark the peaks. Hover over the image to see a point&apos;s sinusoid. Hover over the accumulator to
          see that cell&apos;s line and the points that voted for it. Click the image to add a point. Every edge of the
          corridor passes through one vanishing point, so its peaks lie on that point&apos;s sinusoid. The chessboard
          gives two columns of peaks, at 15° and 105°. With circles, only the straight edge makes a sharp peak.
        </>
      }
      controls={
        <div className="flex gap-2 self-end">
          <Button variant="outline" size="sm" onClick={() => setAdded([])} disabled={added.length === 0}>
            clear added points
          </Button>
        </div>
      }
      readouts={
        <>
          <Readout label="edge points" value={String(points.length)} />
          <Readout label="accumulator" value={`${g.thetas.length} × ${g.rhos.length}`} />
          <Readout label="votes cast" value={String(votes)} />
          <Readout label="top peak" value={peaks.length ? `${peaks[0].votes} votes` : 'none'} />
          <Readout label="hovered" value={hoverText} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot
          x={xAxis}
          y={yAxis}
          ariaLabel={'Image plane with edge points and detected lines'}
          onPointer={onImagePointer}
        >
          <Raster
            x={AXIS}
            y={AXIS}
            z={shown}
            range={SHOWN_RANGE}
            fillOpacity={0.85}
            valueLabel={image && show === 'image' ? 'intensity' : 'edge'}
            colorBar={false}
          />
          <Curve {...imageOverlay[0]} live />
          <Curve {...imageOverlay[1]} live />
          <Points {...imageOverlay[2]} live />
          <Points {...imageOverlay[3]} live />
        </Plot>
        <Plot
          x={xAxis2}
          y={yAxis2}
          height={340}
          ariaLabel={'Hough accumulator over theta and rho'}
          onPointer={onHoughPointer}
        >
          <Raster x={g.thetas} y={g.rhos} z={z} valueLabel={state.logScale ? 'log(1 + votes)' : 'votes'} />
          {seriesLayers(houghOverlay, { live: true })}
          {marker && <Points x={[marker[0]]} y={[marker[1]]} emphasis live />}
        </Plot>
      </div>
    </Figure>
  )
}

const SHOWN_RANGE: [number, number] = [0, 1]
