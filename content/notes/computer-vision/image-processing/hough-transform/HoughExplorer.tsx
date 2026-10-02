import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  ParamSwitch,
  Readout,
  formatNumber,
  type HeatmapOverlay,
  type PlotPointer,
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
  const [scene, setScene] = useState<Scene>('corridor')
  const [show, setShow] = useState<'edges' | 'image'>('edges')
  const [pixelNoise, setPixelNoise] = useState(0.04)
  const [threshold, setThreshold] = useState(0.2)
  const [positionNoise, setPositionNoise] = useState(1)
  const [clutter, setClutter] = useState(80)
  const [dTheta, setDTheta] = useState(1)
  const [dRho, setDRho] = useState(1.5)
  const [lines, setLines] = useState(DEFAULT_LINES.corridor)
  const [directional, setDirectional] = useState(false)
  const [logScale, setLogScale] = useState(true)
  const [seed, setSeed] = useState(1)
  const [added, setAdded] = useState<EdgePoint[]>([])
  const [hover, setHover] = useState<Hover>(null)

  const cloud = scene === 'points'
  const image = useMemo(
    () => (scene === 'points' ? null : sceneImage(scene, pixelNoise, seed)),
    [scene, pixelNoise, seed],
  )
  const detected = useMemo(
    () => (image ? edgePoints(image, threshold) : pointCloud(positionNoise, clutter, seed)),
    [image, threshold, positionNoise, clutter, seed],
  )
  const points = useMemo(() => [...detected, ...added], [detected, added])
  const shown = useMemo(() => (image && show === 'image' ? image : rasterise(detected)), [image, show, detected])

  const g = useMemo(() => grid(dTheta, dRho), [dTheta, dRho])
  const voteWindow = directional ? DIRECTION_WINDOW : null
  const { acc, votes } = useMemo(() => accumulate(g, points, voteWindow), [g, points, voteWindow])
  const z = useMemo(() => (logScale ? acc.map((row) => row.map((v) => Math.log1p(v))) : acc), [acc, logScale])
  const peaks = useMemo(() => findPeaks(g, acc, lines), [g, acc, lines])

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

  const imageOverlay = useMemo((): HeatmapOverlay[] => {
    const found = segmentsPath(peaks)
    const probe = hoveredCell ? segmentsPath([{ theta: g.thetas[hoveredCell.j], rho: g.rhos[hoveredCell.k] }]) : null
    const lit = hoveredCell ? voters : hoveredPoint ? [hoveredPoint] : []
    return [
      { name: 'detected lines', type: 'line', x: found.x, y: found.y, emphasis: true },
      { name: 'line of hovered cell', type: 'line', x: probe?.x ?? [], y: probe?.y ?? [], slot: 1 },
      { name: 'hovered point or voters', type: 'scatter', x: lit.map((p) => p.x), y: lit.map((p) => p.y), slot: 1 },
      { name: 'added points', type: 'scatter', x: added.map((p) => p.x), y: added.map((p) => p.y), slot: 2 },
    ]
  }, [peaks, hoveredCell, hoveredPoint, voters, added, g])

  const houghOverlay = useMemo((): HeatmapOverlay[] => {
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

  const changeScene = (s: Scene) => {
    setScene(s)
    setLines(DEFAULT_LINES[s])
    setAdded([])
    setHover(null)
  }

  const marker: [number, number] | undefined = hoveredCell
    ? [g.thetas[hoveredCell.j], g.rhos[hoveredCell.k]]
    : undefined
  const hoverText = hoveredCell
    ? `θ = ${formatNumber(g.thetas[hoveredCell.j])}°, ρ = ${formatNumber(g.rhos[hoveredCell.k])}: ${acc[hoveredCell.k][hoveredCell.j]} votes`
    : hoveredPoint
      ? `(${formatNumber(hoveredPoint.x)}, ${formatNumber(hoveredPoint.y)})${hoveredPoint.snapped ? ', an edge point' : ''}`
      : 'none'

  return (
    <Interactive
      title="Hough voting on edge points"
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
        <>
          <ParamChoice label="image" value={scene} onChange={changeScene} options={SCENES} />
          {cloud ? (
            <>
              <ParamSlider
                label="position noise σ (pixels)"
                value={positionNoise}
                onChange={setPositionNoise}
                min={0}
                max={4}
                step={0.25}
              />
              <ParamSlider label="clutter points" value={clutter} onChange={setClutter} min={0} max={300} step={10} />
            </>
          ) : (
            <>
              <ParamChoice
                label="left panel"
                value={show}
                onChange={setShow}
                options={[
                  { value: 'edges', label: 'edge points' },
                  { value: 'image', label: 'image' },
                ]}
              />
              <ParamSlider
                label="pixel noise σ"
                value={pixelNoise}
                onChange={setPixelNoise}
                min={0}
                max={0.3}
                step={0.01}
                format={(v) => v.toFixed(2)}
              />
              <ParamSlider
                label="edge threshold (share of the largest gradient)"
                value={threshold}
                onChange={setThreshold}
                min={0.05}
                max={0.8}
                step={0.05}
                format={(v) => v.toFixed(2)}
              />
            </>
          )}
          <ParamSlider label="θ bin width (degrees)" value={dTheta} onChange={setDTheta} min={1} max={6} step={1} />
          <ParamSlider label="ρ bin width (pixels)" value={dRho} onChange={setDRho} min={1} max={6} step={0.5} />
          <ParamSlider label="lines to keep" value={lines} onChange={setLines} min={1} max={20} step={1} withArrows />
          <ParamSwitch label="vote along the gradient direction only" checked={directional} onChange={setDirectional} />
          <ParamSwitch label="log colour scale" checked={logScale} onChange={setLogScale} />
          <div className="flex gap-2 self-end">
            <ParamButton onClick={() => setSeed((s) => s + 1)}>new noise</ParamButton>
            <ParamButton onClick={() => setAdded([])} disabled={added.length === 0}>
              clear added points
            </ParamButton>
          </div>
        </>
      }
      readout={
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
        <Heatmap
          x={AXIS}
          y={AXIS}
          z={shown}
          range={SHOWN_RANGE}
          fillOpacity={0.85}
          xLabel="x (pixels)"
          yLabel="y (pixels)"
          valueLabel={image && show === 'image' ? 'intensity' : 'edge'}
          overlay={imageOverlay}
          onPointer={onImagePointer}
          equalAspect
          colorBar={false}
          ariaLabel="Image plane with edge points and detected lines"
        />
        <Heatmap
          x={g.thetas}
          y={g.rhos}
          z={z}
          xLabel="θ (degrees)"
          yLabel="ρ (pixels)"
          valueLabel={logScale ? 'log(1 + votes)' : 'votes'}
          overlay={houghOverlay}
          marker={marker}
          onPointer={onHoughPointer}
          height={340}
          ariaLabel="Hough accumulator over theta and rho"
        />
      </div>
    </Interactive>
  )
}

const SHOWN_RANGE: [number, number] = [0, 1]
