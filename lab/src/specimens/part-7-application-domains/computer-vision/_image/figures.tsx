/**
 * "Classical vision: edges, corners and lines": a synthetic scene with known geometry (aifn-methods
 * `geometricScene`) run through core `aifn/signal/image`: Canny's stages scored against the true edges; Harris and
 * Shi–Tomasi corners against the true corners; Hough lines and circles against the drawn ones; and morphology and the
 * Laplacian pyramid on the same scene.
 */
import { useMemo } from 'react'
import { geometricScene } from 'aifn-methods/data/synthetic'
import {
  canny,
  closing,
  dilate,
  discElement,
  erode,
  harrisResponse,
  houghCirclePeaks,
  houghCircles,
  houghLinePeaks,
  houghLines,
  imagePeaks,
  laplacianPyramid,
  morphologicalGradient,
  opening,
  shiTomasiResponse,
  topHat,
} from 'aifn-compute/signal/image'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { Figure } from 'aifn-render/layout'
import { choice, row, slider, useFigureState } from 'aifn-render/state'
import { Curve, Plot, Plots, Points, Raster, Readout, formatNumber, useAxis } from 'aifn-render/viz'

const fmt = (v: number) => formatNumber(v)
const SIZE = 96

/** Row ticks of a plot whose y is −row (row 0 at the top): the labels read the row index. */
const rowLabel = (v: number) => formatNumber(-v + 0)

/** An image as a raster with row 0 at the top: x = column, y = −row. */
function raster(img: Tensor) {
  const [h, w] = img.shape
  const v = toFlat(img)
  return {
    x: Array.from({ length: w }, (_, j) => j),
    y: Array.from({ length: h }, (_, i) => i - (h - 1)),
    z: Array.from({ length: h }, (_, i) => v.slice((h - 1 - i) * w, (h - i) * w)),
  }
}

/** The part of the line x cos θ + y sin θ = ρ inside the image, in plot coordinates (column, −row). */
function lineSegment(angle: number, distance: number, w: number, h: number) {
  const c = Math.cos(angle)
  const s = Math.sin(angle)
  const pts: [number, number][] = []
  if (Math.abs(s) > Math.abs(c)) {
    for (const x of [0, w - 1]) pts.push([x, (distance - x * c) / s])
  } else {
    for (const y of [0, h - 1]) pts.push([(distance - y * s) / c, y])
  }
  return { x: pts.map((p) => p[0]), y: pts.map((p) => -p[1]) }
}

function useScene(noise: number, rotation: number) {
  return useMemo(() => geometricScene(stream('lab-vision-scene'), { size: SIZE, noise, rotation }), [noise, rotation])
}

const sceneRow = () =>
  row('1 · scene', {
    noise: slider(0, 0.3, 0.05, { label: 'noise σ', step: 0.01 }),
    rotation: slider(-0.8, 0.8, 0.3, { label: 'square rotation (rad)', step: 0.01 }),
  })

// ── Edges ────────────────────────────────────────────────────────────────────────────────────────────────────────────

const STAGES = {
  smoothed: 'smoothed image',
  magnitude: 'gradient magnitude',
  suppressed: 'after non-maximum suppression',
  hysteresis: 'strong and weak pixels',
  edges: 'edges after hysteresis',
} as const

export function CannySpecimen() {
  const state = useFigureState({
    scene: sceneRow(),
    canny: row('2 · Canny', {
      sigma: slider(0.5, 4, 1.4, { label: 'smoothing σ (px)', step: 0.1 }),
      low: slider(0.01, 0.6, 0.1, { label: 'low threshold (× max |∇|)', step: 0.01 }),
      high: slider(0.02, 0.9, 0.25, { label: 'high threshold (× max |∇|)', step: 0.01 }),
    }),
    show: row('3 · show', {
      stage: choice(
        Object.entries(STAGES).map(([value, label]) => ({ value, label })),
        'edges',
        { label: 'stage' },
      ),
    }),
  })
  const scene = useScene(state.scene.noise, state.scene.rotation)
  const { sigma, low, high } = state.canny
  const c = useMemo(() => {
    const first = canny(scene.image, { sigma, low: 0, high: 0 })
    const max = Math.max(...toFlat(first.magnitude))
    return canny(scene.image, { sigma, low: low * max, high: Math.max(low, high) * max })
  }, [scene, sigma, low, high])
  const stage = state.show.stage as keyof typeof STAGES
  const shown = useMemo(() => {
    if (stage === 'hysteresis') {
      const s = toFlat(c.strong)
      const wk = toFlat(c.weak)
      return raster(
        fromData(
          Float64Array.from(s, (v, i) => v + 0.5 * wk[i]),
          c.strong.shape,
        ),
      )
    }
    return raster(c[stage === 'edges' ? 'edges' : stage] as Tensor)
  }, [c, stage])
  const input = useMemo(() => raster(scene.image), [scene])
  const score = useMemo(() => {
    const e = toFlat(c.edges)
    const t = toFlat(scene.edges)
    const near = (a: number[], b: number[]) => {
      let hit = 0
      let total = 0
      for (let r = 0; r < SIZE; r++)
        for (let k = 0; k < SIZE; k++) {
          if (!a[r * SIZE + k]) continue
          total++
          let ok = false
          for (let dr = -1; dr <= 1 && !ok; dr++)
            for (let dk = -1; dk <= 1; dk++) if (b[(r + dr) * SIZE + k + dk]) ok = true
          if (ok) hit++
        }
      return total ? hit / total : NaN
    }
    return { precision: near(e, t), recall: near(t, e) }
  }, [c, scene])
  const truthPts = useMemo(() => {
    const t = toFlat(scene.edges)
    const x: number[] = []
    const y: number[] = []
    t.forEach((v, i) => {
      if (v) {
        x.push(i % SIZE)
        y.push(-Math.floor(i / SIZE))
      }
    })
    return { x, y }
  }, [scene])
  const xa = useAxis({ label: 'column', range: [-0.5, SIZE - 0.5], zoom: false })
  const ya = useAxis({ label: 'row', format: rowLabel, range: [-(SIZE - 0.5), 0.5], equal: xa, zoom: false })
  return (
    <Figure
      title="Canny edges, stage by stage"
      purpose="Canny's detector smooths, differentiates, thins the gradient to one-pixel ridges by non-maximum suppression, and links ridges by hysteresis: strong pixels are edges, weak ones only when connected to a strong one."
      defaultSize="XL"
      state={state}
      readouts={{
        'against the true edges': (
          <>
            <Readout label="precision (within 1 px)" value={`${Math.round(100 * score.precision)}%`} />
            <Readout label="recall (within 1 px)" value={`${Math.round(100 * score.recall)}%`} />
            <Readout label="thresholds |∇|" value={`${fmt(c.low)}, ${fmt(c.high)}`} />
          </>
        ),
      }}
      caption="Left: the input, a rotated square, a disc and two one-pixel lines with Gaussian noise. Right: the chosen stage of core canny; in the last stage the true edges of the clean scene are drawn faintly under the detected ones. Raise the noise and the weak threshold to see spurious edges; raise σ to suppress them at the cost of rounded corners and merged lines. Precision is the share of detected edge pixels within one pixel of a true edge; recall the share of true edge pixels within one pixel of a detected one."
    >
      <Plots cols={2}>
        <Plot x={xa} y={ya} title="input" legend={false}>
          <Raster {...input} scale="sequential" range={[0, 1]} colorBar={false} valueLabel="intensity" />
        </Plot>
        <Plot x={xa} y={ya} title={STAGES[stage]} legend={false}>
          <Raster {...shown} scale="sequential" colorBar={false} valueLabel={STAGES[stage]} />
          {stage === 'edges' && <Points name="true edges" x={truthPts.x} y={truthPts.y} muted size={2} />}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── Corners, lines and circles ───────────────────────────────────────────────────────────────────────────────────────

export function FeaturesSpecimen() {
  const state = useFigureState({
    scene: sceneRow(),
    corners: row('2 · corners', {
      method: choice(
        [
          { value: 'harris', label: 'Harris det − k tr²' },
          { value: 'shi-tomasi', label: 'Shi–Tomasi λ_min' },
        ],
        'harris',
        { label: 'response' },
      ),
      sigma: slider(0.5, 4, 1.5, { label: 'window σ', step: 0.1 }),
      relative: slider(0.01, 0.6, 0.1, { label: 'threshold (× max)', step: 0.01 }),
    }),
    hough: row('3 · Hough', {
      lineThreshold: slider(0.2, 0.95, 0.5, { label: 'line votes (× max)', step: 0.01 }),
      circleThreshold: slider(0.2, 1, 0.55, { label: 'circle score', step: 0.01 }),
    }),
  })
  const scene = useScene(state.scene.noise, state.scene.rotation)
  const { method, sigma, relative } = state.corners
  const { lineThreshold, circleThreshold } = state.hough
  const response = useMemo(
    () => (method === 'harris' ? harrisResponse(scene.image, { sigma }) : shiTomasiResponse(scene.image, { sigma })),
    [scene, method, sigma],
  )
  const found = useMemo(() => imagePeaks(response, { minDistance: 4, relative }), [response, relative])
  const edges = useMemo(() => canny(scene.image, { sigma: 1.4, quantile: true, low: 0.75, high: 0.9 }).edges, [scene])
  const acc = useMemo(() => houghLines(edges), [edges])
  const lines = useMemo(() => {
    const max = Math.max(...toFlat(acc.votes))
    return houghLinePeaks(acc, { threshold: lineThreshold * max, count: 6 })
  }, [acc, lineThreshold])
  const radii = useMemo(() => Array.from({ length: 11 }, (_, k) => 11 + k), [])
  const circles = useMemo(
    () => houghCirclePeaks(houghCircles(edges, radii), radii, { threshold: circleThreshold, count: 3 }),
    [edges, radii, circleThreshold],
  )
  const img = useMemo(() => raster(scene.image), [scene])
  const resp = useMemo(() => raster(response), [response])
  const accR = useMemo(() => {
    const [nr, nt] = acc.votes.shape
    const v = toFlat(acc.votes)
    const th = toFlat(acc.angles)
    const rho = toFlat(acc.distances)
    return { x: th, y: rho, z: Array.from({ length: nr }, (_, i) => v.slice(i * nt, (i + 1) * nt)) }
  }, [acc])
  // Corners matched to the truth within 3 px.
  const hits = scene.corners.filter((t) => found.some((p) => Math.hypot(p.row - t.row, p.col - t.col) <= 3)).length
  const ring = (row: number, col: number, r: number) => {
    const a = Array.from({ length: 73 }, (_, k) => (2 * Math.PI * k) / 72)
    return { x: a.map((t) => col + r * Math.cos(t)), y: a.map((t) => -(row + r * Math.sin(t))) }
  }
  const xa = useAxis({ label: 'column', range: [-0.5, SIZE - 0.5], zoom: false })
  const ya = useAxis({ label: 'row', format: rowLabel, range: [-(SIZE - 0.5), 0.5], equal: xa, zoom: false })
  const ta = useAxis({ label: 'θ (rad)', range: [-Math.PI / 2, Math.PI / 2] })
  const ra = useAxis({ label: 'ρ (px)', hold: 'initial' })
  return (
    <Figure
      title="Corners, lines and circles"
      purpose="Corners are where the structure tensor has two large eigenvalues; lines and circles are the peaks of a Hough accumulator in which every edge pixel votes for every shape through it."
      defaultSize="XL"
      state={state}
      readouts={{
        detected: (
          <>
            <Readout label="corners found / true corners hit" value={`${found.length} / ${hits} of 4`} />
            <Readout
              label="lines (θ, ρ)"
              value={lines.map((l) => `(${fmt(l.angle)}, ${fmt(l.distance)})`).join(' ') || 'none'}
            />
            <Readout
              label="true lines (θ, ρ)"
              value={scene.lines.map((l) => `(${fmt(l.angle)}, ${fmt(l.distance)})`).join(' ')}
            />
            <Readout
              label="circles (row, col, r)"
              value={circles.map((c) => `(${c.row}, ${c.col}, ${c.radius})`).join(' ') || 'none'}
            />
            <Readout
              label="true circle"
              value={scene.circles.map((c) => `(${fmt(c.row)}, ${fmt(c.col)}, ${fmt(c.radius)})`).join(' ')}
            />
          </>
        ),
      }}
      caption="Left: the corner response of the scene (Harris or Shi–Tomasi, Gaussian window σ) with its local maxima above the threshold (large dots, first colour) and the square's true corners (small ink squares; a found corner is an ink square inside a dot). Harris responds negatively along edges; the lines' crossings with the square also make corners. Middle: the line accumulator over (θ, ρ) built from Canny edges, with the detected peaks (large dots) and the true parameters (small ink squares). Right: the scene with the detected lines and circles drawn over it (circles searched for radii 11–21 px). The square's straight sides are lines too. Canny marks both sides of a one-pixel line, so the peaks for the drawn lines sit a pixel or two in ρ from the truth. Raise the noise to see the accumulator blur and false peaks pass the thresholds."
    >
      <Plots cols={3}>
        <Plot x={xa} y={ya} title="corner response" legend={false}>
          <Raster {...resp} scale="diverging" colorBar={false} valueLabel="response" />
          <Points
            name="true corners"
            x={scene.corners.map((p) => p.col)}
            y={scene.corners.map((p) => -p.row)}
            emphasis
            size={7}
            shape={1}
          />
          <Points name="detected corners" x={found.map((p) => p.col)} y={found.map((p) => -p.row)} slot={0} size={14} />
        </Plot>
        <Plot x={ta} y={ra} title="line accumulator" legend={false}>
          <Raster {...accR} scale="sequential" colorBar={false} valueLabel="votes" />
          <Points
            name="true lines"
            x={scene.lines.map((l) => l.angle)}
            y={scene.lines.map((l) => l.distance)}
            emphasis
            size={7}
            shape={1}
          />
          <Points name="peaks" x={lines.map((l) => l.angle)} y={lines.map((l) => l.distance)} slot={0} size={14} />
        </Plot>
        <Plot x={xa} y={ya} title="detected lines and circles" legend={false}>
          <Raster
            {...img}
            scale="sequential"
            range={[0, 1.2]}
            colorBar={false}
            valueLabel="intensity"
            fillOpacity={0.6}
          />
          {lines.map((l, k) => {
            const s = lineSegment(l.angle, l.distance, SIZE, SIZE)
            return <Curve key={`l${k}`} name="line" x={s.x} y={s.y} slot={0} />
          })}
          {circles.map((c, k) => {
            const r = ring(c.row, c.col, c.radius)
            return <Curve key={`c${k}`} name="circle" x={r.x} y={r.y} slot={1} />
          })}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── Morphology and pyramids ──────────────────────────────────────────────────────────────────────────────────────────

const OPS = {
  erode: 'erosion',
  dilate: 'dilation',
  opening: 'opening',
  closing: 'closing',
  gradient: 'morphological gradient',
  tophat: 'white top-hat',
} as const

export function MorphologySpecimen() {
  const state = useFigureState({
    scene: sceneRow(),
    morph: row('2 · morphology', {
      op: choice(
        Object.entries(OPS).map(([value, label]) => ({ value, label })),
        'opening',
        { label: 'operation' },
      ),
      radius: slider(1, 6, 2, { label: 'disc radius (px)', step: 1 }),
    }),
    pyramid: row('3 · Laplacian pyramid', { level: slider(0, 3, 1, { label: 'level', step: 1 }) }),
  })
  const scene = useScene(state.scene.noise, state.scene.rotation)
  const { op, radius } = state.morph
  const out = useMemo(() => {
    const e = discElement(radius)
    const f = { erode, dilate, opening, closing, gradient: morphologicalGradient, tophat: topHat }[
      op as keyof typeof OPS
    ]
    return f(scene.image, e)
  }, [scene, op, radius])
  const pyr = useMemo(() => laplacianPyramid(scene.image, { levels: 5 }), [scene])
  const level = Math.min(state.pyramid.level, pyr.length - 1)
  const lap = pyr[level]
  const [lh, lw] = lap.shape
  const xa = useAxis({ label: 'column', range: [-0.5, SIZE - 0.5], zoom: false })
  const ya = useAxis({ label: 'row', format: rowLabel, range: [-(SIZE - 0.5), 0.5], equal: xa, zoom: false })
  const xp = useAxis({ label: 'column (level pixels)', range: [-0.5, lw - 0.5], zoom: false, key: level })
  const yp = useAxis({ label: 'row', format: rowLabel, range: [-(lh - 0.5), 0.5], equal: xp, zoom: false, key: level })
  return (
    <Figure
      title="Morphology and the Laplacian pyramid"
      purpose="Morphology filters by shape: erosion and dilation take the minimum and maximum over a structuring element, and opening and closing combine them to remove bright or dark details smaller than it. A Laplacian pyramid splits the image into band-pass levels, each a Gaussian level minus the expansion of the next."
      defaultSize="L"
      state={state}
      readouts={{
        pyramid: (
          <>
            <Readout label="level size" value={`${lh} × ${lw}`} />
            <Readout label="levels" value={pyr.length} />
          </>
        ),
      }}
      caption="Left: the scene after the chosen operation with a disc of the chosen radius: opening removes the one-pixel lines (they are thinner than the disc) and keeps the square and disc; closing fills dark gaps; the gradient outlines the shapes; the top-hat keeps only the thin bright lines. Right: one level of the Laplacian pyramid (diverging scale): level 0 holds the finest detail (edges and noise), deeper levels coarser structure, and the last is the low-pass residual."
    >
      <Plots cols={2}>
        <Plot x={xa} y={ya} title={OPS[op as keyof typeof OPS]} legend={false}>
          <Raster {...raster(out)} scale="sequential" colorBar={false} valueLabel="value" />
        </Plot>
        <Plot
          x={xp}
          y={yp}
          title={level === pyr.length - 1 ? 'low-pass residual' : `band-pass level ${level}`}
          legend={false}
        >
          <Raster
            {...raster(lap)}
            scale={level === pyr.length - 1 ? 'sequential' : 'diverging'}
            colorBar={false}
            valueLabel="value"
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
