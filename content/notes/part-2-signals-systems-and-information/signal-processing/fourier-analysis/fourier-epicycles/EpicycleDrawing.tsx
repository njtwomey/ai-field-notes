import { useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import {
  Annotation,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  FrameContext,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Readout,
  setting,
  slider,
  useAxis,
  useElementSize,
  useFigureState,
  Vectors,
  when,
  type Vector,
} from 'aifn-render'
import { fft, fftfreq, ifft } from 'aifn-compute/foundation/fourier'
import { complex, cumsum, expj, mul, tensor, toComplexFlat, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { shuffle, stream } from 'aifn-compute/foundation/random'
import { freehandPath, glyphPath, loadFont, type Pt } from './paths'
import { SHAPES, shapePath, type ShapeGroup } from './shapes'

type Path = { x: number[]; y: number[] }

/** Samples per closed curve: the DFT length, so there are N − 1 rotating terms. */
const N = 4096
/** Default time steps per cycle, independent of N, so a cycle plays in about 8.5 s at 60 frames per second. */
const STEPS = 512
/** The traced curve is evaluated on a grid four times finer than the frames, for a smooth line. */
const FINE = 4 * N
const MAX_CIRCLES = 80
const MAX_ARROWS = 40
const RANGE: [number, number] = [-1.25, 1.25]
/** Circles and arcs in the zoomed view: points per circle, and the smallest radius drawn, as a share of the window. */
const ARC_POINTS = 48
const MIN_ZOOM_RADIUS = 1 / 200
/** Points on the stretch of traced curve drawn in the zoomed view, evaluated exactly rather than read off the grid. */
const ZOOM_TRACE_POINTS = 200

type Source = 'draw' | 'type' | ShapeGroup
const SOURCES: { value: Source; label: string }[] = [
  { value: 'type', label: 'a typed character' },
  { value: 'icons', label: 'an icon' },
  { value: 'drawings', label: 'a drawing' },
  { value: 'curves', label: 'a curve' },
  { value: 'draw', label: 'draw your own' },
]

const CHARS = [
  ...'&@?§¶ßæœøþðπΣΩλξζβγδεφψω∂∞∫√∮ℵ♪♣♠♥',
  ...'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
  ...'abcdefghijklmnopqrstuvwxyz',
  ...'0123456789',
]

/** Fonts the site loads anyway (KaTeX for the maths, Geist for the interface), and the browser's monospace. */
const FONTS = [
  { value: 'cm-italic', label: 'Computer Modern italic', family: 'KaTeX_Main', style: 'italic' },
  { value: 'cm', label: 'Computer Modern', family: 'KaTeX_Main', style: '' },
  { value: 'cm-bold', label: 'Computer Modern bold', family: 'KaTeX_Main', style: 'bold' },
  { value: 'fraktur', label: 'Fraktur', family: 'KaTeX_Fraktur', style: '' },
  { value: 'script', label: 'Script (capitals)', family: 'KaTeX_Script', style: '' },
  { value: 'geist', label: 'Geist', family: '"Geist Variable"', style: '' },
  { value: 'geist-black', label: 'Geist black', family: '"Geist Variable"', style: '900' },
  { value: 'mono', label: 'monospace', family: 'monospace', style: '' },
] as const
type FontKey = (typeof FONTS)[number]['value']

const ORDERS = [
  { value: 'magnitude', label: 'largest first' },
  { value: 'speed', label: 'slowest first: 0, 1, −1, 2, −2, …' },
  { value: 'frequency', label: 'by frequency, most negative first' },
  { value: 'random', label: 'random' },
] as const
type Order = (typeof ORDERS)[number]['value']

const optionsOf = (group: ShapeGroup) =>
  SHAPES.filter((s) => s.group === group).map((s) => ({ value: s.id, label: s.label }))
const toPath = (p: readonly Pt[]): Path => ({ x: p.map((q) => q[0]), y: p.map((q) => q[1]) })

type Spectrum = {
  /** c_m in DFT order (m = 0, …, N − 1), split into real and imaginary parts. */
  re: number[]
  im: number[]
  /** The frequency k of bin m: m for m < N/2, m − N from N/2 on. */
  k: number[]
  mag: number[]
  /** Bins m ≠ 0, largest |c_m| first. */
  byMag: number[]
  /** Σ_{k ≠ 0} |c_k|²: the mean squared distance of the curve from its centroid c₀ (Parseval). */
  energy: number
}

/** Complex Fourier coefficients c_k = (1/N) Σ_n z_n e^{−2πikn/N} of the closed curve z = x + iy. */
function spectrum(path: Path): Spectrum {
  const n = path.x.length
  const c = toComplexFlat(fft(complex(tensor(path.x), tensor(path.y)), { norm: 'forward' }))
  const re = c.map((v) => v.re)
  const im = c.map((v) => v.im)
  const k = toFlat(fftfreq(n, 1 / n)).map(Math.round)
  const mag = re.map((r, m) => Math.hypot(r, im[m]))
  const byMag = Array.from({ length: n - 1 }, (_, i) => i + 1).sort((a, b) => mag[b] - mag[a])
  const energy = byMag.reduce((s, m) => s + mag[m] ** 2, 0)
  return { re, im, k, mag, byMag, energy }
}

/**
 * The partial sum c₀ + Σ_{kept} c_k e^{2πikt} at t = j/m, j = 0, …, m: the kept coefficients placed at their
 * frequencies in a length-m spectrum (m > 2 max |k|, so no frequency aliases) and inverted with one inverse DFT.
 */
function partialSum(s: Spectrum, kept: readonly number[], m: number): Path {
  const re = new Array<number>(m).fill(0)
  const im = new Array<number>(m).fill(0)
  re[0] = s.re[0]
  im[0] = s.im[0]
  for (const b of kept) {
    const at = ((s.k[b] % m) + m) % m
    re[at] += s.re[b]
    im[at] += s.im[b]
  }
  const z = toComplexFlat(ifft(complex(tensor(re), tensor(im)), { norm: 'forward' }))
  return { x: [...z.map((v) => v.re), z[0].re], y: [...z.map((v) => v.im), z[0].im] }
}

/** The kept bins in the order the arrows are chained. */
function chainOrder(s: Spectrum, kept: readonly number[], order: Order, seed: number): number[] {
  const bins = [...kept]
  if (order === 'magnitude') return bins
  if (order === 'speed') return bins.sort((a, b) => Math.abs(s.k[a]) - Math.abs(s.k[b]) || s.k[b] - s.k[a])
  if (order === 'frequency') return bins.sort((a, b) => s.k[a] - s.k[b])
  return shuffle(stream(seed), bins)
}

/** Gives the charts inside a fixed height in pixels instead of the figure frame's. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/**
 * A closed curve redrawn by its complex Fourier series: one rotating arrow per kept coefficient, chained tip to tail,
 * with the traced partial sum. Shapes are typed characters (traced from the rendered font), icons, drawings, curves,
 * or the reader's own stroke; all are resampled by arc length in the browser (paths.ts, shapes.ts).
 */
export function EpicycleDrawing() {
  const state = useFigureState({
    source: choice<Source>(SOURCES, 'type', { label: 'shape' }),
    char: choice(CHARS, '&', { label: 'character', when: when('source', 'type') }),
    font: choice<FontKey>(FONTS, 'cm-italic', { label: 'font', when: when('source', 'type') }),
    icons: choice(optionsOf('icons'), 'clef-treble', { label: 'icon', when: when('source', 'icons') }),
    drawings: choice(optionsOf('drawings'), 'brontosaurus', { label: 'drawing', when: when('source', 'drawings') }),
    curves: choice(optionsOf('curves'), 'trefoil', { label: 'curve', when: when('source', 'curves') }),
    keep: choice(
      [
        { value: 'count', label: 'a number of terms' },
        { value: 'energy', label: 'a share of the energy' },
      ],
      'count',
      { label: 'keep' },
    ),
    terms: slider(1, N - 1, 60, { step: 1, label: 'terms K', when: when('keep', 'count') }),
    energy: float(99.5, {
      gt: 0,
      lt: 100,
      suggestions: [50, 90, 99, 99.5, 99.9, 99.99],
      label: 'energy kept, %',
      when: when('keep', 'energy'),
    }),
    order: choice<Order>(ORDERS, 'magnitude', { label: 'arrow order' }),
    seed: int(1, { ge: 0, label: 'seed', when: when('order', 'random') }),
    target: setting(true, 'target'),
    circles: setting(true, 'circles'),
    zoomed: setting(true, 'zoom on the pen'),
    zoom: float(100, {
      ge: 10,
      le: 1000,
      scale: 'log10',
      suggestions: [10, 30, 100, 300, 1000],
      label: 'magnification',
      format: (v) => `${formatNumber(v)}×`,
      when: when('zoomed', true),
    }),
    steps: int(STEPS, {
      ge: 64,
      le: 65536,
      scale: 'log10',
      suggestions: [256, 512, 2048, 8192, 32768, 65536],
      label: 'time steps per cycle',
    }),
  })
  const source = state.source
  const font = FONTS.find((f) => f.value === state.font) ?? FONTS[0]

  // Glyphs are traced from the rendered font, so wait for the browser to load it, then trace again.
  const [loaded, setLoaded] = useState<string | null>(null)
  useEffect(() => {
    let live = true
    const key = `${font.style} ${font.family}`
    loadFont(font.family, font.style).finally(() => live && setLoaded(key))
    return () => {
      live = false
    }
  }, [font])

  // The reader's stroke while the pointer is down, and the closed, resampled curve made from the last stroke.
  const [stroke, setStroke] = useState<Pt[] | null>(null)
  const [drawn, setDrawn] = useState<Path | null>(null)
  const [pos, setPos] = useState(0)

  const shapeId = source === 'icons' || source === 'drawings' || source === 'curves' ? state[source] : null
  const path: Path | null = useMemo(() => {
    if (source === 'draw') return drawn
    if (source === 'type') {
      if (loaded === null) return null
      const p = glyphPath(state.char, font.family, N, font.style)
      return p.length === N ? toPath(p) : null
    }
    return shapeId ? toPath(shapePath(shapeId, N)) : null
  }, [source, drawn, loaded, state.char, font, shapeId])

  const spec = useMemo(() => (path ? spectrum(path) : null), [path])

  // Kept bins: the K largest coefficients, K set directly or as the fewest that reach the energy share.
  const kept = useMemo(() => {
    if (!spec) return []
    if (state.keep === 'count') return spec.byMag.slice(0, state.terms)
    const goal = (state.energy / 100) * spec.energy
    let acc = 0
    let K = 0
    while (K < spec.byMag.length && acc < goal) acc += spec.mag[spec.byMag[K++]] ** 2
    return spec.byMag.slice(0, Math.max(1, K))
  }, [spec, state.keep, state.terms, state.energy])

  const order = state.order
  const chain = useMemo(() => (spec ? chainOrder(spec, kept, order, state.seed) : []), [spec, kept, order, state.seed])
  const terms = useMemo(() => {
    if (!spec) return null
    return {
      c: complex(tensor(chain.map((b) => spec.re[b])), tensor(chain.map((b) => spec.im[b]))) as Tensor,
      w: tensor(chain.map((b) => 2 * Math.PI * spec.k[b])),
      radius: chain.map((b) => spec.mag[b]),
      // Plain copies for the per-point sums of the zoomed view.
      re: chain.map((b) => spec.re[b]),
      im: chain.map((b) => spec.im[b]),
      freq: chain.map((b) => 2 * Math.PI * spec.k[b]),
    }
  }, [spec, chain])
  const trace = useMemo(() => (spec ? partialSum(spec, kept, FINE) : null), [spec, kept])

  // Which arrows get a circle and an arrowhead: the largest, wherever they sit in the chain.
  const marked = useMemo(() => {
    if (!terms) return { circles: [] as number[], arrows: [] as number[] }
    const rank = terms.radius.map((_, i) => i).sort((a, b) => terms.radius[b] - terms.radius[a])
    return {
      circles: rank.slice(0, MAX_CIRCLES).filter((i) => terms.radius[i] > 0.004),
      arrows: rank.slice(0, MAX_ARROWS).filter((i) => terms.radius[i] > 0.02),
    }
  }, [terms])

  // Player positions t = j / steps for j = 0, …, steps, so the last frame closes the curve. More steps at the same
  // frame rate turn the arrows more slowly, which a high magnification needs. A new step count keeps the current t.
  const steps = state.steps
  const [prevSteps, setPrevSteps] = useState(steps)
  if (prevSteps !== steps) {
    setPrevSteps(steps)
    setPos((p) => Math.round((p / prevSteps) * steps))
  }
  const at = Math.min(pos, steps)
  const t = at / steps

  // The chain at time t: c₀, then c₀ plus the running sums of c_k e^{2πikt} in chain order.
  const frame = useMemo(() => {
    if (!spec || !terms || !trace) return null
    const sums = chain.length ? toComplexFlat(cumsum(mul(terms.c, expj(mul(terms.w, t))) as Tensor)) : []
    const jx = [spec.re[0], ...sums.map((v) => v.re + spec.re[0])]
    const jy = [spec.im[0], ...sums.map((v) => v.im + spec.im[0])]
    const cx: number[] = []
    const cy: number[] = []
    for (const i of marked.circles) {
      const r = terms.radius[i]
      for (let a = 0; a <= 64; a++) {
        cx.push(jx[i] + r * Math.cos((2 * Math.PI * a) / 64))
        cy.push(jy[i] + r * Math.sin((2 * Math.PI * a) / 64))
      }
      cx.push(NaN)
      cy.push(NaN)
    }
    const arrows: Vector[] = marked.arrows.map((i) => ({
      from: [jx[i], jy[i]],
      to: [jx[i + 1], jy[i + 1]],
      head: Math.min(8, 4 + 40 * terms.radius[i]),
      width: 1,
    }))
    const upto = Math.round(t * FINE) + 1
    return {
      joints: { x: jx, y: jy },
      circles: { x: cx, y: cy },
      arrows,
      tip: [jx[jx.length - 1], jy[jy.length - 1]] as Pt,
      traced: { x: trace.x.slice(0, upto), y: trace.y.slice(0, upto) },
    }
  }, [spec, terms, trace, chain, marked, t])

  // The zoomed view, in coordinates relative to the pen, so its axes stay fixed while the pen moves. Every arrow's
  // circle that crosses the window is drawn, however small, as an arc fine enough to stay round at this scale.
  const half = RANGE[1] / state.zoom
  const zoomView = useMemo(() => {
    if (!state.zoomed || !frame || !terms || !trace) return null
    const [px, py] = frame.tip
    const jx = frame.joints.x
    const jy = frame.joints.y
    const cx: number[] = []
    const cy: number[] = []
    const arrows: Vector[] = []
    for (let i = 0; i < terms.radius.length; i++) {
      const r = terms.radius[i]
      if (r < MIN_ZOOM_RADIUS * half) continue
      const ox = jx[i] - px
      const oy = jy[i] - py
      const d = Math.hypot(ox, oy)
      if (Math.abs(d - r) > 1.5 * half) continue
      // A whole circle when it fits in the window; otherwise the arc nearest the pen.
      const span = r < 1.5 * half ? Math.PI : Math.min(Math.PI, (3 * half) / r)
      const mid = Math.atan2(-oy, -ox)
      for (let a = 0; a <= ARC_POINTS; a++) {
        const th = mid - span + (2 * span * a) / ARC_POINTS
        cx.push(ox + r * Math.cos(th))
        cy.push(oy + r * Math.sin(th))
      }
      cx.push(NaN)
      cy.push(NaN)
      if (r > half / 12 && arrows.length < MAX_ARROWS)
        arrows.push({ from: [ox, oy], to: [jx[i + 1] - px, jy[i + 1] - py], head: 6, width: 1 })
    }
    // The last stretch of the traced curve, long enough to cross the window, evaluated exactly at fine t.
    let vx = 0
    let vy = 0
    for (let i = 0; i < chain.length; i++) {
      const w = terms.freq[i]
      const a = w * t
      const [cr, ci] = [terms.re[i], terms.im[i]]
      vx += -w * (cr * Math.sin(a) + ci * Math.cos(a))
      vy += w * (cr * Math.cos(a) - ci * Math.sin(a))
    }
    const speed = Math.hypot(vx, vy)
    const span = Math.min(t, speed > 0 ? (4 * half) / speed : t)
    const tx: number[] = []
    const ty: number[] = []
    for (let j = 0; j <= ZOOM_TRACE_POINTS; j++) {
      const s = t - span + (span * j) / ZOOM_TRACE_POINTS
      let x = 0
      let y = 0
      for (let i = 0; i < chain.length; i++) {
        const a = terms.freq[i] * s
        x += terms.re[i] * Math.cos(a) - terms.im[i] * Math.sin(a)
        y += terms.re[i] * Math.sin(a) + terms.im[i] * Math.cos(a)
      }
      tx.push(x + jx[0] - px)
      ty.push(y + jy[0] - py)
    }
    const coarse = frame.traced
    const before = coarse.x.length - Math.ceil(span * FINE) - 1
    return {
      circles: { x: cx, y: cy },
      arrows,
      joints: { x: jx.map((v) => v - px), y: jy.map((v) => v - py) },
      // The traced line ends exactly at the pen, not at the nearest point of its grid.
      traced: {
        x: [...coarse.x.slice(0, Math.max(0, before)).map((v) => v - px), ...tx],
        y: [...coarse.y.slice(0, Math.max(0, before)).map((v) => v - py), ...ty],
      },
      target: path
        ? { x: [...path.x, path.x[0]].map((v) => v - px), y: [...path.y, path.y[0]].map((v) => v - py) }
        : null,
      window: {
        x: [px - half, px + half, px + half, px - half, px - half],
        y: [py - half, py - half, py + half, py + half, py - half],
      },
    }
  }, [state.zoomed, frame, terms, trace, half, path, chain, t])

  const finish = (points: Pt[] | null) => {
    setStroke(null)
    if (!points || points.length < 4) return
    let length = 0
    for (let i = 1; i < points.length; i++)
      length += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
    if (length < 0.2) return
    setDrawn(toPath(freehandPath(points, N)))
    setPos(0)
  }
  const clamp = (v: number) => Math.min(RANGE[1], Math.max(RANGE[0], v))
  const drawing = source === 'draw'
  const pen: Handle = {
    kind: 'point',
    at: stroke?.length ? stroke[stroke.length - 1] : (frame?.tip ?? [0, 0]),
    label: 'pen',
    onDrag: (p) => {
      const q: Pt = [clamp(p[0]), clamp(p[1])]
      setStroke((s) => (s ? [...s, q] : [q]))
    },
    onRelease: () => finish(stroke),
  }

  const energyKept = spec && spec.energy > 0 ? kept.reduce((a, b) => a + spec.mag[b] ** 2, 0) / spec.energy : 0
  // The cut-off on the spectrum: the smallest kept magnitude. Dragging it keeps every coefficient at least that large.
  const cutoff = spec && kept.length ? spec.mag[kept[kept.length - 1]] : 1
  // A log axis from four decades below the largest coefficient, at whole powers of ten.
  const top = spec ? 10 ** Math.ceil(Math.log10(spec.mag[spec.byMag[0]] || 1)) : 1
  const floor = top * 1e-5
  const dots = useMemo(() => {
    if (!spec) return null
    const keptSet = new Set(kept)
    const on = { x: [] as number[], y: [] as number[] }
    const off = { x: [] as number[], y: [] as number[] }
    for (const b of spec.byMag) {
      if (spec.mag[b] < floor) continue
      const side = keptSet.has(b) ? on : off
      side.x.push(spec.k[b])
      side.y.push(spec.mag[b])
    }
    return { on, off }
  }, [spec, kept, floor])
  const cut: Handle = {
    kind: 'y',
    at: cutoff,
    label: 'cut-off',
    onDrag: (y) => {
      if (!spec) return
      const K = Math.max(1, spec.byMag.filter((b) => spec.mag[b] >= y).length)
      if (state.keep === 'count') state.set('terms', K)
      else {
        const e = spec.byMag.slice(0, K).reduce((a, b) => a + spec.mag[b] ** 2, 0) / spec.energy
        state.set('energy', Math.min(99.999, Math.max(0.1, Number((100 * e).toFixed(3)))))
      }
    },
  }

  // The square plots' height follows their width; the spectrum strip below takes half of it.
  const [cell, cellSize] = useElementSize<HTMLDivElement>()
  const stripHeight = Math.max(140, Math.round(cellSize.width / 2))

  const xAxis = useAxis({ label: 'Re z', range: RANGE })
  const yAxis = useAxis({ label: 'Im z', range: RANGE, equal: xAxis })
  const zxAxis = useAxis({ label: 'Re z − pen', range: [-half, half] })
  const zyAxis = useAxis({ label: 'Im z − pen', range: [-half, half], equal: zxAxis })
  // The frequency axis spans the coefficients the log axis shows, not all N bins, most of which are below its floor.
  const kmax = useMemo(() => {
    if (!dots) return 64
    const m = Math.max(32, ...dots.on.x.map(Math.abs), ...dots.off.x.map(Math.abs))
    return Math.min(N / 2, 2 ** Math.ceil(Math.log2(m)))
  }, [dots])
  const kAxis = useAxis({ label: 'frequency k (turns per cycle)', integer: true, range: [-kmax, kmax] })
  const magAxis = useAxis({ label: '|cₖ|', log: true, range: [floor, top] })

  // While the reader draws, only the stroke is shown.
  const shown = stroke ? null : frame

  const credit =
    source === 'type'
      ? `${font.label}, traced from the rendered font`
      : shapeId
        ? SHAPES.find((s) => s.id === shapeId)?.credit
        : 'your drawing'

  return (
    <Figure
      title="Drawing with rotating arrows"
      state={state}
      defaultSize="XL"
      hoverReadout={false}
      controls={
        <Player
          value={at}
          onChange={setPos}
          count={steps + 1}
          label="time t"
          format={(p) => (p / steps).toFixed(steps > 5000 ? 5 : 3)}
          loop
        />
      }
      readouts={
        <>
          <Readout label="terms K" value={kept.length} />
          <Readout label="energy kept" value={`${formatNumber(100 * energyKept)}%`} />
          <Readout
            label="RMS error / RMS radius"
            value={`${formatNumber(100 * Math.sqrt(Math.max(0, 1 - energyKept)))}%`}
          />
          <Readout
            label="highest |k| kept"
            value={spec && kept.length ? Math.max(...kept.map((b) => Math.abs(spec.k[b]))) : '–'}
          />
          <Readout label="source" value={credit ?? '–'} />
        </>
      }
      caption={
        <>
          Each kept coefficient cₖ is an arrow of length |cₖ| that turns k times per cycle, anticlockwise for k &gt; 0
          and clockwise for k &lt; 0. The arrows are chained tip to tail from the centroid c₀, and the last tip draws
          the partial sum (blue) over the target shape (grey); the thin blue circles are the paths the arrow tips ride
          on. With “zoom on the pen”, the right plot follows the pen at the chosen magnification (10× to 1000×, typed or
          stepped) and shows every circle that passes through its window, so the small fast arrows near the tip become
          visible; the dashed box on the left marks the window. Play or scrub t; more time steps per cycle play the same
          cycle more slowly, for high magnifications. Changing the arrow order changes the chain but not the curve,
          because addition commutes. With “draw your own”, press and drag on the upper plot to draw one stroke; on
          release it is closed with a straight segment, resampled by arc length, centred and redrawn. On the spectrum
          below, drag the cut-off line to keep every coefficient above it. Icons are from Lucide (ISC licence).
        </>
      }
    >
      <div className="flex flex-col gap-2">
        {/* Two square plots, each half the width; their height follows their width, so the figure grows to fit. */}
        <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
          <div ref={cell}>
            <Plot x={xAxis} y={yAxis} legend={false} renderer="canvas" fitHeight>
              {path && state.target && (
                <Curve name="target" x={[...path.x, path.x[0]]} y={[...path.y, path.y[0]]} muted width={1.5} silent />
              )}
              {shown && state.circles && (
                <Curve name="circles" x={shown.circles.x} y={shown.circles.y} slot={0} thin width={0.75} silent live />
              )}
              {shown && <Curve name="chain" x={shown.joints.x} y={shown.joints.y} emphasis width={1} silent live />}
              {shown && <Vectors name="arrows" vectors={shown.arrows} live />}
              {shown && <Curve name="traced" x={shown.traced.x} y={shown.traced.y} slot={0} width={2} silent live />}
              {shown && !drawing && <Points name="pen" x={[shown.tip[0]]} y={[shown.tip[1]]} emphasis size={7} live />}
              {stroke && (
                <Curve
                  name="stroke"
                  x={stroke.map((p) => p[0])}
                  y={stroke.map((p) => p[1])}
                  slot={1}
                  width={2}
                  silent
                  live
                />
              )}
              {drawing && !drawn && !stroke && <Annotation at={[0, 0]} text="press and drag to draw" />}
              {shown && zoomView && (
                <Curve
                  name="zoom window"
                  x={zoomView.window.x}
                  y={zoomView.window.y}
                  emphasis
                  dashed
                  width={1}
                  silent
                  live
                />
              )}
              {drawing && <Handle {...pen} />}
            </Plot>
          </div>
          {zoomView && !stroke && (
            <Plot x={zxAxis} y={zyAxis} legend={false} renderer="canvas" fitHeight>
              {state.target && zoomView.target && (
                <Curve name="target" x={zoomView.target.x} y={zoomView.target.y} muted width={1.5} silent live />
              )}
              {state.circles && (
                <Curve
                  name="circles"
                  x={zoomView.circles.x}
                  y={zoomView.circles.y}
                  slot={0}
                  thin
                  width={0.75}
                  silent
                  live
                />
              )}
              <Curve name="chain" x={zoomView.joints.x} y={zoomView.joints.y} emphasis width={1} silent live />
              <Vectors name="arrows" vectors={zoomView.arrows} live />
              <Curve name="traced" x={zoomView.traced.x} y={zoomView.traced.y} slot={0} width={2} silent live />
              <Points name="pen" x={[0]} y={[0]} emphasis size={7} live />
            </Plot>
          )}
        </div>
        <FixedHeight height={stripHeight}>
          <Plot x={kAxis} y={magAxis} legend={false}>
            {dots && <Points name="dropped" x={dots.off.x} y={dots.off.y} muted thin />}
            {dots && <Points name="kept" x={dots.on.x} y={dots.on.y} slot={0} size={5} />}
            {spec && <Handle {...cut} />}
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
