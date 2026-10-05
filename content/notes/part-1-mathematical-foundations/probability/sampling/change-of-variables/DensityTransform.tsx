import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Bars,
  Button,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  type PlotPointer,
  Plots,
  Points,
  Readout,
  Segments,
  setting,
  Shapes,
  slider,
  useAxis,
  useFigureState,
  type FilledShape,
  type Vec2,
} from 'aifn-render'

type Seg = { from: Vec2; to: Vec2 }
/** A closed polygon as a filled shape: ink unless a palette slot is given. */
const fill = (points: Vec2[], tone: FilledShape['tone'], opacity: number): FilledShape => ({
  contours: [points],
  tone,
  opacity,
})
const rect = (x: Vec2, y: Vec2): FilledShape =>
  fill(
    [
      [x[0], y[0]],
      [x[1], y[0]],
      [x[1], y[1]],
      [x[0], y[1]],
    ],
    'muted',
    0.12,
  )
/** Swap (y, p) pairs to (p, y) for the right panel, whose x is the density and y the shared position. */
const swap = (pts: Vec2[]): Vec2[] => pts.map(([u, v]) => [v, u])
const xsOf = (pts: Vec2[]) => pts.map((p) => p[0])
const ysOf = (pts: Vec2[]) => pts.map((p) => p[1])
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream } from 'aifn-compute/foundation/random'
import { normalCdf, normalPdf, sigmoid } from 'aifn-compute/numerics/special'

type MapId = 'custom' | 'affine' | 'exp' | 'logistic' | 'ramp'

/** A strictly increasing map with its derivative, and the y-range to plot it on. */
type Transform = { g: (x: number) => number; dg: (x: number) => number; yRange: Vec2 }

const X_RANGE: Vec2 = [-4, 4]
/** Width of the highlighted band on the x-axis, and on the y-axis as a share of its range. */
const BAND_X = 0.4
const BAND_Y = 0.05
const BINS = 40
const PATHS = 24
/** Knots of the custom map: the middle segment is steep (stretch), the outer ones shallow (squash). */
const KNOTS: Vec2[] = [
  [-4, -3.8],
  [-2.4, -2.9],
  [-0.8, -2.2],
  [0.8, 1.6],
  [2.4, 2.5],
  [4, 3.8],
]
const KNOT_GAP_X = 0.4
const KNOT_GAP_Y = 0.1

const LABELS: Record<MapId, string> = {
  custom: 'custom',
  affine: 'y = ax',
  exp: 'y = eˣ',
  logistic: 'y = σ(2x)',
  ramp: 'y = x + c tanh 3x',
}

function transform(id: MapId, a: number, c: number, knots: Vec2[]): Transform {
  switch (id) {
    case 'affine':
      return { g: (x) => a * x, dg: () => a, yRange: [-6, 6] }
    case 'exp':
      return { g: Math.exp, dg: Math.exp, yRange: [0, 8] }
    case 'logistic':
      return {
        g: (x) => sigmoid(2 * x),
        dg: (x) => 2 * sigmoid(2 * x) * (1 - sigmoid(2 * x)),
        yRange: [0, 1],
      }
    case 'ramp':
      return {
        g: (x) => x + c * Math.tanh(3 * x),
        dg: (x) => 1 + (3 * c) / Math.cosh(3 * x) ** 2,
        yRange: [-7, 7],
      }
    case 'custom':
      return { ...monotoneCubic(knots), yRange: [-4, 4] }
  }
}

/**
 * Monotone cubic interpolation through increasing knots (Fritsch and Carlson): Hermite cubics with tangents limited so
 * that no segment overshoots. Beyond the end knots the map continues linearly with the end tangents.
 */
function monotoneCubic(knots: Vec2[]): Pick<Transform, 'g' | 'dg'> {
  const n = knots.length
  const xs = knots.map((k) => k[0])
  const ys = knots.map((k) => k[1])
  const h = xs.slice(1).map((x, i) => x - xs[i])
  const d = h.map((w, i) => (ys[i + 1] - ys[i]) / w)
  const m = xs.map((_, i) => (i === 0 ? d[0] : i === n - 1 ? d[n - 2] : (d[i - 1] + d[i]) / 2))
  d.forEach((di, i) => {
    const alpha = m[i] / di
    const beta = m[i + 1] / di
    const r = Math.hypot(alpha, beta)
    if (r > 3) {
      m[i] = (3 * alpha * di) / r
      m[i + 1] = (3 * beta * di) / r
    }
  })
  const segment = (x: number) => {
    let i = 0
    while (i < n - 2 && x > xs[i + 1]) i++
    return i
  }
  const g = (x: number) => {
    if (x <= xs[0]) return ys[0] + m[0] * (x - xs[0])
    if (x >= xs[n - 1]) return ys[n - 1] + m[n - 1] * (x - xs[n - 1])
    const i = segment(x)
    const t = (x - xs[i]) / h[i]
    const t2 = t * t
    const t3 = t2 * t
    return (
      (2 * t3 - 3 * t2 + 1) * ys[i] +
      (t3 - 2 * t2 + t) * h[i] * m[i] +
      (-2 * t3 + 3 * t2) * ys[i + 1] +
      (t3 - t2) * h[i] * m[i + 1]
    )
  }
  const dg = (x: number) => {
    if (x <= xs[0]) return m[0]
    if (x >= xs[n - 1]) return m[n - 1]
    const i = segment(x)
    const t = (x - xs[i]) / h[i]
    const t2 = t * t
    const slope =
      ((6 * t2 - 6 * t) * ys[i] + (-6 * t2 + 6 * t) * ys[i + 1]) / h[i] +
      (3 * t2 - 4 * t + 1) * m[i] +
      (3 * t2 - 2 * t) * m[i + 1]
    return Math.max(slope, 1e-9)
  }
  return { g, dg }
}

/** g⁻¹(y) by bisection, since every map here is increasing. Clamps to the bracket when y is outside the image. */
function inverse(g: (x: number) => number, y: number, lo = -40, hi = 40): number {
  for (let i = 0; i < 64; i++) {
    const mid = (lo + hi) / 2
    if (g(mid) < y) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}

function histogram(values: number[], [lo, hi]: Vec2) {
  const width = (hi - lo) / BINS
  const counts = new Array<number>(BINS).fill(0)
  for (const v of values) {
    const bin = Math.floor((v - lo) / width)
    if (bin >= 0 && bin < BINS) counts[bin]++
  }
  return {
    edges: toFlat(linspace(lo, hi, BINS + 1)),
    heights: counts.map((k) => k / (values.length * width)),
  }
}

function trapezoid(xs: number[], ys: number[]) {
  let total = 0
  for (let i = 1; i < xs.length; i++) total += ((xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1])) / 2
  return total
}

/** Round up to 1, 1.5, 2, 2.5, 3, 4, 5, 6 or 8 times a power of ten. */
function niceCeil(v: number) {
  const p = 10 ** Math.floor(Math.log10(v))
  return ([1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].find((s) => s * p >= v) ?? 10) * p
}

/**
 * An axis maximum that follows `peak` with hysteresis: it grows as soon as the peak would be clipped and shrinks only
 * when the peak falls below 40% of it, so the axis holds still through most slider drags.
 */
function useStickyMax(peak: number) {
  const [max, setMax] = useState(() => niceCeil(peak * 1.15))
  const stale = peak > max || peak < 0.4 * max
  const next = stale ? niceCeil(peak * 1.15) : max
  if (next !== max) setMax(next)
  return next
}

/** Where the band is anchored: a centre on the x-axis, or a centre on the y-axis carried back through g⁻¹. */
type Probe = { from: 'x' | 'y'; at: number }

const DEFAULT_PROBE: Probe = { from: 'x', at: 0.2 }
const SAMPLE_SIZES = ['200', '1000', '5000', '20000'] as const
type SampleSize = (typeof SAMPLE_SIZES)[number]

/**
 * Three aligned panels: y = g(x) in the middle, the Gaussian source density of x below it, and the density of
 * Y = g(X) rotated on its right. A band of x values, its image under g and the equal mass in both are highlighted
 * from the pointer; the Jacobian switch shows what the density of Y becomes without the stretch factor.
 */
export function DensityTransform() {
  const state = useFigureState({
    id: choice<MapId>(
      (Object.keys(LABELS) as MapId[]).map((k) => ({ value: k, label: LABELS[k] })),
      'custom',
      { label: 'map g' },
    ),
    mu: float(0, { min: -1.5, max: 1.5, step: 0.05, label: 'source mean μ' }),
    sigma: float(1, { min: 0.25, max: 1.25, step: 0.05, label: 'source standard deviation σ' }),
    size: choice<SampleSize>(
      SAMPLE_SIZES.map((v) => ({ value: v, label: Number(v).toLocaleString('en') })),
      '5000',
      { label: 'samples' },
    ),
    seed: int(1, { min: 1, max: 20, step: 1, label: 'seed', format: (v) => String(v) }),
    jacobian: setting(true, 'Jacobian factor |dg⁻¹/dy|'),
    paths: setting(false, 'show sample paths'),
    slope: slider(0.25, 2, 1.5, { step: 0.05, label: 'slope a', when: (v) => v.id === 'affine' }),
    ramp: slider(0, 3, 2, { step: 0.1, label: 'step height c', when: (v) => v.id === 'ramp' }),
  })
  const [knots, setKnots] = useState<Vec2[]>(KNOTS)
  const [hover, setHover] = useState<Probe | null>(null)
  const [pinned, setPinned] = useState<Probe | null>(null)

  const map = useMemo(
    () => transform(state.id, state.slope, state.ramp, knots),
    [state.id, state.slope, state.ramp, knots],
  )
  const { g, dg, yRange } = map
  const [x0, x1] = X_RANGE
  const [y0, y1] = yRange
  const m = state.mu
  const s = state.sigma
  const pX = useCallback((x: number) => normalPdf((x - m) / s) / s, [m, s])

  // Standard normal draws depend only on the seed and size; the source samples are m + s z, so they move smoothly.
  const z = useMemo(() => {
    const r = stream(state.seed)
    return Array.from({ length: Number(state.size) }, () => normal(r))
  }, [state.seed, state.size])
  const xs = useMemo(() => z.map((v) => m + s * v), [z, m, s])
  const ys = useMemo(() => xs.map(g), [xs, g])
  const sourceHist = useMemo(() => histogram(xs, X_RANGE), [xs])
  const targetHist = useMemo(() => histogram(ys, yRange), [ys, yRange])

  // Density curves on a fixed grid. The target curve is parametrised by x: an even grid in x resolves regions that g
  // squashes, and the preimage of an even grid in y resolves regions that g stretches.
  const curves = useMemo(() => {
    const grid = toFlat(linspace(x0, x1, 241))
    const source: Vec2[] = grid.map((x) => [x, pX(x)])
    const params = [
      ...toFlat(linspace(m - 6 * s, m + 6 * s, 300)),
      ...toFlat(linspace(y0, y1, 300)).map((y) => inverse(g, y)),
    ]
      .filter((x) => {
        const y = g(x)
        return y >= y0 && y <= y1
      })
      .sort((a, b) => a - b)
    const target: Vec2[] = params.map((x) => [g(x), pX(x) / dg(x)])
    const naive: Vec2[] = params.map((x) => [g(x), pX(x)])
    // ∫ p_X(g⁻¹(y)) dy = ∫ p_X(x) g′(x) dx = E[g′(X)] after substituting y = g(x).
    const wide = toFlat(linspace(m - 8 * s, m + 8 * s, 801))
    const naiveMass = trapezoid(
      wide,
      wide.map((x) => pX(x) * dg(x)),
    )
    const peak = (pts: Vec2[]) => pts.reduce((a, p) => Math.max(a, p[1]), 0)
    return {
      source,
      target,
      naive,
      naiveMass,
      sourcePeak: peak(source),
      targetPeak: Math.max(peak(target), peak(naive)),
    }
  }, [g, dg, pX, m, s, x0, x1, y0, y1])

  const bottomMax = useStickyMax(curves.sourcePeak)
  const rightMax = useStickyMax(Math.min(curves.targetPeak, 8))

  const curve = useMemo(() => {
    const pts = toFlat(linspace(x0, x1, 241)).map((x): Vec2 => [x, g(x)])
    return { x: xsOf(pts), y: ysOf(pts) }
  }, [g, x0, x1])
  const targetBars = useMemo(() => {
    const centres = targetHist.edges.slice(1).map((e, i) => (e + targetHist.edges[i]) / 2)
    return { centres, ...targetHist }
  }, [targetHist])
  const sourceCentres = useMemo(
    () => sourceHist.edges.slice(1).map((e, i) => (e + sourceHist.edges[i]) / 2),
    [sourceHist],
  )
  const rightCurves = useMemo(() => {
    const target = swap(curves.target)
    const naive = swap(curves.naive)
    // The gap between the curve without the Jacobian and the true density, which the histogram follows.
    const gap = swap([
      ...curves.naive,
      ...curves.target.map(([y, p]): Vec2 => [y, Math.min(p, 2 * rightMax)]).reverse(),
    ])
    return { target, naive, gap }
  }, [curves, rightMax])
  // A few sample paths: up from the x-axis to g, then across to the right panel.
  const paths = useMemo(() => {
    if (!state.paths) return undefined
    const shown = xs.slice(0, PATHS).filter((x) => x > x0 && x < x1 && g(x) > y0 && g(x) < y1)
    const lift = (max: number) => 0.04 * max
    return {
      bottom: shown.map((x): Seg => ({ from: [x, lift(bottomMax)], to: [x, bottomMax] })),
      bottomDots: { x: shown, y: shown.map(() => lift(bottomMax)) },
      main: shown.flatMap((x): Seg[] => [
        { from: [x, y0], to: [x, g(x)] },
        { from: [x, g(x)], to: [x1, g(x)] },
      ]),
      mainDots: { x: shown, y: shown.map(g) },
      rightDots: { x: shown.map(() => lift(rightMax)), y: shown.map(g) },
    }
  }, [state.paths, xs, g, x0, x1, y0, y1, bottomMax, rightMax])

  // The band: pinned if the reader pinned one, else under the pointer, else a default.
  const probe = pinned ?? hover ?? DEFAULT_PROBE
  const band = useMemo(() => {
    if (probe.from === 'x') {
      const c = Math.min(Math.max(probe.at, x0 + BAND_X / 2), x1 - BAND_X / 2)
      return { a: c - BAND_X / 2, b: c + BAND_X / 2 }
    }
    const h = (BAND_Y * (y1 - y0)) / 2
    const c = Math.min(Math.max(probe.at, y0 + h), y1 - h)
    return { a: inverse(g, c - h), b: inverse(g, c + h) }
  }, [probe, g, x0, x1, y0, y1])

  const { a, b } = band
  const ya = g(a)
  const yb = g(b)
  const massX = normalCdf((b - m) / s) - normalCdf((a - m) / s)
  // The curve's mass over the image band, integrated in y, so the Jacobian's effect shows as a number.
  const massY = useMemo(() => {
    const grid = toFlat(linspace(ya, yb, 41))
    return trapezoid(
      grid,
      grid.map((y) => {
        const x = inverse(g, y)
        return state.jacobian ? pX(x) / dg(x) : pX(x)
      }),
    )
  }, [ya, yb, g, dg, pX, state.jacobian])

  const overlay = useMemo(() => {
    const ts = toFlat(linspace(a, b, 24))
    const arc = ts.map((t): Vec2 => [t, g(t)])
    const end = arc[arc.length - 1]
    const pY = (t: number) => (state.jacobian ? pX(t) / dg(t) : pX(t))
    return {
      bottom: [rect([a, b], [0, bottomMax]), fill([[a, 0], ...ts.map((t): Vec2 => [t, pX(t)]), [b, 0]], 0, 0.7)],
      bottomGuides: [
        { from: [a, 0], to: [a, bottomMax] },
        { from: [b, 0], to: [b, bottomMax] },
      ] satisfies Seg[],
      main: [fill([[a, y0], ...arc, [b, y0]], 'ink', 0.08), fill([...arc, [x1, yb], [x1, ya]], 'ink', 0.08)],
      mainGuides: [
        { from: [a, y0], to: arc[0] },
        { from: [b, y0], to: end },
        { from: arc[0], to: [x1, ya] },
        { from: end, to: [x1, yb] },
      ] satisfies Seg[],
      arc: { x: xsOf(arc), y: ysOf(arc) },
      right: [
        rect([0, rightMax], [ya, yb]),
        fill(swap([[ya, 0], ...ts.map((t): Vec2 => [g(t), pY(t)]), [yb, 0]]), state.jacobian ? 1 : 2, 0.7),
      ],
      rightGuides: [
        { from: [0, ya], to: [rightMax, ya] },
        { from: [0, yb], to: [rightMax, yb] },
      ] satisfies Seg[],
    }
  }, [a, b, ya, yb, g, dg, pX, state.jacobian, bottomMax, rightMax, x1, y0])

  // Hover in the main panel follows the nearer of two points on the curve: straight above or below the pointer, or
  // level with it. Distances are in shares of each axis, so a steep or a flat curve is equally easy to follow.
  const locate = useCallback(
    (panel: 'main' | 'bottom' | 'right', [u, v]: [number, number]): Probe => {
      if (panel === 'bottom') return { from: 'x', at: u }
      if (panel === 'right') return { from: 'y', at: v }
      const level = inverse(g, v)
      const above = Math.abs(g(u) - v) / (y1 - y0)
      const beside = Math.abs(level - u) / (x1 - x0)
      return above <= beside ? { from: 'x', at: u } : { from: 'x', at: level }
    },
    [g, x0, x1, y0, y1],
  )
  const pointer = useCallback(
    (panel: 'main' | 'bottom' | 'right') => (e: PlotPointer) => {
      if (e.type === 'leave') return setHover(null)
      const next = locate(panel, e.point)
      if (e.type === 'move') return setHover(next)
      setPinned((p) => (p ? null : next))
      setHover(next)
    },
    [locate],
  )
  const onMain = useMemo(() => pointer('main'), [pointer])
  const onBottom = useMemo(() => pointer('bottom'), [pointer])
  const onRight = useMemo(() => pointer('right'), [pointer])
  useEffect(() => {
    if (!pinned) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setPinned(null)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [pinned])

  const dragKnot = useCallback(
    (i: number) =>
      ([x, y]: Vec2) =>
        setKnots((prev) => {
          const last = prev.length - 1
          const nx =
            i === 0 || i === last
              ? prev[i][0]
              : Math.min(Math.max(x, prev[i - 1][0] + KNOT_GAP_X), prev[i + 1][0] - KNOT_GAP_X)
          const lo = i === 0 ? y0 : prev[i - 1][1] + KNOT_GAP_Y
          const hi = i === last ? y1 : prev[i + 1][1] - KNOT_GAP_Y
          const next = [...prev]
          next[i] = [nx, Math.min(Math.max(y, lo), hi)]
          return next
        }),
    [y0, y1],
  )

  const xAxis = useAxis({ label: 'x', range: X_RANGE })
  const yAxis = useAxis({ label: 'y = g(x)', range: yRange })
  const bottomAxis = useAxis({ label: 'p_X(x)', range: [0, bottomMax] })
  const rightAxis = useAxis({ label: 'density of y', range: [0, rightMax] })

  return (
    <Figure
      title="Stretching space thins out density"
      state={state}
      caption={
        <>
          X is Gaussian with mean μ and standard deviation σ (histogram and curve below); Y = g(X) is on the right,
          sharing the y-axis of g. Hover over any panel to follow a band of x values through g: its image is wider where
          g is steep, and the same probability spread over it gives a lower density. Click to pin the band, and click
          again or press Esc to release it. In the custom map, drag the knots to reshape g. Turn off the Jacobian factor
          to plot p_X(g⁻¹(y)) alone: it no longer matches the histogram, and its total area is not 1.
        </>
      }
      controls={
        state.id === 'custom' ? (
          <Button variant="outline" size="sm" onClick={() => setKnots(KNOTS)}>
            Reset knots
          </Button>
        ) : undefined
      }
      readouts={
        <>
          <Readout label="variance σ²" value={formatNumber(s * s)} />
          <Readout label="stretch Δy/Δx" value={formatNumber((yb - ya) / (b - a))} />
          <Readout label="P(X in x-band)" value={formatNumber(massX)} />
          <Readout
            label={state.jacobian ? '∫ p_Y over y-band' : '∫ p_X(g⁻¹(y)) over y-band'}
            value={formatNumber(massY)}
          />
          <Readout
            label={state.jacobian ? '∫ p_Y dy' : '∫ p_X(g⁻¹(y)) dy'}
            value={state.jacobian ? '1.00' : `${curves.naiveMass.toFixed(2)}, not 1`}
          />
          {pinned && <Readout label="band" value="pinned" />}
        </>
      }
    >
      <Plots rows={2} cols={2} heights={[3, 1]} widths={[3, 1]} height={540}>
        <Plot
          x={xAxis}
          y={yAxis}
          onPointer={onMain}
          ariaLabel="The map y = g(x), with the source density of x below it and the density of y = g(x) to its right"
        >
          <Curve name="g" x={curve.x} y={curve.y} emphasis />
          {paths && <Segments segments={paths.main} muted />}
          {paths && <Points name="sample" x={paths.mainDots.x} y={paths.mainDots.y} emphasis size={5} />}
          <Shapes shapes={overlay.main} live />
          <Segments segments={overlay.mainGuides} dashed live />
          <Curve name="band" x={overlay.arc.x} y={overlay.arc.y} emphasis width={4} live />
          {state.id === 'custom' &&
            knots.map((k, i) => <Handle key={i} kind="point" at={k} label={`knot ${i + 1}`} onDrag={dragKnot(i)} />)}
        </Plot>
        <Plot x={rightAxis} y={yAxis} onPointer={onRight}>
          <Bars
            name="samples of Y"
            x={targetBars.centres}
            y={targetBars.heights}
            edges={targetBars.edges}
            orient="y"
            slot={1}
          />
          {state.jacobian ? (
            <Curve name="p_Y(y)" x={xsOf(rightCurves.target)} y={ysOf(rightCurves.target)} slot={1} />
          ) : (
            <>
              <Shapes shapes={[fill(rightCurves.gap, 2, 0.22)]} />
              <Curve name="p_X(g⁻¹(y)), no Jacobian" x={xsOf(rightCurves.naive)} y={ysOf(rightCurves.naive)} slot={2} />
            </>
          )}
          {paths && <Points name="sample" x={paths.rightDots.x} y={paths.rightDots.y} slot={1} />}
          <Shapes shapes={overlay.right} live />
          <Segments segments={overlay.rightGuides} dashed live />
        </Plot>
        <Plot x={xAxis} y={bottomAxis} onPointer={onBottom}>
          <Bars name="samples of X" x={sourceCentres} y={sourceHist.heights} edges={sourceHist.edges} slot={0} />
          <Curve name="p_X(x)" x={xsOf(curves.source)} y={ysOf(curves.source)} slot={0} />
          {paths && <Segments segments={paths.bottom} muted />}
          {paths && <Points name="sample" x={paths.bottomDots.x} y={paths.bottomDots.y} slot={0} />}
          <Shapes shapes={overlay.bottom} live />
          <Segments segments={overlay.bottomGuides} dashed live />
        </Plot>
        <div />
      </Plots>
    </Figure>
  )
}
