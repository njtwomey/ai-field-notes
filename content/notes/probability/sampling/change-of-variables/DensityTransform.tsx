import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Interactive,
  MarginalPanels,
  ParamButton,
  ParamChoice,
  ParamSlider,
  ParamSwitch,
  Readout,
  formatNumber,
  useParam,
  type Handle,
  type PanelMark,
  type PanelMarks,
  type PanelPointer,
  type Vec2,
} from 'aifn-render'
import { linspace, rng, sigmoid } from '@/lib/math'
import { normalCdf, normalPdf } from '@/lib/math/special'

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
    edges: linspace(lo, hi, BINS + 1),
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
  const [id, setId] = useState<MapId>('custom')
  const slope = useParam(1.5, { min: 0.25, max: 2, step: 0.05 })
  const ramp = useParam(2, { min: 0, max: 3, step: 0.1 })
  const mu = useParam(0, { min: -1.5, max: 1.5, step: 0.05 })
  const sigma = useParam(1, { min: 0.25, max: 1.25, step: 0.05 })
  const seed = useParam(1, { min: 1, max: 20, step: 1 })
  const [size, setSize] = useState<SampleSize>('5000')
  const [jacobian, setJacobian] = useState(true)
  const [paths, setPaths] = useState(false)
  const [knots, setKnots] = useState<Vec2[]>(KNOTS)
  const [hover, setHover] = useState<Probe | null>(null)
  const [pinned, setPinned] = useState<Probe | null>(null)

  const map = useMemo(() => transform(id, slope.value, ramp.value, knots), [id, slope.value, ramp.value, knots])
  const { g, dg, yRange } = map
  const [x0, x1] = X_RANGE
  const [y0, y1] = yRange
  const m = mu.value
  const s = sigma.value
  const pX = useCallback((x: number) => normalPdf((x - m) / s) / s, [m, s])

  // Standard normal draws depend only on the seed and size; the source samples are m + s z, so they move smoothly.
  const z = useMemo(() => {
    const r = rng(seed.value)
    return Array.from({ length: Number(size) }, () => r.normal())
  }, [seed.value, size])
  const xs = useMemo(() => z.map((v) => m + s * v), [z, m, s])
  const ys = useMemo(() => xs.map(g), [xs, g])
  const sourceHist = useMemo(() => histogram(xs, X_RANGE), [xs])
  const targetHist = useMemo(() => histogram(ys, yRange), [ys, yRange])

  // Density curves on a fixed grid. The target curve is parametrised by x: an even grid in x resolves regions that g
  // squashes, and the preimage of an even grid in y resolves regions that g stretches.
  const curves = useMemo(() => {
    const grid = linspace(x0, x1, 241)
    const source: Vec2[] = grid.map((x) => [x, pX(x)])
    const params = [...linspace(m - 6 * s, m + 6 * s, 300), ...linspace(y0, y1, 300).map((y) => inverse(g, y))]
      .filter((x) => {
        const y = g(x)
        return y >= y0 && y <= y1
      })
      .sort((a, b) => a - b)
    const target: Vec2[] = params.map((x) => [g(x), pX(x) / dg(x)])
    const naive: Vec2[] = params.map((x) => [g(x), pX(x)])
    // ∫ p_X(g⁻¹(y)) dy = ∫ p_X(x) g′(x) dx = E[g′(X)] after substituting y = g(x).
    const wide = linspace(m - 8 * s, m + 8 * s, 801)
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

  const marks = useMemo((): PanelMarks => {
    const curve = linspace(x0, x1, 241).map((x): Vec2 => [x, g(x)])
    const right: PanelMark[] = [{ kind: 'bars', ...targetHist, slot: 1 }]
    if (jacobian) right.push({ kind: 'line', points: curves.target, name: 'p_Y(y)', slot: 1 })
    else {
      // The gap between the curve without the Jacobian and the true density, which the histogram follows.
      const gap = [...curves.naive, ...curves.target.map(([y, p]): Vec2 => [y, Math.min(p, 2 * rightMax)]).reverse()]
      right.push(
        { kind: 'fill', points: gap, slot: 2, opacity: 0.22 },
        { kind: 'line', points: curves.naive, name: 'p_X(g⁻¹(y)), no Jacobian', slot: 2 },
      )
    }
    const out: Required<PanelMarks> = {
      main: [{ kind: 'line', points: curve, emphasis: true }],
      bottom: [
        { kind: 'bars', ...sourceHist, slot: 0 },
        { kind: 'line', points: curves.source, name: 'p_X(x)', slot: 0 },
      ],
      right,
    }
    if (paths) {
      const shown = xs.slice(0, PATHS).filter((x) => x > x0 && x < x1 && g(x) > y0 && g(x) < y1)
      const lift = (max: number) => 0.04 * max
      out.bottom.push(
        {
          kind: 'segments',
          segments: shown.map((x) => [
            [x, lift(bottomMax)],
            [x, bottomMax],
          ]),
          opacity: 0.35,
        },
        { kind: 'dots', points: shown.map((x) => [x, lift(bottomMax)]), slot: 0 },
      )
      out.main.push(
        {
          kind: 'segments',
          segments: shown.flatMap((x): [Vec2, Vec2][] => [
            [
              [x, y0],
              [x, g(x)],
            ],
            [
              [x, g(x)],
              [x1, g(x)],
            ],
          ]),
          opacity: 0.35,
        },
        { kind: 'dots', points: shown.map((x) => [x, g(x)]), emphasis: true, size: 2.5 },
      )
      out.right.push({ kind: 'dots', points: shown.map((x) => [g(x), lift(rightMax)]), slot: 1 })
    }
    return out
  }, [g, x0, x1, y0, y1, curves, sourceHist, targetHist, jacobian, paths, xs, bottomMax, rightMax])

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
    const grid = linspace(ya, yb, 41)
    return trapezoid(
      grid,
      grid.map((y) => {
        const x = inverse(g, y)
        return jacobian ? pX(x) / dg(x) : pX(x)
      }),
    )
  }, [ya, yb, g, dg, pX, jacobian])

  const overlay = useMemo((): PanelMarks => {
    const ts = linspace(a, b, 24)
    const arc = ts.map((t): Vec2 => [t, g(t)])
    const guide = { kind: 'segments' as const, dashed: true, opacity: 0.7 }
    return {
      bottom: [
        { kind: 'rect', u: [a, b], v: [0, bottomMax] },
        { kind: 'fill', points: [[a, 0], ...ts.map((t): Vec2 => [t, pX(t)]), [b, 0]], slot: 0, opacity: 0.7 },
        {
          ...guide,
          segments: [
            [
              [a, 0],
              [a, bottomMax],
            ],
            [
              [b, 0],
              [b, bottomMax],
            ],
          ],
        },
      ],
      main: [
        { kind: 'fill', points: [[a, y0], ...arc, [b, y0]], opacity: 0.08 },
        { kind: 'fill', points: [...arc, [x1, yb], [x1, ya]], opacity: 0.08 },
        {
          ...guide,
          segments: [
            [[a, y0], arc[0]],
            [[b, y0], arc[arc.length - 1]],
            [arc[0], [x1, ya]],
            [arc[arc.length - 1], [x1, yb]],
          ],
        },
        { kind: 'line', points: arc, emphasis: true, width: 4 },
      ],
      right: [
        { kind: 'rect', u: [ya, yb], v: [0, rightMax] },
        {
          kind: 'fill',
          points: [[ya, 0], ...ts.map((t): Vec2 => [g(t), jacobian ? pX(t) / dg(t) : pX(t)]), [yb, 0]],
          slot: jacobian ? 1 : 2,
          opacity: 0.7,
        },
        {
          ...guide,
          segments: [
            [
              [ya, 0],
              [ya, rightMax],
            ],
            [
              [yb, 0],
              [yb, rightMax],
            ],
          ],
        },
      ],
    }
  }, [a, b, ya, yb, g, dg, pX, jacobian, bottomMax, rightMax, x1, y0])

  // Hover in the main panel follows the nearer of two points on the curve: straight above or below the pointer, or
  // level with it. Distances are in shares of each axis, so a steep or a flat curve is equally easy to follow.
  const locate = useCallback(
    (e: Exclude<PanelPointer, { type: 'leave' }>): Probe => {
      const [u, v] = e.point
      if (e.panel === 'bottom') return { from: 'x', at: u }
      if (e.panel === 'right') return { from: 'y', at: u }
      const level = inverse(g, v)
      const above = Math.abs(g(u) - v) / (y1 - y0)
      const beside = Math.abs(level - u) / (x1 - x0)
      return above <= beside ? { from: 'x', at: u } : { from: 'x', at: level }
    },
    [g, x0, x1, y0, y1],
  )
  const onPointer = useCallback(
    (e: PanelPointer) => {
      if (e.type === 'leave') return setHover(null)
      const next = locate(e)
      if (e.type === 'move') return setHover(next)
      setPinned((p) => (p ? null : next))
      setHover(next)
    },
    [locate],
  )
  useEffect(() => {
    if (!pinned) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setPinned(null)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [pinned])

  const handles = useMemo(
    (): Handle[] | undefined =>
      id === 'custom'
        ? knots.map((k, i) => ({
            kind: 'point',
            at: k,
            onDrag: ([x, y]) =>
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
          }))
        : undefined,
    [id, knots, y0, y1],
  )

  const param =
    id === 'affine' ? (
      <ParamSlider label="slope a" param={slope} />
    ) : id === 'ramp' ? (
      <ParamSlider label="step height c" param={ramp} />
    ) : id === 'custom' ? (
      <ParamButton onClick={() => setKnots(KNOTS)}>Reset knots</ParamButton>
    ) : null

  return (
    <Interactive
      title="Stretching space thins out density"
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
        <>
          <ParamChoice
            label="map g"
            value={id}
            onChange={setId}
            options={(Object.keys(LABELS) as MapId[]).map((k) => ({ value: k, label: LABELS[k] }))}
          />
          <ParamSlider label="source mean μ" param={mu} />
          <ParamSlider label="source standard deviation σ" param={sigma} />
          {param}
          <ParamChoice
            label="samples"
            value={size}
            onChange={setSize}
            options={SAMPLE_SIZES.map((v) => ({ value: v, label: Number(v).toLocaleString('en') }))}
          />
          <ParamSlider label="seed" param={seed} withArrows format={(v) => String(v)} />
          <ParamSwitch label="Jacobian factor |dg⁻¹/dy|" checked={jacobian} onChange={setJacobian} />
          <ParamSwitch label="show sample paths" checked={paths} onChange={setPaths} />
        </>
      }
      readout={
        <>
          <Readout label="variance σ²" value={formatNumber(s * s)} />
          <Readout label="stretch Δy/Δx" value={formatNumber((yb - ya) / (b - a))} />
          <Readout label="P(X in x-band)" value={formatNumber(massX)} />
          <Readout label={jacobian ? '∫ p_Y over y-band' : '∫ p_X(g⁻¹(y)) over y-band'} value={formatNumber(massY)} />
          <Readout
            label={jacobian ? '∫ p_Y dy' : '∫ p_X(g⁻¹(y)) dy'}
            value={jacobian ? '1.00' : `${curves.naiveMass.toFixed(2)}, not 1`}
          />
          {pinned && <Readout label="band" value="pinned" />}
        </>
      }
    >
      <MarginalPanels
        marks={marks}
        overlay={overlay}
        xRange={X_RANGE}
        yRange={yRange}
        bottomMax={bottomMax}
        rightMax={rightMax}
        xLabel="x"
        yLabel="y = g(x)"
        bottomLabel="p_X(x)"
        rightLabel="density of y"
        handles={handles}
        onPointer={onPointer}
        height={540}
        ariaLabel="The map y = g(x), with the source density of x below it and the density of y = g(x) to its right"
      />
    </Interactive>
  )
}
