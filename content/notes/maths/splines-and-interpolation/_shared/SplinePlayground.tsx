import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import {
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type Segment,
  type XYSeries,
} from '@/components/viz'
import {
  bendingEnergy,
  buildCurve,
  elevateBezier,
  insertKnot,
  interiorKnots,
  isFunctionType,
  openUniformKnots,
  sampleCurve,
  uniformKnots,
  unitNormal,
  type EndCondition,
  type SplineType,
  type Vec2,
} from './splines'

const LABELS: Record<SplineType, string> = {
  linear: 'Piecewise linear',
  polynomial: 'Interpolating polynomial',
  cubic: 'Cubic spline',
  pchip: 'PCHIP (monotone cubic)',
  akima: 'Akima',
  smoothing: 'Smoothing spline',
  'catmull-rom': 'Catmull–Rom / cardinal',
  bspline: 'B-spline',
  bezier: 'Bézier',
  nurbs: 'NURBS',
}
const ALL_TYPES = Object.keys(LABELS) as SplineType[]

const DEFAULT_POINTS: Vec2[] = [
  [1, 1.5],
  [2.5, 4.2],
  [4.2, 4.6],
  [5.8, 2],
  [7.5, 1.8],
  [9, 4.5],
]

type Preset = 'initial' | 'runge' | 'runge-chebyshev' | 'step' | 'uneven' | 'circle'
const PRESET_LABELS: Record<Preset, string> = {
  initial: 'Starting points',
  runge: 'Runge function, equispaced',
  'runge-chebyshev': 'Runge function, Chebyshev x',
  step: 'Step data (monotone)',
  uneven: 'Unevenly spaced points',
  circle: 'Circle (NURBS)',
}

/** 11 samples of 1 + 4 / (1 + 25 s^2) for s in [-1, 1], mapped to x in [0.5, 9.5]. */
function rungePoints(chebyshev: boolean): Vec2[] {
  const n = 11
  return Array.from({ length: n }, (_, k) => {
    const s = chebyshev ? -Math.cos(((2 * k + 1) * Math.PI) / (2 * n)) : -1 + (2 * k) / (n - 1)
    return [5 + 4.5 * s, 1 + 4 / (1 + 25 * s * s)] as Vec2
  })
}

const R2 = Math.SQRT1_2
const CIRCLE = {
  points: [
    [7.5, 3],
    [7.5, 5.5],
    [5, 5.5],
    [2.5, 5.5],
    [2.5, 3],
    [2.5, 0.5],
    [5, 0.5],
    [7.5, 0.5],
    [7.5, 3],
  ] as Vec2[],
  weights: [1, R2, 1, R2, 1, R2, 1, R2, 1],
  knots: [0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1],
}

const PRESET_POINTS: Record<Exclude<Preset, 'initial' | 'circle'>, Vec2[]> = {
  runge: rungePoints(false),
  'runge-chebyshev': rungePoints(true),
  step: [
    [0.8, 1],
    [2, 1.1],
    [3.2, 1.2],
    [4.2, 1.25],
    [5, 4.6],
    [6.2, 4.7],
    [7.6, 4.75],
    [9.2, 4.8],
  ],
  uneven: [
    [1, 1],
    [1.6, 4.8],
    [2.1, 5],
    [6.5, 5.2],
    [7, 1.2],
    [9.2, 1.4],
  ],
}

type Alpha = '0' | '0.5' | '1'
type Lower = 'derivatives' | 'curvature'

export type SplinePlaygroundProps = {
  title?: string
  caption?: ReactNode
  initialType?: SplineType
  /** Spline types offered in the menu. Defaults to all of them. */
  types?: SplineType[]
  initialPoints?: Vec2[]
  initialWeights?: number[]
  /** Full knot vector for B-spline and NURBS types; must have points + degree + 1 entries. */
  initialKnots?: number[]
  initialDegree?: number
  initialKnotMode?: 'open' | 'uniform'
  initialEnd?: EndCondition
  initialAlpha?: 0 | 0.5 | 1
  initialTension?: number
  initialLogLambda?: number
  initialT?: number
  /** Presets offered in the menu. */
  presets?: Preset[]
  /** A second curve drawn dashed for comparison, e.g. a natural cubic spline under PCHIP. */
  overlay?: SplineType
  showPolygon?: boolean
  showComb?: boolean
  showConstruction?: boolean
  showOsculating?: boolean
  showBasis?: boolean
  showLower?: boolean
  initialLower?: Lower
  xRange?: [number, number]
  yRange?: [number, number]
}

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi)
const MIN_POINTS = 3
const MAX_POINTS = 16

/**
 * A spline playground: draggable points, every common spline type, the basis functions below the curve, and
 * curvature, continuity and overshoot readouts. The maths lives in splines.ts.
 */
export function SplinePlayground({
  title = 'Spline playground',
  caption,
  initialType = 'cubic',
  types = ALL_TYPES,
  initialPoints = DEFAULT_POINTS,
  initialWeights,
  initialKnots,
  initialDegree = 3,
  initialKnotMode = 'open',
  initialEnd = 'natural',
  initialAlpha = 0.5,
  initialTension = 0,
  initialLogLambda = -1,
  initialT = 0.35,
  presets = ['initial', 'runge', 'runge-chebyshev', 'step', 'uneven', 'circle'],
  overlay,
  showPolygon = true,
  showComb = false,
  showConstruction = false,
  showOsculating = false,
  showBasis = true,
  showLower = false,
  initialLower = 'derivatives',
  xRange = [0, 10],
  yRange = [0, 6],
}: SplinePlaygroundProps) {
  const [type, setType] = useState<SplineType>(initialType)
  const [points, setPoints] = useState<Vec2[]>(initialPoints)
  const [weights, setWeights] = useState<number[]>(initialWeights ?? initialPoints.map(() => 1))
  const [selected, setSelected] = useState(0)
  const [end, setEnd] = useState<EndCondition>(initialEnd)
  const [alpha, setAlpha] = useState<Alpha>(String(initialAlpha) as Alpha)
  const tension = useParam(initialTension, { min: 0, max: 1, step: 0.05 })
  const degree = useParam(initialDegree, { min: 1, max: 5, step: 1 })
  const [knotMode, setKnotMode] = useState<'open' | 'uniform'>(initialKnotMode)
  const sig = (n: number, p: number, mode: string) => `${n}|${p}|${mode}`
  const [custom, setCustom] = useState<{ sig: string; knots: number[] } | null>(
    initialKnots
      ? {
          sig: sig(initialPoints.length, Math.min(initialDegree, initialPoints.length - 1), initialKnotMode),
          knots: initialKnots,
        }
      : null,
  )
  const [selectedKnot, setSelectedKnot] = useState<number | null>(null)
  const logLambda = useParam(initialLogLambda, { min: -4, max: 3, step: 0.05 })
  const t = useParam(initialT, { min: 0, max: 1, step: 0.005 })
  const [polygon, setPolygon] = useState(showPolygon)
  const [comb, setComb] = useState(showComb)
  const [construction, setConstruction] = useState(showConstruction)
  const [osculating, setOsculating] = useState(showOsculating)
  const [lowerOn, setLowerOn] = useState(showLower)
  const [lower, setLower] = useState<Lower>(initialLower)
  const [overlayOn, setOverlayOn] = useState(Boolean(overlay))
  const [preset, setPreset] = useState<Preset>('initial')

  // Shift-click removes a point. The chart reports clicks without modifier keys, so track Shift here.
  const shift = useRef(false)
  useEffect(() => {
    const track = (e: KeyboardEvent) => {
      shift.current = e.shiftKey
    }
    window.addEventListener('keydown', track)
    window.addEventListener('keyup', track)
    return () => {
      window.removeEventListener('keydown', track)
      window.removeEventListener('keyup', track)
    }
  }, [])

  const n = points.length
  const parametric = !isFunctionType(type)
  const usesKnots = type === 'bspline' || type === 'nurbs'
  const p = Math.min(degree.value, n - 1)
  const knotSig = sig(n, p, knotMode)
  const knots = useMemo(
    () => (custom?.sig === knotSig ? custom.knots : knotMode === 'open' ? openUniformKnots(n, p) : uniformKnots(n, p)),
    [custom, knotSig, knotMode, n, p],
  )

  const model = useMemo(
    () =>
      buildCurve({
        type,
        points,
        weights,
        end,
        alpha: Number(alpha),
        tension: tension.value,
        degree: p,
        knots,
        lambda: 10 ** logLambda.value,
      }),
    [type, points, weights, end, alpha, tension.value, p, knots, logLambda.value],
  )
  const overlayModel = useMemo(
    () =>
      overlay && overlayOn && overlay !== type
        ? buildCurve({
            type: overlay,
            points,
            weights,
            end: 'natural',
            alpha: 0.5,
            tension: 0,
            degree: p,
            knots,
            lambda: 10 ** logLambda.value,
          })
        : null,
    [overlay, overlayOn, type, points, weights, p, knots, logLambda.value],
  )
  const [lo, hi] = model.domain
  const u = lo + t.value * (hi - lo)

  const samples = useMemo(() => sampleCurve(model, 400), [model])
  const stats = useMemo(() => {
    const maxKappa = Math.max(...samples.map((s) => Math.abs(s.kappa)))
    const energy = bendingEnergy(samples)
    let overshoot: number | null = null
    if (!parametric && model.interpolates) {
      const ys = points.map((q) => q[1])
      const yMax = Math.max(...ys)
      const yMin = Math.min(...ys)
      const fs = samples.map((s) => s.e.p[1])
      const over = Math.max(0, Math.max(...fs) - yMax, yMin - Math.min(...fs))
      overshoot = yMax > yMin ? over / (yMax - yMin) : 0
    }
    return { maxKappa, energy, overshoot }
  }, [samples, parametric, model.interpolates, points])

  const current = model.at(u)
  const kappaAtT = (() => {
    const s = Math.hypot(...current.d1)
    return s < 1e-12 ? 0 : (current.d1[0] * current.d2[1] - current.d1[1] * current.d2[0]) / s ** 3
  })()

  const curveSeries = useMemo(() => {
    const series: XYSeries[] = []
    const segments: Segment[] = []
    if (polygon && parametric) {
      series.push({
        name: 'control polygon',
        type: 'line',
        x: points.map((q) => q[0]),
        y: points.map((q) => q[1]),
        muted: true,
        dashed: true,
      })
    }
    if (overlayModel && overlay) {
      const os = sampleCurve(overlayModel, 300)
      series.push({
        name: LABELS[overlay],
        type: 'line',
        x: os.map((s) => s.e.p[0]),
        y: os.map((s) => s.e.p[1]),
        slot: 1,
        dashed: true,
      })
    }
    series.push({
      name: LABELS[type],
      type: 'line',
      x: samples.map((s) => s.e.p[0]),
      y: samples.map((s) => s.e.p[1]),
      slot: 0,
    })
    if (comb) {
      const scale = stats.maxKappa > 1e-9 ? 1.2 / stats.maxKappa : 0
      const tips: Vec2[] = []
      samples.forEach((s, i) => {
        const nrm = unitNormal(s.e.d1)
        const tip: Vec2 = [s.e.p[0] - scale * s.kappa * nrm[0], s.e.p[1] - scale * s.kappa * nrm[1]]
        tips.push(tip)
        if (i % 4 === 0) segments.push({ from: s.e.p, to: tip })
      })
      series.push({
        name: 'curvature comb',
        type: 'line',
        x: tips.map((q) => q[0]),
        y: tips.map((q) => q[1]),
        muted: true,
      })
    }
    if (usesKnots && model.joins.length) {
      const at = model.joins.map((k) => model.at(k).p)
      series.push({ name: 'knots', type: 'scatter', x: at.map((q) => q[0]), y: at.map((q) => q[1]), slot: 2 })
    }
    return { series, segments }
  }, [polygon, parametric, points, overlayModel, overlay, type, samples, comb, stats.maxKappa, usesKnots, model])

  // Fast-changing layers (at the parameter t) are built every render; they are small.
  const tSeries: XYSeries[] = []
  if (construction && model.construction) {
    const levels = model.construction(u)
    levels.slice(1).forEach((level) => {
      if (level.length > 1)
        tSeries.push({
          name: 'construction',
          type: 'line',
          x: level.map((q) => q[0]),
          y: level.map((q) => q[1]),
          slot: 3,
        })
    })
    const pts = levels.slice(1, -1).flat()
    if (pts.length)
      tSeries.push({ name: 'construction', type: 'scatter', x: pts.map((q) => q[0]), y: pts.map((q) => q[1]), slot: 3 })
  }
  if (osculating && Math.abs(kappaAtT) > 1e-3) {
    const nrm = unitNormal(current.d1)
    const r = 1 / kappaAtT
    const c: Vec2 = [current.p[0] + r * nrm[0], current.p[1] + r * nrm[1]]
    const ring = Array.from({ length: 181 }, (_, i) => (2 * Math.PI * i) / 180)
    tSeries.push({
      name: 'osculating circle',
      type: 'line',
      x: ring.map((a) => c[0] + Math.abs(r) * Math.cos(a)),
      y: ring.map((a) => c[1] + Math.abs(r) * Math.sin(a)),
      muted: true,
      dashed: true,
    })
  }
  if (points[selected]) {
    const ring = Array.from({ length: 41 }, (_, i) => (2 * Math.PI * i) / 40)
    const [sx, sy] = points[selected]
    tSeries.push({
      name: 'selected point',
      type: 'line',
      x: ring.map((a) => sx + 0.28 * Math.cos(a)),
      y: ring.map((a) => sy + 0.28 * Math.sin(a)),
      emphasis: true,
    })
  }
  tSeries.push({ name: 'C(t)', type: 'scatter', x: [current.p[0]], y: [current.p[1]], emphasis: true })

  const allSeries = [...curveSeries.series, ...tSeries]

  const clampPoint = (q: Vec2): Vec2 => [clamp(q[0], xRange[0], xRange[1]), clamp(q[1], yRange[0], yRange[1])]
  const pointHandles: Handle[] = points.map((q, i) => ({
    kind: 'point',
    at: q,
    onDrag: (next) => {
      setSelected(i)
      setPreset('initial')
      setPoints((prev) => prev.map((old, j) => (j === i ? clampPoint(next) : old)))
    },
  }))

  const near = (q: Vec2) => {
    const tol = 0.04 * (xRange[1] - xRange[0])
    let best = -1
    let bestD = tol
    points.forEach((r, i) => {
      const d = Math.hypot(r[0] - q[0], r[1] - q[1])
      if (d < bestD) {
        best = i
        bestD = d
      }
    })
    return best
  }
  const removePoint = (i: number) => {
    if (n <= MIN_POINTS) return
    setPoints((prev) => prev.filter((_, j) => j !== i))
    setWeights((prev) => prev.filter((_, j) => j !== i))
    setSelected((s) => Math.max(0, Math.min(s > i ? s - 1 : s, n - 2)))
  }
  const onPlotClick = (q: [number, number]) => {
    const hit = near(q)
    if (hit >= 0) {
      if (shift.current) removePoint(hit)
      return
    }
    if (n >= MAX_POINTS) return
    // Parametric curves: insert where the control polygon grows least. Function types sort by x anyway.
    let at = n
    if (parametric) {
      const d = (a: Vec2, b: Vec2) => Math.hypot(a[0] - b[0], a[1] - b[1])
      let best = d(points[n - 1], q)
      if (d(points[0], q) < best) {
        best = d(points[0], q)
        at = 0
      }
      for (let i = 0; i < n - 1; i++) {
        const cost = d(points[i], q) + d(q, points[i + 1]) - d(points[i], points[i + 1])
        if (cost < best) {
          best = cost
          at = i + 1
        }
      }
    }
    setPoints((prev) => [...prev.slice(0, at), q, ...prev.slice(at)])
    setWeights((prev) => [...prev.slice(0, at), 1, ...prev.slice(at)])
    setSelected(at)
  }

  // Knots: interior ones (strictly inside the domain) can be dragged on the basis chart.
  const knotIndices = usesKnots ? knots.map((_, j) => j).filter((j) => j > p && j < knots.length - 1 - p) : []
  const setKnots = (next: number[]) => setCustom({ sig: knotSig, knots: next })
  const knotHandles: Handle[] = knotIndices.map((j) => ({
    kind: 'x',
    at: knots[j],
    onDrag: (v) => {
      const next = [...knots]
      next[j] = clamp(v, Math.max(knots[j - 1], lo + 1e-3), Math.min(knots[j + 1], hi - 1e-3))
      setSelectedKnot(j)
      setKnots(next)
    },
  }))
  const activeKnot =
    selectedKnot !== null && knotIndices.includes(selectedKnot)
      ? selectedKnot
      : knotIndices[Math.floor(knotIndices.length / 2)]
  const multiplicity = activeKnot === undefined ? 1 : knotIndices.filter((j) => knots[j] === knots[activeKnot]).length
  const setMultiplicity = (m: number) => {
    if (activeKnot === undefined) return
    const v = knots[activeKnot]
    const group = knotIndices.filter((j) => knots[j] === v)
    const last = knotIndices[knotIndices.length - 1]
    let start = group[0]
    if (start + m - 1 > last) start = last - m + 1
    const next = [...knots]
    const inGroup = (j: number) => j >= start && j < start + m
    knotIndices.forEach((j) => {
      if (inGroup(j)) next[j] = v
    })
    // Knots that left the group spread evenly towards the next distinct knot, keeping the vector non-decreasing.
    const freed = group.filter((j) => !inGroup(j))
    if (freed.length) {
      const after = knots.find((k, j) => j > group[group.length - 1] && k > v) ?? hi
      freed.forEach((j, k) => {
        next[j] = v + ((after - v) * (k + 1)) / (freed.length + 1)
      })
    }
    for (let j = 1; j < next.length; j++) next[j] = Math.max(next[j], next[j - 1])
    setSelectedKnot(start)
    setKnots(next)
  }

  const tHandle: Handle = { kind: 'x', at: u, label: 't', onDrag: (v) => t.set(hi > lo ? (v - lo) / (hi - lo) : 0) }

  const basisChart = useMemo(() => {
    if (!model.basis) return null
    const us = Array.from({ length: 241 }, (_, i) => lo + ((hi - lo) * i) / 240)
    const rows = us.map((v) => model.basis!(v))
    const count = rows[0].length
    const sel = model.order.indexOf(selected)
    const series: XYSeries[] = []
    for (let j = 0; j < count; j++) {
      if (j === sel) continue
      series.push({ name: 'basis functions', type: 'line', x: us, y: rows.map((r) => r[j]), muted: true })
    }
    if (sel >= 0)
      series.push({
        name: 'basis function of the selected point',
        type: 'line',
        x: us,
        y: rows.map((r) => r[sel]),
        slot: 0,
      })
    const all = rows.flat()
    const yMin = Math.max(-1.5, Math.min(-0.1, ...all))
    const yMax = Math.min(2.5, Math.max(1.05, ...all))
    return { series, yRange: [yMin, yMax] as [number, number] }
  }, [model, lo, hi, selected])

  const basisAtT = model.basis ? model.basis(u) : []
  const basisDots: XYSeries = {
    name: 'values at t',
    type: 'scatter',
    x: basisAtT.flatMap((v) => (Math.abs(v) > 1e-9 ? [u] : [])),
    y: basisAtT.filter((v) => Math.abs(v) > 1e-9),
    emphasis: true,
  }
  const basisSeries = basisChart ? [...basisChart.series, basisDots] : []

  const lowerSeries = useMemo(() => {
    const us = samples.map((s) => s.u)
    if (lower === 'curvature')
      return [{ name: 'curvature κ', type: 'line', x: us, y: samples.map((s) => s.kappa), slot: 0 }] as XYSeries[]
    if (!parametric)
      return [
        { name: "f'(x)", type: 'line', x: us, y: samples.map((s) => s.e.d1[1]), slot: 0 },
        { name: "f''(x)", type: 'line', x: us, y: samples.map((s) => s.e.d2[1]), slot: 1 },
      ] as XYSeries[]
    return [
      { name: "x'(u)", type: 'line', x: us, y: samples.map((s) => s.e.d1[0]), slot: 0 },
      { name: "y'(u)", type: 'line', x: us, y: samples.map((s) => s.e.d1[1]), slot: 1 },
      { name: "x''(u)", type: 'line', x: us, y: samples.map((s) => s.e.d2[0]), slot: 2 },
      { name: "y''(u)", type: 'line', x: us, y: samples.map((s) => s.e.d2[1]), slot: 3 },
    ] as XYSeries[]
  }, [samples, lower, parametric])

  const loadPreset = (pr: Preset) => {
    setPreset(pr)
    setSelected(0)
    setCustom(null)
    if (pr === 'circle') {
      setType('nurbs')
      degree.set(2)
      setKnotMode('open')
      setPoints(CIRCLE.points)
      setWeights(CIRCLE.weights)
      setCustom({ sig: sig(CIRCLE.points.length, 2, 'open'), knots: CIRCLE.knots })
      return
    }
    const pts = pr === 'initial' ? initialPoints : PRESET_POINTS[pr]
    setPoints(pts)
    setWeights(pr === 'initial' && initialWeights ? initialWeights : pts.map(() => 1))
    if (pr === 'initial' && initialKnots)
      setCustom({ sig: sig(pts.length, Math.min(initialDegree, pts.length - 1), initialKnotMode), knots: initialKnots })
  }

  const offered = types.filter((ty) => ALL_TYPES.includes(ty))
  const continuity =
    model.continuity === Infinity
      ? 'C∞'
      : model.continuity < 0
        ? 'discontinuous'
        : `C${String(model.continuity).replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(d)])}`

  const controls = (
    <>
      {offered.length > 1 && (
        <div className="flex flex-col gap-2">
          <span className="text-xs text-muted-foreground">Spline type</span>
          <Select
            items={Object.fromEntries(offered.map((ty) => [ty, LABELS[ty]]))}
            value={type}
            onValueChange={(v) => v && setType(v as SplineType)}
          >
            <SelectTrigger size="sm" className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {offered.map((ty) => (
                <SelectItem key={ty} value={ty}>
                  {LABELS[ty]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}
      {presets.length > 1 && (
        <div className="flex flex-col gap-2">
          <span className="text-xs text-muted-foreground">Load points</span>
          <Select
            items={Object.fromEntries(presets.map((pr) => [pr, PRESET_LABELS[pr]]))}
            value={preset}
            onValueChange={(v) => v && loadPreset(v as Preset)}
          >
            <SelectTrigger size="sm" className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {presets.map((pr) => (
                <SelectItem key={pr} value={pr}>
                  {PRESET_LABELS[pr]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}
      {type === 'cubic' && (
        <div className="sm:col-span-2">
          <ParamChoice
            label="End condition"
            value={end}
            onChange={setEnd}
            options={[
              { value: 'natural', label: 'natural' },
              { value: 'clamped', label: 'clamped' },
              { value: 'not-a-knot', label: 'not-a-knot' },
              { value: 'periodic', label: 'periodic' },
            ]}
          />
        </div>
      )}
      {type === 'catmull-rom' && (
        <>
          <ParamChoice
            label="Parameterisation (α)"
            value={alpha}
            onChange={setAlpha}
            options={[
              { value: '0', label: 'uniform' },
              { value: '0.5', label: 'centripetal' },
              { value: '1', label: 'chordal' },
            ]}
          />
          <ParamSlider label="Tension τ" param={tension} />
        </>
      )}
      {usesKnots && (
        <>
          <ParamSlider label="Degree p" param={degree} withArrows format={(v) => String(Math.min(v, n - 1))} />
          <ParamChoice
            label="Knots"
            value={knotMode}
            onChange={(v) => {
              setKnotMode(v)
              setCustom(null)
            }}
            options={[
              { value: 'open', label: 'open uniform' },
              { value: 'uniform', label: 'uniform' },
            ]}
          />
          {activeKnot !== undefined && p > 1 && (
            <ParamSlider
              label={`Multiplicity of the knot at ${formatNumber(knots[activeKnot])}`}
              value={multiplicity}
              onChange={(m) => setMultiplicity(Math.round(m))}
              min={1}
              max={Math.min(p, knotIndices.length)}
              step={1}
              withArrows
            />
          )}
        </>
      )}
      {type === 'nurbs' && (
        <ParamSlider
          label={`Weight of point ${selected + 1}`}
          value={weights[selected] ?? 1}
          onChange={(v) => setWeights((prev) => prev.map((w, i) => (i === selected ? v : w)))}
          min={0.05}
          max={5}
          step={0.05}
        />
      )}
      {type === 'smoothing' && (
        <ParamSlider
          label="log₁₀ λ"
          param={logLambda}
          format={(v) => `${formatNumber(v)} (λ = ${formatNumber(10 ** v)})`}
        />
      )}
      <ParamSlider label={`Parameter t (u = ${formatNumber(u)})`} param={t} withArrows format={(v) => v.toFixed(3)} />
      <div className="flex flex-wrap gap-x-4 gap-y-3">
        {parametric && <ParamSwitch label="control polygon" checked={polygon} onChange={setPolygon} />}
        <ParamSwitch label="curvature comb" checked={comb} onChange={setComb} />
        {model.construction && (
          <ParamSwitch label="construction at t" checked={construction} onChange={setConstruction} />
        )}
        <ParamSwitch label="osculating circle" checked={osculating} onChange={setOsculating} />
        <ParamSwitch label="derivative plots" checked={lowerOn} onChange={setLowerOn} />
        {overlay && overlay !== type && (
          <ParamSwitch
            label={`compare with ${LABELS[overlay].toLowerCase()}`}
            checked={overlayOn}
            onChange={setOverlayOn}
          />
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        {usesKnots && (
          <ParamButton
            disabled={n >= MAX_POINTS}
            onClick={() => {
              const r = insertKnot(points, type === 'nurbs' ? weights : points.map(() => 1), knots, p, u)
              setPoints(r.points)
              setWeights(r.weights)
              setCustom({ sig: sig(r.points.length, p, knotMode), knots: r.knots })
            }}
          >
            Insert knot at t
          </ParamButton>
        )}
        {type === 'bezier' && (
          <ParamButton
            disabled={n >= MAX_POINTS}
            onClick={() => {
              setPoints(elevateBezier(points))
              setWeights((prev) => [...prev, 1])
            }}
          >
            Elevate degree
          </ParamButton>
        )}
        <ParamButton disabled={n <= MIN_POINTS} onClick={() => removePoint(selected)}>
          Remove point {selected + 1}
        </ParamButton>
      </div>
    </>
  )

  const knotText = usesKnots ? knots.map((k) => formatNumber(Number(k.toFixed(3)))).join(', ') : null
  const inner = usesKnots ? interiorKnots(knots, p) : []

  return (
    <Interactive
      title={title}
      caption={
        caption ?? (
          <>
            Drag a point to move it. Click empty space to add a point; Shift-click a point to remove it. Drag the t line
            {usesKnots ? ' or a knot line' : ''} in the lower panel.
          </>
        )
      }
      controls={controls}
      readout={
        <>
          <Readout label="continuity" value={continuity} />
          {usesKnots && inner.length > 0 && (
            <Readout label="knot multiplicities" value={inner.map((k) => k.multiplicity).join(', ')} />
          )}
          <Readout label="basis functions" value={model.basisCount ?? 'none (not linear in the data)'} />
          <Readout label="max |κ|" value={formatNumber(stats.maxKappa)} />
          <Readout label="bending energy ∫κ² ds" value={formatNumber(stats.energy)} />
          {stats.overshoot !== null && (
            <Readout label="overshoot" value={`${formatNumber(100 * stats.overshoot)}% of the data range`} />
          )}
          <Readout label="κ at t" value={formatNumber(kappaAtT)} />
          {type === 'cubic' && end === 'periodic' && (
            <span>Periodic: the last point takes the first point's height.</span>
          )}
          {knotText && <Readout label="knot vector" value={`[${knotText}]`} />}
        </>
      }
    >
      <XYChart
        series={allSeries}
        segments={curveSeries.segments}
        xRange={xRange}
        yRange={[yRange[0], yRange[1]]}
        equalAspect
        handles={pointHandles}
        onPlotClick={onPlotClick}
        ariaLabel={`${LABELS[type]} through ${n} draggable points`}
      />
      {showBasis &&
        (basisChart ? (
          <XYChart
            series={basisSeries}
            xRange={[lo, hi]}
            yRange={basisChart.yRange}
            xLabel={parametric ? 'parameter u' : 'x'}
            yLabel="basis value"
            height={220}
            handles={[tHandle, ...knotHandles]}
            ariaLabel="Basis functions over the parameter, with the current t marked"
          />
        ) : (
          <p className="rounded-md bg-muted p-3 text-xs text-muted-foreground">
            {LABELS[type]} chooses its slopes from the data by a non-linear rule, so the curve is not a fixed linear
            combination of the data values and has no basis functions to plot.
          </p>
        ))}
      {lowerOn && (
        <>
          <ParamChoice
            label="Lower panel"
            value={lower}
            onChange={setLower}
            options={[
              { value: 'derivatives', label: 'first and second derivatives' },
              { value: 'curvature', label: 'curvature' },
            ]}
          />
          <XYChart
            series={lowerSeries}
            xRange={[lo, hi]}
            xLabel={parametric ? 'parameter u' : 'x'}
            height={220}
            handles={[tHandle]}
            ariaLabel="Derivatives or curvature along the curve"
          />
        </>
      )}
    </Interactive>
  )
}
