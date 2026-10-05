import { useMemo, type ReactNode } from 'react'
import { tField } from './t-field'
import {
  Figure,
  formatNumber,
  Handle,
  type FigureState,
  type NumberDef,
  type Param,
  Plot,
  Readout,
  type Segment,
  Segments,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
  type Vec2,
  Vectors,
} from 'aifn-render'
import { contour, contours, quantileLevels, sampleGrid } from './contours'
import { linspace, toFlat } from 'aifn/foundation/tensor'

/**
 * One objective f(x, y) and one constraint curve g(x, y) = 0 in the plane. The curve is given by a parametrisation
 * t ↦ (x, y) so that a point can be dragged along it and f can be plotted against t.
 */
export type ConstrainedProblem = {
  f: (x: number, y: number) => number
  gradF: (x: number, y: number) => Vec2
  gradG: (x: number, y: number) => Vec2
  curve: (t: number) => Vec2
  tRange: [number, number]
  /** The curve returns to its start at the end of tRange (a circle), so stationary points may sit at the seam. */
  closed?: boolean
  xRange: [number, number]
  yRange: [number, number]
  /** Contour levels of f; defaults to quantiles of f over the plotted window. */
  levels?: number[]
  /** Which stationary point is the optimum to emphasise. */
  goal: 'min' | 'max'
}

export type Stationary = { t: number; point: Vec2; value: number; kind: 'min' | 'max' }

/** The state of the dragged point, for readouts that a particular example adds. */
export type PointState = {
  t: number
  point: Vec2
  value: number
  lambda: number
  gradF: Vec2
  gradG: Vec2
  stationary: Stationary[]
}

const SAMPLES = 2001
const dot = (a: Vec2, b: Vec2) => a[0] * b[0] + a[1] * b[1]
const norm = (a: Vec2) => Math.hypot(a[0], a[1])

/** Everything that depends on the problem only: samples of the curve, f along it, stationary points and contours. */
function analyse(problem: ConstrainedProblem) {
  const { f, curve, tRange, closed, xRange, yRange } = problem
  const ts = toFlat(linspace(tRange[0], tRange[1], SAMPLES))
  const points = ts.map(curve)
  const values = points.map(([x, y]) => f(x, y))
  const stationary: Stationary[] = []
  const finite = values.filter(Number.isFinite)
  const spread = Math.max(...finite) - Math.min(...finite)
  // A flat profile (every point stationary, e.g. A = cI in the Rayleigh quotient) has no isolated stationary points.
  if (spread > 1e-9 * (1 + Math.abs(Math.max(...finite)))) {
    const n = closed ? SAMPLES - 1 : SAMPLES
    const at = (i: number) => values[closed ? (i + n) % n : i]
    const h = ts[1] - ts[0]
    // Parabolic interpolation through three samples places the stationary point between them.
    const refine = (i: number, kind: 'min' | 'max'): Stationary => {
      const curvature = at(i + 1) - 2 * at(i) + at(i - 1)
      const shift = Math.abs(curvature) > 0 ? (-0.5 * (at(i + 1) - at(i - 1))) / curvature : 0
      const raw = ts[i] + Math.max(-1, Math.min(1, shift)) * h
      const period = tRange[1] - tRange[0]
      const t = closed ? tRange[0] + ((((raw - tRange[0]) % period) + period) % period) : raw
      const point = curve(t)
      return { t, point, value: f(...point), kind }
    }
    for (let i = closed ? 0 : 1; i < (closed ? n : n - 1); i++) {
      const left = at(i) - at(i - 1)
      const right = at(i + 1) - at(i)
      if (left < 0 && right >= 0) stationary.push(refine(i, 'min'))
      else if (left > 0 && right <= 0) stationary.push(refine(i, 'max'))
    }
  }
  // The optimum, and every other stationary point with the same value (symmetric problems have several).
  const target =
    problem.goal === 'min' ? Math.min(...stationary.map((s) => s.value)) : Math.max(...stationary.map((s) => s.value))
  const tolerance = 1e-4 * (1 + spread)
  const optima = stationary.filter((s) => Math.abs(s.value - target) <= tolerance)
  const best = optima[0] ?? null
  const grid = sampleGrid(f, xRange, yRange)
  const background = contours(grid, problem.levels ?? quantileLevels(grid))
  const step = Math.max(1, Math.floor(SAMPLES / 500))
  const drawn = points.filter((_, i) => i % step === 0 || i === SAMPLES - 1)
  return {
    ts,
    points,
    values,
    stationary,
    best,
    optima,
    grid,
    background,
    curveX: drawn.map((p) => p[0]),
    curveY: drawn.map((p) => p[1]),
  }
}

/** The sample of the curve nearest to a point, for dragging along the curve. */
function nearestT(ts: number[], points: Vec2[], [x, y]: Vec2): number {
  let best = 0
  let bestD = Infinity
  points.forEach(([px, py], i) => {
    const d = (px - x) ** 2 + (py - y) ** 2
    if (d < bestD) {
      bestD = d
      best = i
    }
  })
  return ts[best]
}

export type ConstrainedExplorerProps = {
  problem: ConstrainedProblem
  title: string
  caption: ReactNode
  /** The opening t and its step, for the explorer's own state (unused when `state` is given). */
  initialT?: number
  tStep?: number
  xLabel: string
  yLabel: string
  tLabel: string
  /** The parameter's symbol in readouts, e.g. θ. */
  tSymbol: string
  fLabel: string
  formatT?: (t: number) => string
  /**
   * The figure's state when the caller adds its own fields (the parameters that reshape the problem): it must hold
   * `t: tField(…)`. Omitted, the explorer keeps a state with `t` alone.
   */
  state?: FigureState<{ t: NumberDef }>
  /** Extra readouts computed from the dragged point. */
  readout?: (s: PointState) => ReactNode
  /** Fixed reference points on the main chart, e.g. an unconstrained optimum. */
  markers?: { name: string; points: Vec2[] }
  /** An extra chart under the profile, e.g. the optimal value against the constraint level. */
  extra?: ReactNode
  profileHeight?: number
}

/**
 * The standard picture of a problem with one constraint in the plane: contours of f, the constraint curve, a point
 * dragged along it with the direction of ∇f against the normal line of the constraint (the direction of ∇g), and a
 * profile of f along the curve. At a stationary point the arrow lies on the normal line and the profile is flat.
 */
export function ConstrainedExplorer({
  problem,
  title,
  caption,
  initialT,
  tStep,
  xLabel,
  yLabel,
  tLabel,
  tSymbol,
  fLabel,
  formatT = formatNumber,
  readout,
  markers,
  extra,
  profileHeight = 320,
  state: shared,
}: ConstrainedExplorerProps) {
  const own = useFigureState({ t: tField(problem.tRange, initialT ?? problem.tRange[0], tStep, tLabel, formatT) })
  const state = shared ?? own
  const a = useMemo(() => analyse(problem), [problem])
  const { f, gradF, gradG, curve, xRange, yRange } = problem
  const [px, py] = curve(state.t)
  const value = f(px, py)
  const gf = gradF(px, py)
  const gg = gradG(px, py)
  const ggNorm = norm(gg)
  const regular = ggNorm > 1e-9
  // Least-squares multiplier: the λ that makes λ∇g closest to ∇f. It is exact at a stationary point.
  const lambda = regular ? dot(gf, gg) / ggNorm ** 2 : NaN
  const residual: Vec2 = [gf[0] - lambda * gg[0], gf[1] - lambda * gg[1]]
  const cosAngle = regular && norm(gf) > 1e-12 ? dot(gf, gg) / (norm(gf) * ggNorm) : NaN
  const angle = (Math.acos(Math.max(-1, Math.min(1, cosAngle))) * 180) / Math.PI
  const h = (problem.tRange[1] - problem.tRange[0]) * 1e-5
  const slope = (f(...curve(state.t + h)) - f(...curve(state.t - h))) / (2 * h)

  const series = useMemo((): SeriesSpec[] => {
    const level = contour(a.grid, value)
    const others = a.stationary.filter((s) => !a.optima.includes(s))
    const out: SeriesSpec[] = [
      { name: '__contours', type: 'line', x: a.background.x, y: a.background.y, muted: true },
      { name: 'constraint', type: 'line', x: a.curveX, y: a.curveY, slot: 0 },
      { name: 'level set', type: 'line', x: level.x, y: level.y, slot: 1, dashed: true },
    ]
    if (markers) {
      out.push({
        name: markers.name,
        type: 'scatter',
        x: markers.points.map((p) => p[0]),
        y: markers.points.map((p) => p[1]),
        muted: true,
      })
    }
    if (others.length) {
      out.push({
        name: 'stationary',
        type: 'scatter',
        x: others.map((s) => s.point[0]),
        y: others.map((s) => s.point[1]),
        slot: 2,
      })
    }
    if (a.best) {
      out.push({
        name: problem.goal === 'min' ? 'minimum' : 'maximum',
        type: 'scatter',
        x: a.optima.map((s) => s.point[0]),
        y: a.optima.map((s) => s.point[1]),
        emphasis: true,
      })
    }
    return out
  }, [a, value, markers, problem.goal])

  // Both guides have the same length, a fixed fraction of the window, so only their directions carry information.
  const length = 0.13 * (xRange[1] - xRange[0])
  const gfx = gf[0]
  const gfy = gf[1]
  const ggx = gg[0]
  const ggy = gg[1]
  const vectors = useMemo((): Segment[] => {
    const n = Math.hypot(gfx, gfy)
    return n > 1e-12 ? [{ from: [px, py], to: [px + (length * gfx) / n, py + (length * gfy) / n] }] : []
  }, [px, py, gfx, gfy, length])
  const normal = useMemo((): Segment[] => {
    const n = Math.hypot(ggx, ggy)
    if (n <= 1e-9) return []
    const [ux, uy] = [(length * ggx) / n, (length * ggy) / n]
    return [{ from: [px - 1.2 * ux, py - 1.2 * uy], to: [px + 1.2 * ux, py + 1.2 * uy] }]
  }, [px, py, ggx, ggy, length])

  const profile = useMemo((): SeriesSpec[] => {
    const others = a.stationary.filter((s) => !a.optima.includes(s))
    const out: SeriesSpec[] = [{ name: fLabel, type: 'line', x: a.ts, y: a.values, slot: 0 }]
    if (others.length) {
      out.push({
        name: 'stationary',
        type: 'scatter',
        x: others.map((s) => s.t),
        y: others.map((s) => s.value),
        slot: 2,
      })
    }
    if (a.best) {
      const name = problem.goal === 'min' ? 'minimum' : 'maximum'
      out.push({
        name,
        type: 'scatter',
        x: a.optima.map((s) => s.t),
        y: a.optima.map((s) => s.value),
        emphasis: true,
      })
    }
    out.push({ name: 'point', type: 'scatter', x: [state.t], y: [value], slot: 1 })
    return out
  }, [a, state.t, value, fLabel, problem.goal])

  const xAxis = useAxis({ label: xLabel, range: xRange })
  const yAxis = useAxis({ label: yLabel, range: yRange, equal: xAxis })
  const xAxis2 = useAxis({ label: tLabel, range: problem.tRange })
  const yAxis2 = useAxis({ label: fLabel, hold: 'union' })
  return (
    <Figure
      title={title}
      state={state}
      caption={caption}
      readouts={
        <>
          <Readout label="(x, y)" value={`(${formatNumber(px)}, ${formatNumber(py)})`} />
          <Readout label={fLabel} value={formatNumber(value)} />
          <Readout label={`slope of ${fLabel} in ${tSymbol}`} value={formatNumber(slope)} />
          {regular ? (
            <>
              <Readout label="angle between ∇f and ∇g" value={`${formatNumber(angle)}°`} />
              <Readout label="λ = ∇f·∇g / ‖∇g‖²" value={formatNumber(lambda)} />
              <Readout label="‖∇f − λ∇g‖" value={formatNumber(norm(residual))} />
            </>
          ) : (
            <Readout label="∇g" value="0: no normal, no multiplier" />
          )}
          {a.best && (
            <Readout
              label={problem.goal === 'min' ? 'minimum' : 'maximum'}
              value={`${formatNumber(a.best.value)} at ${tSymbol} = ${formatT(a.best.t)}`}
            />
          )}
          {readout?.({ t: state.t, point: [px, py], value, lambda, gradF: gf, gradG: gg, stationary: a.stationary })}
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers(series)}
          <Segments segments={normal} />
          <Vectors vectors={vectors} />
          <Handle
            kind="point"
            at={[px, py]}
            label="point on the constraint"
            onDrag={(p) => state.set('t', nearestT(a.ts, a.points, p))}
          />
        </Plot>
        <div className="flex flex-col gap-4">
          <Plot x={xAxis2} y={yAxis2} height={profileHeight}>
            {seriesLayers(profile)}
            <Handle {...state.handle('t', { label: tSymbol })} />
          </Plot>
          {extra}
        </div>
      </div>
    </Figure>
  )
}

export type SensitivityPanelProps = {
  /** The constraint level c, shared with the slider that sets it. */
  param: Param
  /** Optimal value f*(c). Must be stable (module level), because the curve is computed once. */
  optimum: (c: number) => number
  /** The multiplier λ(c), which is the slope d(f*)/dc. */
  multiplier: (c: number) => number
  xLabel: string
  /** Short name of the level, for the handle's label. */
  symbol: string
  yLabel: string
  height?: number
}

/**
 * The optimal value against the constraint level, with the tangent line of slope λ at the current level. The tangent
 * touching the curve is the statement d(f*)/dc = λ. The level is draggable along the axis.
 */
export function SensitivityPanel({
  param,
  optimum,
  multiplier,
  xLabel,
  symbol,
  yLabel,
  height = 220,
}: SensitivityPanelProps) {
  const curve = useMemo(() => {
    const x = toFlat(linspace(param.min, param.max, 201))
    return { x, y: x.map(optimum) }
  }, [param.min, param.max, optimum])
  const c = param.value
  const series = useMemo((): SeriesSpec[] => {
    const f0 = optimum(c)
    const slope = multiplier(c)
    const half = 0.18 * (param.max - param.min)
    return [
      { name: yLabel, type: 'line', x: curve.x, y: curve.y, slot: 0 },
      {
        name: 'slope λ',
        type: 'line',
        x: [c - half, c + half],
        y: [f0 - slope * half, f0 + slope * half],
        slot: 1,
        dashed: true,
      },
      { name: 'level', type: 'scatter', x: [c], y: [f0], slot: 1 },
    ]
  }, [curve, c, optimum, multiplier, param.min, param.max, yLabel])
  const xAxis = useAxis({ label: xLabel, range: [param.min, param.max] })
  const yAxis = useAxis({ label: yLabel, hold: 'union' })
  return (
    <Plot x={xAxis} y={yAxis} height={height}>
      {seriesLayers(series)}
      <Handle kind="x" at={c} label={symbol} onDrag={param.set} />
    </Plot>
  )
}
