import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { grid } from '../_shared/geometry'

type Fn = 'sin' | 'cos' | 'tan'
const SPAN = 2 * Math.PI
const TH = grid(-SPAN, SPAN, 1601)
const CAP = 3.2

const SPEC: Record<
  Fn,
  { f: (t: number) => number; inv: (y: number) => number; branch: [number, number]; period: number; name: string }
> = {
  sin: { f: Math.sin, inv: Math.asin, branch: [-Math.PI / 2, Math.PI / 2], period: 2 * Math.PI, name: 'arcsin' },
  cos: { f: Math.cos, inv: Math.acos, branch: [0, Math.PI], period: 2 * Math.PI, name: 'arccos' },
  tan: { f: Math.tan, inv: Math.atan, branch: [-Math.PI / 2, Math.PI / 2], period: Math.PI, name: 'arctan' },
}

/** Every θ in view with f(θ) = y, from the principal value and the function's symmetries. */
function solutions(fn: Fn, y: number): number[] {
  const s = SPEC[fn]
  if (fn !== 'tan' && Math.abs(y) > 1) return []
  const p = s.inv(y)
  // sin θ = y also at π − p; cos θ = y also at −p; tan has one solution per period.
  const seeds = fn === 'sin' ? [p, Math.PI - p] : fn === 'cos' ? [p, -p] : [p]
  const out: number[] = []
  for (const seed of seeds) {
    for (let k = -4; k <= 4; k++) {
      const t = seed + k * s.period
      if (t >= -SPAN - 1e-9 && t <= SPAN + 1e-9 && !out.some((u) => Math.abs(u - t) < 1e-9)) out.push(t)
    }
  }
  return out.sort((a, b) => a - b)
}

/**
 * Solving f(θ) = y for θ. A horizontal line at height y meets the graph of sin, cos or tan infinitely often; the
 * inverse function returns the one solution on the principal branch, drawn bold.
 */
export function PrincipalBranches() {
  const state = useFigureState({
    fn: choice<Fn>(['sin', 'cos', 'tan'], 'sin', { label: 'function' }),
    y: slider(-CAP, CAP, 0.5, { step: 0.01, onChart: true }),
  })
  const { fn, y } = state
  const s = SPEC[fn]

  const curves = useMemo(() => {
    const all = TH.map((t) => {
      const v = s.f(t)
      return Math.abs(v) > CAP + 0.5 ? NaN : v
    })
    const bt = grid(s.branch[0] + 1e-3, s.branch[1] - 1e-3, 401)
    const bv = bt.map((t) => {
      const v = s.f(t)
      return Math.abs(v) > CAP + 0.5 ? NaN : v
    })
    return { all, bt, bv }
  }, [s])

  const roots = useMemo(() => solutions(fn, y), [fn, y])
  const has = fn === 'tan' || Math.abs(y) <= 1
  const principal = has ? s.inv(y) : NaN
  const general =
    fn === 'sin'
      ? 'θ = arcsin y + 2πk  or  π − arcsin y + 2πk'
      : fn === 'cos'
        ? 'θ = ± arccos y + 2πk'
        : 'θ = arctan y + πk'

  const x = useAxis({ label: 'θ (rad)', range: [-SPAN, SPAN] })
  const yAxis = useAxis({ label: `${fn} θ`, range: [-CAP, CAP] })
  return (
    <Figure
      title="Principal branches"
      state={state}
      caption="Drag the horizontal line to a value y. Every marked point on the graph is an angle θ with f(θ) = y; there are infinitely many, a whole family repeating with the period. The bold stretch is the principal branch: [−π/2, π/2] for sine, [0, π] for cosine and (−π/2, π/2) for tangent. On it the function is one-to-one and takes every value in its range once, so the inverse function returns the single solution there (the black point). For sine and cosine, a line above 1 or below −1 meets the graph nowhere and the inverse is undefined."
      readouts={
        <>
          <Readout label="y" value={formatNumber(y)} />
          <Readout label={`principal value ${s.name} y`} value={has ? formatNumber(principal) : 'undefined'} />
          <Readout label="in degrees" value={has ? `${formatNumber((principal * 180) / Math.PI)}°` : '–'} />
          <Readout label="all solutions" value={general} />
          <Readout label="solutions in view" value={roots.length} />
        </>
      }
    >
      <Plot x={x} y={yAxis}>
        <Curve name={`${fn} θ`} x={TH} y={curves.all} muted />
        <Curve name="principal branch" x={curves.bt} y={curves.bv} slot={0} width={4} />
        <Points name="solutions" x={roots} y={roots.map(() => y)} slot={1} size={7} live />
        {has && <Points name="principal value" x={[principal]} y={[y]} emphasis size={10} live />}
        <Handle {...state.handle('y', { axis: 'y', label: 'y' })} />
      </Plot>
    </Figure>
  )
}
