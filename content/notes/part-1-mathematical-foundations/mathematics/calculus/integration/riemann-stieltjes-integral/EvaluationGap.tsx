import { useMemo } from 'react'
import { choice, Curve, Figure, formatNumber, Handle, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn/foundation/random'

type Integrator = 'smooth' | 'jump' | 'brownian'

const FINE = 2 ** 14
const K_MAX = 14

/** The integrator on a fine grid over [0, 1]: t², a unit jump at 0.5, or one Brownian path. */
function fineGrid(kind: Integrator): Float64Array {
  const g = new Float64Array(FINE + 1)
  if (kind === 'brownian') {
    const g5 = stream(5)
    const sd = Math.sqrt(1 / FINE)
    for (let i = 1; i <= FINE; i++) g[i] = g[i - 1] + sd * normal(g5)
  } else {
    for (let i = 0; i <= FINE; i++) {
      const t = i / FINE
      // The jump sits strictly inside a coarse step for every resolution, so one step always straddles it.
      g[i] = kind === 'smooth' ? t * t : t < 0.5 + 1 / (3 * FINE) ? 0 : 1
    }
  }
  return g
}

/** Left and right Riemann–Stieltjes sums of ∫ g dg over 2^k equal steps, with the variation and squared increments. */
function sums(g: Float64Array, k: number) {
  const m = 2 ** k
  const stride = FINE / m
  let left = 0
  let right = 0
  let variation = 0
  for (let j = 0; j < m; j++) {
    const a = g[j * stride]
    const b = g[(j + 1) * stride]
    left += a * (b - a)
    right += b * (b - a)
    variation += Math.abs(b - a)
  }
  return { left, right, gap: right - left, variation }
}

/**
 * Left- against right-point sums of ∫₀¹ g dg. Their gap is Σ(Δg)²: it vanishes for a smooth g, stays at 1 for a jump
 * shared by integrand and integrator, and settles at 1 = T for a Brownian path, whose variation also grows without
 * bound. Only in the first case does the Riemann–Stieltjes integral exist.
 */
export function EvaluationGap() {
  const state = useFigureState({
    kind: choice<Integrator>(
      [
        { value: 'smooth', label: 'smooth t²' },
        { value: 'jump', label: 'jump at 0.5' },
        { value: 'brownian', label: 'Brownian path' },
      ],
      'brownian',
      { label: 'integrator g' },
    ),
    k: int(6, { min: 1, max: K_MAX, step: 1, label: 'k (2ᵏ steps)' }),
  })
  const { kind, k, set } = state
  const g = useMemo(() => fineGrid(kind), [kind])
  const now = useMemo(() => sums(g, k), [g, k])

  const series = useMemo(() => {
    const ks = Array.from({ length: K_MAX }, (_, i) => i + 1)
    const all = ks.map((kk) => sums(g, kk))
    return [
      { name: 'left-point sum', x: ks, y: all.map((s) => s.left), slot: 0 },
      { name: 'right-point sum', x: ks, y: all.map((s) => s.right), slot: 1 },
      { name: 'gap Σ(Δg)²', x: ks, y: all.map((s) => s.gap), slot: 2, dashed: true },
    ] as const
  }, [g])

  const handles = useMemo(
    () => [{ kind: 'x' as const, at: k, label: 'k', onDrag: (x: number) => set('k', Math.round(x)) }],
    [k, set],
  )

  const xAxis = useAxis({ label: 'k (2ᵏ steps)', range: [1, K_MAX] })
  const yAxis = useAxis({ label: 'sum', hold: 'union' })
  return (
    <Figure
      title="When the evaluation point matters"
      state={state}
      caption="∫₀¹ g dg computed with 2ᵏ equal steps, evaluating the integrand at the left or the right end of each step. The gap between the two sums is Σ(Δg)². For the smooth g = t² it shrinks to 0 and both sums reach ½. For a unit jump at 0.5 the gap stays at 1: the integral does not exist. For a Brownian path the gap settles at the elapsed time 1 while the total variation Σ|Δg| keeps growing. Drag the vertical line or use the slider to change the resolution."

      readouts={
        <>
          <Readout label="left sum" value={formatNumber(now.left)} />
          <Readout label="right sum" value={formatNumber(now.right)} />
          <Readout label="gap Σ(Δg)²" value={formatNumber(now.gap)} />
          <Readout label="variation Σ|Δg|" value={formatNumber(now.variation)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        {(handles ?? []).map((h, i) => (
          <Handle key={i} {...h} />
        ))}
      </Plot>
    </Figure>
  )
}
