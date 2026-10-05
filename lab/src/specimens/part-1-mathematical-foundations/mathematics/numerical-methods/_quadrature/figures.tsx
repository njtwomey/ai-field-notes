import {
  gaussHermite,
  gaussKronrod,
  gaussLaguerre,
  gaussLegendre,
  integrate,
  integrateGauss,
  integrateMonteCarlo,
  quasiMonteCarlo,
  romberg,
  simpson,
  trapezoid,
  type GaussKronrodState,
} from 'aifn-compute/numerics/quadrature'
import { child, stream } from 'aifn-compute/foundation/random'
import { toFlat, type Vector } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { logGamma } from 'aifn-compute/numerics/special'
import { useMemo } from 'react'
import { Figure } from 'aifn-render/layout'
import { choice, slider, useFigureState } from 'aifn-render/state'
import { Curve, Plot, Points, Readout, Segments, useAxis } from 'aifn-render/viz'
import { TracePanel } from '@lab/views'

// ---------------------------------------------------------------------------------------------------------------------
// Integrands on [0, 1] with their exact integrals.

type Integrand = { value: string; label: string; f: (x: number) => number; exact: number }
type Line = { name: string; x: number[]; y: number[] }

const INTEGRANDS: Integrand[] = [
  { value: 'exp', label: 'eˣ (smooth)', f: Math.exp, exact: Math.E - 1 },
  {
    value: 'runge',
    label: '1/(1 + 25(2x − 1)²) (Runge)',
    f: (x) => 1 / (1 + 25 * (2 * x - 1) ** 2),
    exact: ((2 / 5) * Math.atan(5)) / 2,
  },
  { value: 'sqrt', label: '√x (singular derivative at 0)', f: Math.sqrt, exact: 2 / 3 },
  { value: 'kink', label: '|x − 0.3| (kink)', f: (x) => Math.abs(x - 0.3), exact: (0.3 ** 2 + 0.7 ** 2) / 2 },
  { value: 'oscillating', label: 'cos(40x)', f: (x) => Math.cos(40 * x), exact: Math.sin(40) / 40 },
]

// ---------------------------------------------------------------------------------------------------------------------
// 1. Error against the number of evaluations.

const INTEGRAND_OPTIONS = INTEGRANDS.map(({ value, label }) => ({ value, label }))

export function ErrorVsNSpecimen() {
  const state = useFigureState({ which: choice(INTEGRAND_OPTIONS, 'exp', { label: 'integrand on [0, 1]' }) })
  const g = INTEGRANDS.find((i) => i.value === state.which)!
  const lines = useMemo<Line[]>(() => {
    const err = (v: number) => Math.max(Math.abs(v - g.exact), 1e-17)
    const panels = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    const points = Array.from({ length: 40 }, (_, i) => i + 1)
    const rom = trace(romberg(g.f, { atol: 0, rtol: 0 }), { a: 0, b: 1 }, 14, {
      record: { value: (s) => s.value, evaluations: (s) => s.evaluations },
    })
    const gk = trace(gaussKronrod(g.f, { atol: 0, rtol: 0 }), { a: 0, b: 1 }, 60, {
      record: { value: (s) => s.value, evaluations: (s) => s.evaluations },
    })
    return [
      { name: 'trapezoid', x: panels.map((n) => n + 1), y: panels.map((n) => err(trapezoid(g.f, 0, 1, { n }))) },
      { name: 'Simpson', x: panels.map((n) => n + 1), y: panels.map((n) => err(simpson(g.f, 0, 1, { n }))) },
      { name: 'Gauss–Legendre', x: points, y: points.map((n) => err(integrateGauss(g.f, 0, 1, { n }))) },
      { name: 'Romberg', x: toFlat(rom.series.evaluations), y: toFlat(rom.series.value).map(err) },
      { name: 'adaptive Gauss–Kronrod', x: toFlat(gk.series.evaluations), y: toFlat(gk.series.value).map(err) },
    ]
  }, [g])
  const x = useAxis({ label: 'evaluations of f', log: true, range: [1, 3000] })
  const y = useAxis({ label: 'absolute error', log: true, range: [1e-17, 1] })
  return (
    <Figure
      title="Quadrature error against evaluations"
      purpose="On a smooth integrand the trapezoid error falls as n⁻², Simpson's as n⁻⁴ and Gauss–Legendre's faster than any power; a kink or singular derivative caps every fixed rule at a low order, and only adaptive subdivision recovers."
      defaultSize="L"
      state={state}
      caption="|estimate − exact| against evaluations of f, log-log, on axes held across integrands so the slopes compare. Errors below 10⁻¹⁷ (machine precision) are drawn at 10⁻¹⁷. Switch to |x − 0.3| or √x: every fixed rule flattens to a low order, while adaptive Gauss–Kronrod keeps falling."
    >
      <Plot x={x} y={y}>
        {lines.map((l, i) => (
          <Curve key={l.name} name={l.name} x={l.x} y={l.y} slot={i} showPoints />
        ))}
      </Plot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Adaptive Gauss–Kronrod subdivision.

const ADAPTIVE_X = Array.from({ length: 400 }, (_, i) => i / 399)

/** The integrand with the current intervals' boundaries. */
function Subdivision({ g, s }: { g: Integrand; s: GaussKronrodState }) {
  const ys = useMemo(() => ADAPTIVE_X.map(g.f), [g])
  const segments = useMemo(() => {
    const top = Math.max(...ys)
    const bottom = Math.min(0, ...ys)
    return s.intervals.map((i) => ({ from: [i.a, bottom] as const, to: [i.a, top] as const }))
  }, [s.intervals, ys])
  const x = useAxis({ label: 'x', range: [0, 1] })
  const y = useAxis({ label: 'f(x)', key: g.value, hold: 'initial' })
  return (
    <Plot x={x} y={y}>
      <Segments segments={segments} />
      <Curve name="f" x={ADAPTIVE_X} y={ys} slot={0} />
    </Plot>
  )
}

export function AdaptiveSpecimen() {
  const state = useFigureState({
    which: choice(INTEGRAND_OPTIONS, 'sqrt', { label: 'integrand' }),
    logTol: slider(-14, -3, -10, { step: 1, label: 'log₁₀ tolerance' }),
  })
  const g = INTEGRANDS.find((i) => i.value === state.which)!
  const logTol = state.logTol
  const t = useMemo(
    () =>
      trace(gaussKronrod(g.f, { atol: 10 ** logTol, rtol: 0 }), { a: 0, b: 1 }, 200, {
        record: {
          'estimated error': (s) => s.error,
          'actual error': (s) => Math.max(Math.abs(s.value - g.exact), 1e-17),
          intervals: (s) => s.intervals.length,
        },
      }),
    [g, logTol],
  )
  return (
    <Figure
      title="Adaptive Gauss–Kronrod subdivision"
      purpose="Adaptive quadrature bisects the interval with the largest error estimate, so the intervals crowd where the integrand is hard and the estimated error bounds the actual one."
      defaultSize="L"
      state={state}
      caption="Play from step 0: each step bisects one interval. The vertical bars are the interval boundaries; they crowd at the √x singularity (or the kink, or the Runge peak). The estimated error comes from the 7-point Gauss rule nested inside the 15-point Kronrod rule; switch the error series to a log scale to compare it with the actual error."
    >
      <TracePanel
        trace={t}
        show={['estimated error', 'actual error']}
        renderState={(s: GaussKronrodState) => <Subdivision g={g} s={s} />}
      />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. Monte Carlo against quasi–Monte Carlo.

export function MonteCarloSpecimen() {
  const state = useFigureState({
    d: slider(1, 12, 4, { step: 1, label: 'dimension d' }),
    seed: slider(1, 30, 1, { step: 1, label: 'seed' }),
  })
  const { d, seed } = state
  const results = useMemo(() => {
    // ∫ over [0, 1]^d of ∏ (π/2) sin(π x_j) = 1.
    const f = (x: Vector) => toFlat(x).reduce((p, v) => p * (Math.PI / 2) * Math.sin(Math.PI * v), 1)
    const lo = new Array<number>(d).fill(0)
    const hi = new Array<number>(d).fill(1)
    const ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    const s = stream(seed)
    const mc = ns.map((n) => integrateMonteCarlo(child(s, 'mc', n), f, lo, hi, { n: 8 * n }))
    const qmc = ns.map((n) => quasiMonteCarlo(child(s, 'qmc', n), f, lo, hi, { n, replicates: 8 }))
    const halton = ns.map((n) =>
      quasiMonteCarlo(child(s, 'halton', n), f, lo, hi, { n, replicates: 8, sequence: 'halton' }),
    )
    const err = (r: { value: number }) => Math.max(Math.abs(r.value - 1), 1e-17)
    return {
      ns: ns.map((n) => 8 * n),
      mc: errorsOf(mc, err),
      qmc: errorsOf(qmc, err),
      halton: errorsOf(halton, err),
      lastSe: mc.at(-1)!.standardError,
    }
  }, [d, seed])
  const x = useAxis({ label: 'evaluations n', log: true })
  const y = useAxis({ label: 'error', log: true, hold: 'union' })
  return (
    <Figure
      title="Monte Carlo against quasi–Monte Carlo"
      purpose="Monte Carlo's error falls as n^(−1/2) in every dimension; randomly shifted low-discrepancy points fall nearly as n⁻¹ in low dimension and lose their edge as d grows."
      defaultSize="L"
      state={state}
      readouts={<Readout label="largest n: MC standard error" value={results.lastSe.toExponential(2)} />}
      caption="∫ ∏(π/2)sin(πx_j) dx = 1 over [0, 1]^d. Solid: |error|; dashed: the standard error from 8 independent replicates (8 random shifts for Sobol and Halton). Raise d from 1 to 12 and the Sobol line's slope approaches Monte Carlo's."
    >
      <Plot x={x} y={y}>
        <Curve name="Monte Carlo |error|" x={results.ns} y={results.mc.error} slot={0} showPoints />
        <Curve name="Monte Carlo standard error" x={results.ns} y={results.mc.se} slot={0} dashed />
        <Curve name="Sobol (shifted) |error|" x={results.ns} y={results.qmc.error} slot={1} showPoints />
        <Curve name="Sobol standard error" x={results.ns} y={results.qmc.se} slot={1} dashed />
        <Curve name="Halton (shifted) |error|" x={results.ns} y={results.halton.error} slot={2} showPoints />
      </Plot>
    </Figure>
  )
}

/** Errors and standard errors of a list of quadrature results. */
const errorsOf = (rs: { value: number; standardError: number }[], err: (r: { value: number }) => number) => ({
  error: rs.map(err),
  se: rs.map((r) => r.standardError),
})

// ---------------------------------------------------------------------------------------------------------------------
// 4. Gaussian nodes and weights.

const RULES = [
  { value: 'legendre', label: 'Gauss–Legendre, weight 1 on [−1, 1]' },
  { value: 'hermite', label: 'Gauss–Hermite, weight e^{−x²}' },
  { value: 'laguerre', label: 'Gauss–Laguerre, weight e^{−x} on [0, ∞)' },
] as const

export function GaussNodesSpecimen() {
  const state = useFigureState({
    family: choice(RULES, 'legendre', { label: 'rule' }),
    n: slider(1, 40, 8, { step: 1, label: 'points n' }),
  })
  const { family, n } = state
  const { nodes, weights, check, stems } = useMemo(() => {
    const rule = family === 'legendre' ? gaussLegendre(n) : family === 'hermite' ? gaussHermite(n) : gaussLaguerre(n)
    const x = toFlat(rule.nodes)
    const w = toFlat(rule.weights)
    // ∫ w(x) x^{2n−2} dx against the exact moment, a check that the rule integrates high degrees exactly:
    // 2/(k + 1) on [−1, 1]; Γ((k + 1)/2) for Hermite (even k); k! = Γ(k + 1) for Laguerre.
    const k = 2 * n - 2
    const estimate = w.reduce((s, wi, i) => s + wi * x[i] ** k, 0)
    const exact =
      family === 'legendre' ? 2 / (k + 1) : Math.exp(logGamma(family === 'hermite' ? (k + 1) / 2 : k + 1) as number)
    // Stems from the axis (or, on a log axis, from below the smallest weight) to each weight.
    const base = family === 'legendre' ? 0 : Math.min(...w.filter((v) => v > 0)) / 10
    const stems = x.map((xi, i) => ({ from: [xi, base] as const, to: [xi, w[i]] as const }))
    return { nodes: x, weights: w, check: Math.abs(estimate - exact) / exact, stems }
  }, [family, n])
  const x = useAxis({ label: 'node xᵢ' })
  const y = useAxis({ label: 'weight wᵢ', log: family !== 'legendre' })
  return (
    <Figure
      title="Gaussian quadrature nodes and weights"
      purpose="An n-point Gaussian rule puts its nodes at the roots of the n-th orthogonal polynomial for its weight, and so integrates every polynomial of degree up to 2n − 1 exactly."
      state={state}
      readouts={<Readout label={`relative error of the degree-${2 * n - 2} moment`} value={check.toExponential(2)} />}
      caption="Each stem is one node and its weight. Weights shrink towards the ends; for Hermite and Laguerre (log scale) they fall by many orders of magnitude far out. The readout checks the highest even moment the rule should integrate exactly."
    >
      <Plot x={x} y={y}>
        <Segments segments={stems} slot={0} />
        <Points name="weights" x={nodes} y={weights} slot={0} />
      </Plot>
    </Figure>
  )
}

/** ∫ of the integrands on [0, ∞): a check that `integrate` handles infinite limits. */
export function InfiniteLimitsSpecimen() {
  const cases = useMemo(
    () =>
      [
        {
          label: '∫ e^{−x²} over ℝ = √π',
          f: (x: number) => Math.exp(-x * x),
          a: -Infinity,
          b: Infinity,
          exact: Math.sqrt(Math.PI),
        },
        { label: '∫₀^∞ 1/(1 + x²) = π/2', f: (x: number) => 1 / (1 + x * x), a: 0, b: Infinity, exact: Math.PI / 2 },
        { label: '∫₀^∞ x³e^{−x} = 6', f: (x: number) => x ** 3 * Math.exp(-x), a: 0, b: Infinity, exact: 6 },
        { label: '∫_{−∞}^0 eˣ = 1', f: Math.exp, a: -Infinity, b: 0, exact: 1 },
      ].map((c) => ({ ...c, r: integrate(c.f, c.a, c.b, { atol: 1e-12, rtol: 1e-12 }) })),
    [],
  )
  const romb = useMemo(() => run(romberg(Math.exp, {}), { a: 0, b: 1 }, 20), [])
  return (
    <Figure
      title="Infinite limits and error estimates"
      purpose="integrate maps an infinite interval onto a finite one by a change of variables (QUADPACK's QAGI) and integrates there adaptively; its error estimate bounds the actual error."
      hoverReadout={false}
    >
      <table className="w-full text-xs">
        <thead className="text-muted-foreground">
          <tr>
            <th className="text-left font-normal">integral</th>
            <th className="text-right font-normal">estimate</th>
            <th className="text-right font-normal">estimated error</th>
            <th className="text-right font-normal">actual error</th>
            <th className="text-right font-normal">evaluations</th>
          </tr>
        </thead>
        <tbody className="font-mono tabular-nums">
          {cases.map((c) => (
            <tr key={c.label}>
              <td className="font-sans">{c.label}</td>
              <td className="text-right">{c.r.value.toPrecision(15)}</td>
              <td className="text-right">{c.r.error.toExponential(1)}</td>
              <td className="text-right">{Math.abs(c.r.value - c.exact).toExponential(1)}</td>
              <td className="text-right">{c.r.evaluations}</td>
            </tr>
          ))}
          <tr>
            <td className="font-sans">Romberg ∫₀¹ eˣ = e − 1</td>
            <td className="text-right">{romb.value.toPrecision(15)}</td>
            <td className="text-right">{romb.error.toExponential(1)}</td>
            <td className="text-right">{Math.abs(romb.value - (Math.E - 1)).toExponential(1)}</td>
            <td className="text-right">{romb.evaluations}</td>
          </tr>
        </tbody>
      </table>
    </Figure>
  )
}
