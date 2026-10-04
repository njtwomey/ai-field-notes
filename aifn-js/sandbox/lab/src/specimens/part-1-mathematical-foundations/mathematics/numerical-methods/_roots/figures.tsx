import { bisection, brent, fixedPoint, newtonRoot, regulaFalsi, secant, type RootState } from 'aifn/numerics/roots'
import { polyFromRoots, polynomialRoots } from 'aifn/numerics/polynomial'
import { imagPart, realPart, toFlat } from 'aifn/foundation/tensor'
import { trace, type Algorithm, type Trace } from 'aifn/foundation/trace'
import { useMemo } from 'react'
import { Figure } from '@lab/layout'
import { choice, row, slider, useFigureState, type FigureStateApi } from '@lab/state'
import { Annotation, Curve, Handle, Plot, Points, Readout, useAxis } from '@lab/viz'
import { TracePanel } from '@lab/views'

// ---------------------------------------------------------------------------------------------------------------------
// Scalar test equations with a bracket, a derivative and the root.

type Equation = {
  value: string
  label: string
  f: (x: number) => number
  df: (x: number) => number
  lo: number
  hi: number
  root: number
}

const EQUATIONS: Equation[] = [
  {
    value: 'cubic',
    label: 'x³ − 2x − 5 (Newton’s example)',
    f: (x) => x ** 3 - 2 * x - 5,
    df: (x) => 3 * x * x - 2,
    lo: 1,
    hi: 3,
    root: 2.0945514815423265,
  },
  {
    value: 'cos',
    label: 'cos x − x',
    f: (x) => Math.cos(x) - x,
    df: (x) => -Math.sin(x) - 1,
    lo: 0,
    hi: 1.5,
    root: 0.7390851332151607,
  },
  {
    value: 'flat',
    label: 'x·eˣ − 1 on a wide bracket',
    f: (x) => x * Math.exp(x) - 1,
    df: (x) => (x + 1) * Math.exp(x),
    lo: -1,
    hi: 4,
    root: 0.5671432904097838,
  },
  {
    value: 'steep',
    label: 'x¹⁰ − 1 (regula falsi stalls)',
    f: (x) => x ** 10 - 1,
    df: (x) => 10 * x ** 9,
    lo: 0,
    hi: 1.3,
    root: 1,
  },
]

type RootMethod = 'bisection' | 'regula-falsi' | 'plain-regula-falsi' | 'brent' | 'secant' | 'newton'
const ROOT_METHODS: { value: RootMethod; label: string; slot: number }[] = [
  { value: 'bisection', label: 'bisection', slot: 0 },
  { value: 'plain-regula-falsi', label: 'regula falsi', slot: 1 },
  { value: 'regula-falsi', label: 'regula falsi (Illinois)', slot: 2 },
  { value: 'brent', label: 'Brent', slot: 3 },
  { value: 'secant', label: 'secant', slot: 4 },
  { value: 'newton', label: 'Newton', slot: 5 },
]

type AnyRootState = RootState & { lo?: number; hi?: number; previous?: number; derivative?: number; method?: string }

function traceRoot(method: RootMethod, eq: Equation, steps = 60): Trace<AnyRootState> {
  const record = {
    x: (s: AnyRootState) => s.x,
    '|f(x)|': (s: AnyRootState) => Math.abs(s.fx),
    '|x − root|': (s: AnyRootState) => Math.abs(s.x - eq.root),
  }
  const opts = { record, stopOnNonFinite: false }
  const bracket = { lo: eq.lo, hi: eq.hi }
  const as = <O,>(alg: Algorithm<O, AnyRootState>, o: O) => trace(alg, o, steps, opts)
  switch (method) {
    case 'bisection':
      return as(bisection(eq.f) as Algorithm<typeof bracket, AnyRootState>, bracket)
    case 'regula-falsi':
      return as(regulaFalsi(eq.f) as Algorithm<typeof bracket, AnyRootState>, bracket)
    case 'plain-regula-falsi':
      return as(regulaFalsi(eq.f, { illinois: false }) as Algorithm<typeof bracket, AnyRootState>, bracket)
    case 'brent':
      return as(brent(eq.f) as Algorithm<typeof bracket, AnyRootState>, bracket)
    case 'secant':
      return as(secant(eq.f) as Algorithm<{ x0: number; x1: number }, AnyRootState>, {
        x0: eq.hi,
        x1: eq.hi - 0.1 * (eq.hi - eq.lo),
      })
    case 'newton':
      return as(
        newtonRoot((x) => ({ value: eq.f(x), derivative: eq.df(x) })) as Algorithm<{ x0: number }, AnyRootState>,
        {
          x0: eq.hi,
        },
      )
  }
}

const EQUATION_OPTIONS = EQUATIONS.map(({ value, label }) => ({ value, label }))
const METHOD_OPTIONS = ROOT_METHODS.map(({ value, label }) => ({ value, label }))

// ---------------------------------------------------------------------------------------------------------------------
// 1. Root-finding steps.

/** f with the current estimate, the bracket on the axis and the line whose zero is the next estimate. */
function RootPicture({ eq, s, next }: { eq: Equation; s: AnyRootState; next?: AnyRootState }) {
  const curve = useMemo(() => {
    const pad = 0.1 * (eq.hi - eq.lo)
    const xs = Array.from({ length: 300 }, (_, i) => eq.lo - pad + ((eq.hi - eq.lo + 2 * pad) * i) / 299)
    return { x: xs, y: xs.map(eq.f) }
  }, [eq])
  const line =
    next && s.derivative !== undefined
      ? { name: 'tangent', x: [s.x, next.x], y: [s.fx, 0] }
      : next && s.previous !== undefined
        ? { name: 'secant', x: [s.previous, next.x], y: [eq.f(s.previous), 0] }
        : null
  const x = useAxis({ label: 'x', key: eq.value, hold: 'initial' })
  const y = useAxis({ label: 'f(x)', key: eq.value, hold: 'initial' })
  return (
    <Plot x={x} y={y}>
      <Annotation y={0} />
      <Curve name="f" x={curve.x} y={curve.y} slot={0} />
      {s.lo !== undefined && s.hi !== undefined && (
        <Curve name="bracket" x={[s.lo, s.hi]} y={[0, 0]} slot={3} width={6} />
      )}
      {line && <Curve name={line.name} x={line.x} y={line.y} slot={2} dashed />}
      <Points name="estimate" x={[s.x]} y={[s.fx]} emphasis />
    </Plot>
  )
}

export function RootStepsSpecimen() {
  const state = useFigureState({
    which: choice(EQUATION_OPTIONS, 'cubic', { label: 'equation' }),
    method: choice(METHOD_OPTIONS, 'brent', { label: 'method' }),
  })
  const eq = EQUATIONS.find((e) => e.value === state.which)!
  const method = state.method as RootMethod
  const t = useMemo(() => traceRoot(method, eq), [method, eq])
  return (
    <Figure
      title="Root-finding steps"
      purpose="Each root finder replaces its estimate by the zero of a simple model of f: the bracket's midpoint, a secant through two points, or the tangent at one."
      defaultSize="L"
      state={state}
      caption="Play from step 0. The current estimate (ink) on f, the bracket (the thick bar on the axis) for the bracketing methods, and the dashed secant or tangent whose zero gives the next estimate. Try x¹⁰ − 1 with plain regula falsi: one end of the bracket never moves and progress stalls; the Illinois variant fixes it. Brent's state names the kind of step it took."
    >
      <TracePanel
        trace={t}
        show={['|f(x)|']}
        renderState={(s, { position, trace: tr }) => <RootPicture eq={eq} s={s} next={tr.steps[position + 1]} />}
      />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Orders of convergence.

export function ConvergenceOrderSpecimen() {
  const state = useFigureState({ which: choice(EQUATION_OPTIONS, 'cubic', { label: 'equation' }) })
  const eq = EQUATIONS.find((e) => e.value === state.which)!
  const runs = useMemo(
    () =>
      ROOT_METHODS.map((m) => {
        const tr = traceRoot(m.value, eq, 80)
        return {
          ...m,
          last: tr.steps.at(-1)!.evaluations,
          x: tr.steps.map((s) => s.evaluations),
          y: toFlat(tr.series['|x − root|']).map((v) => Math.max(v, 1e-17)),
        }
      }),
    [eq],
  )
  const x = useAxis({ label: 'evaluations', range: [0, 60] })
  const y = useAxis({ label: '|x − root|', log: true, range: [1e-17, 10] })
  return (
    <Figure
      title="Orders of convergence"
      purpose="The error per evaluation of f falls linearly for bisection, with order 1.618 for the secant method and order 2 for Newton: on a log scale the curves are a line, then steeper and steeper drops."
      state={state}
      readouts={runs.map(({ label, last }) => (
        <Readout key={label} label={label} value={`${last} evaluations`} />
      ))}
      caption="Error against evaluations of f, on held axes. Newton counts one evaluation of f and f′ together per step. Brent is usually superlinear; plain regula falsi can be slower than bisection when one end sticks (try x¹⁰ − 1). Errors below 10⁻¹⁷ are drawn at 10⁻¹⁷."
    >
      <Plot x={x} y={y}>
        {runs.map((r) => (
          <Curve key={r.value} name={r.label} x={r.x} y={r.y} slot={r.slot} showPoints />
        ))}
      </Plot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. Polynomial roots: Wilkinson's perturbation.

export function WilkinsonSpecimen() {
  const state = useFigureState({
    n: slider(4, 24, 20, { step: 1, label: 'degree n' }),
    logEps: slider(-14, -2, -7, { step: 0.5, label: 'log₁₀ ε (perturbation of the xⁿ⁻¹ coefficient)' }),
  })
  const { n, logEps } = state
  const { exact, perturbed, converged } = useMemo(() => {
    // Coefficients of ∏ (x − k), k = 1…n, highest degree first.
    const c = toFlat(
      polyFromRoots(
        Array.from({ length: n }, (_, k) => k + 1),
        { real: true },
      ),
    )
    // polynomialRoots: the companion matrix's eigenvalues, with the QR iteration's convergence flag.
    const e = polynomialRoots(c)
    const d = [...c]
    d[1] -= 10 ** logEps // Wilkinson (1959) perturbed the x¹⁹ coefficient by 2⁻²³.
    const p = polynomialRoots(d)
    return {
      exact: { x: toFlat(realPart(e.roots)), y: toFlat(imagPart(e.roots)) },
      perturbed: { x: toFlat(realPart(p.roots)), y: toFlat(imagPart(p.roots)) },
      converged: p.converged && e.converged,
    }
  }, [n, logEps])
  const maxImag = Math.max(...perturbed.y.map(Math.abs))
  const x = useAxis({ label: 'real part', hold: 'union', key: n })
  const y = useAxis({ label: 'imaginary part', hold: 'union', key: n })
  return (
    <Figure
      title="Roots of Wilkinson's polynomial"
      purpose="The roots of ∏(x − k) are ill conditioned: changing one coefficient by 10⁻⁷ sends half of the 20 roots into the complex plane."
      state={state}
      readouts={
        <>
          <Readout label="largest |imaginary part|" value={maxImag.toPrecision(3)} />
          <Readout label="QR converged" value={converged ? 'yes' : 'no'} />
        </>
      }
      caption="The roots of ∏(x − k), k = 1…n, from polynomialRoots (companion-matrix eigenvalues), before and after changing the coefficient of xⁿ⁻¹ by ε. Slide ε down to 10⁻¹⁴ and the perturbed roots return to the real axis; lower the degree and much larger changes are harmless."
    >
      <Plot x={x} y={y}>
        <Points name="roots of ∏(x − k)" x={exact.x} y={exact.y} slot={0} />
        <Points name="perturbed roots" x={perturbed.x} y={perturbed.y} slot={1} />
      </Plot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 4. Fixed-point iteration with diagnostics.

const MAPS = [
  { value: 'cos', label: 'g(x) = cos x', g: (x: number) => Math.cos(x), range: [0, 1.6] as [number, number] },
  {
    value: 'logistic',
    label: 'g(x) = 3.2·x(1 − x)',
    g: (x: number) => 3.2 * x * (1 - x),
    range: [0, 1] as [number, number],
  },
  { value: 'sqrt', label: 'g(x) = √(x + 2)', g: (x: number) => Math.sqrt(x + 2), range: [0, 3] as [number, number] },
]
const MAP_OPTIONS = MAPS.map(({ value, label }) => ({ value, label }))

const FIXED_SCHEMA = {
  setup: row('1 · map', {
    which: choice(MAP_OPTIONS, 'cos', { label: 'map' }),
    omega: slider(0.1, 1.5, 1, { label: 'relaxation ω' }),
  }),
  x0: slider(0, 3, 0.2, { label: 'start x₀' }),
}

/** The cobweb up to the current step over the relaxed map and y = x; the start is dragged along the x axis. */
function Cobweb({
  map,
  omega,
  xs,
  state,
}: {
  map: (typeof MAPS)[number]
  omega: number
  xs: number[]
  state: FigureStateApi<typeof FIXED_SCHEMA>
}) {
  const curve = useMemo(() => {
    const [a, b] = map.range
    const x = Array.from({ length: 200 }, (_, i) => a + ((b - a) * i) / 199)
    return { x, g: x.map((v) => (1 - omega) * v + omega * map.g(v)) }
  }, [map, omega])
  const web = useMemo(() => {
    const cx: number[] = []
    const cy: number[] = []
    xs.forEach((x, k) => {
      if (k === 0) {
        cx.push(x)
        cy.push(0)
      }
      const next = xs[k + 1]
      if (next === undefined) return
      cx.push(x, next)
      cy.push(next, next)
    })
    return { x: cx, y: cy }
  }, [xs])
  const last = xs[xs.length - 1]
  const x = useAxis({ label: 'x', range: map.range, key: map.value })
  const y = useAxis({ label: 'next x', range: map.range, key: map.value, equal: x })
  return (
    <Plot x={x} y={y}>
      <Curve name="y = x" x={curve.x} y={curve.x} muted />
      <Curve name="relaxed map" x={curve.x} y={curve.g} slot={0} />
      <Curve name="cobweb" x={web.x} y={web.y} slot={1} />
      <Points name="x_t" x={[last]} y={[last]} emphasis />
      <Handle {...state.handle('x0', { label: 'x₀' })} />
    </Plot>
  )
}

export function FixedPointSpecimen() {
  const state = useFigureState(FIXED_SCHEMA)
  const { which, omega } = state.setup
  const map = MAPS.find((m) => m.value === which)!
  const x0 = Math.min(Math.max(state.x0, map.range[0]), map.range[1])
  const t = useMemo(
    () =>
      trace(
        fixedPoint((x) => toFlat(x).map(map.g), { relaxation: omega, patience: 40 }),
        { x0: [x0] },
        80,
        {
          record: {
            x: (s) => toFlat(s.x)[0],
            '|g(x) − x|': (s) => s.residualNorm,
            contraction: (s) => s.contraction,
            'error bound': (s) => (Number.isFinite(s.errorBound) ? s.errorBound : NaN),
          },
          stopOnNonFinite: false,
        },
      ),
    [map, omega, x0],
  )
  return (
    <Figure
      title="Fixed-point iteration"
      purpose="x ← g(x) converges when |g′| < 1 at the fixed point, at that rate; relaxation x ← (1 − ω)x + ω g(x) changes the rate and can make a repelling fixed point attracting."
      defaultSize="L"
      state={state}
      caption="Play from step 0 to draw the cobweb; drag x₀ along the x axis (or use its slider). The observed contraction ‖Δx_t‖/‖Δx_{t−1}‖ estimates |g′| at the fixed point. For 3.2·x(1 − x) the fixed point is repelling (|g′| = 1.2) and the iterates settle on a 2-cycle; set ω below about 0.9 and it attracts."
    >
      <TracePanel
        trace={t}
        show={['|g(x) − x|', 'contraction']}
        renderState={(_, { position, trace: tr }) => (
          <Cobweb
            map={map}
            omega={omega}
            xs={tr.steps.slice(0, position + 1).map((s) => toFlat(s.x)[0])}
            state={state}
          />
        )}
      />
    </Figure>
  )
}
