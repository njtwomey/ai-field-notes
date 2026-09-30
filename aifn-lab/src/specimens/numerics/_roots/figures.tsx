import { bisection, brent, fixedPoint, newtonRoot, regulaFalsi, secant, type RootState } from 'aifn/numerics/roots'
import { polynomialRoots } from 'aifn/numerics/polynomial'
import { toFlat } from 'aifn/foundation/tensor'
import { trace, type Algorithm, type Trace } from 'aifn/foundation/trace'
import { useMemo, useState } from 'react'
import { Select, Slider } from '@lab/controls'
import { Figure } from '@lab/layout'
import { Readout, XYChart, type Segment, type XYSeries } from '@lab/viz'
import { TraceView } from '@lab/views'

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

// ---------------------------------------------------------------------------------------------------------------------
// 1. Root-finding steps.

export function RootStepsSpecimen() {
  const [which, setWhich] = useState('cubic')
  const [method, setMethod] = useState<RootMethod>('brent')
  const eq = EQUATIONS.find((e) => e.value === which)!
  const t = useMemo(() => traceRoot(method, eq), [method, eq])
  const curve = useMemo<XYSeries>(() => {
    const pad = 0.1 * (eq.hi - eq.lo)
    const xs = Array.from({ length: 300 }, (_, i) => eq.lo - pad + ((eq.hi - eq.lo + 2 * pad) * i) / 299)
    return { name: 'f', type: 'line', x: xs, y: xs.map(eq.f), slot: 0 }
  }, [eq])
  return (
    <TraceView
      title="Root-finding steps"
      trace={t}
      show={['|f(x)|']}
      startAtFirst
      defaultSize="L"
      controls={
        <>
          <Select label="equation" value={which} onChange={setWhich} options={EQUATIONS} />
          <Select label="method" value={method} onChange={setMethod} options={ROOT_METHODS} />
        </>
      }
      caption="The current estimate (ringed) on f, the bracket (the bar on the axis) for the bracketing methods, and the secant or tangent line whose zero gives the next estimate. Brent's state names the kind of step it took."
      renderState={(s, { position, trace: tr }) => {
        const segments: Segment[] = []
        if (s.lo !== undefined && s.hi !== undefined) segments.push({ from: [s.lo, 0], to: [s.hi, 0] })
        const next = tr.steps[position + 1]
        const line: XYSeries[] = []
        if (next && s.derivative !== undefined) {
          line.push({ name: 'tangent', type: 'line', x: [s.x, next.x], y: [s.fx, 0], dashed: true, slot: 2 })
        } else if (next && s.previous !== undefined) {
          line.push({
            name: 'secant',
            type: 'line',
            x: [s.previous, next.x],
            y: [eq.f(s.previous), 0],
            dashed: true,
            slot: 2,
          })
        }
        return (
          <XYChart
            xLabel="x"
            yLabel="f(x)"
            segments={segments}
            series={[
              curve,
              { name: 'zero', type: 'line', x: [curve.x[0], curve.x[curve.x.length - 1]], y: [0, 0], muted: true },
              ...line,
              { name: 'estimate', type: 'scatter', x: [s.x], y: [s.fx], emphasis: true },
            ]}
          />
        )
      }}
    />
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Orders of convergence.

export function ConvergenceOrderSpecimen() {
  const [which, setWhich] = useState('cubic')
  const eq = EQUATIONS.find((e) => e.value === which)!
  const runs = useMemo(() => ROOT_METHODS.map((m) => ({ ...m, tr: traceRoot(m.value, eq, 80) })), [eq])
  const series = useMemo<XYSeries[]>(
    () =>
      runs.map(({ label, slot, tr }) => ({
        name: label,
        slot,
        type: 'line',
        showPoints: true,
        x: tr.steps.map((s) => s.evaluations),
        y: toFlat(tr.series['|x − root|']).map((v) => Math.max(v, 1e-17)),
      })),
    [runs],
  )
  return (
    <Figure
      title="Orders of convergence"
      controls={<Select label="equation" value={which} onChange={setWhich} options={EQUATIONS} />}
      readouts={runs.map(({ label, tr }) => (
        <Readout key={label} label={label} value={`${tr.steps.at(-1)!.evaluations} evaluations`} />
      ))}
      caption="Error against evaluations of f on a log scale: bisection halves the error per evaluation (linear), the secant method has order 1.618, Newton order 2 (one evaluation of f and f′ together per step), and Brent is usually superlinear. Plain regula falsi can be slower than bisection when one end sticks."
    >
      <XYChart series={series} yLog xLabel="evaluations" yLabel="|x − root|" />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. Polynomial roots: Wilkinson's perturbation.

export function WilkinsonSpecimen() {
  const [n, setN] = useState(20)
  const [logEps, setLogEps] = useState(-7)
  const { exact, perturbed, converged } = useMemo(() => {
    // Coefficients of ∏ (x − k), k = 1…n, highest degree first.
    let c = [1]
    for (let k = 1; k <= n; k++) c = [...c, 0].map((v, i) => v - k * (i > 0 ? c[i - 1] : 0))
    const e = polynomialRoots(c)
    const d = [...c]
    d[1] -= 10 ** logEps // Wilkinson (1959) perturbed the x¹⁹ coefficient by 2⁻²³.
    const p = polynomialRoots(d)
    return { exact: e, perturbed: p, converged: p.converged && e.converged }
  }, [n, logEps])
  const maxImag = Math.max(...toFlat(perturbed.imag).map(Math.abs))
  return (
    <Figure
      title="Roots of Wilkinson's polynomial"
      controls={
        <>
          <Slider label="degree n" value={n} min={4} max={24} step={1} onChange={setN} />
          <Slider
            label="log₁₀ ε (perturbation of the xⁿ⁻¹ coefficient)"
            value={logEps}
            min={-14}
            max={-2}
            step={0.5}
            onChange={setLogEps}
          />
        </>
      }
      readouts={
        <>
          <Readout label="largest |imaginary part|" value={maxImag.toPrecision(3)} />
          <Readout label="QR converged" value={converged ? 'yes' : 'no'} />
        </>
      }
      caption="The roots of ∏(x − k), k = 1…n, as companion-matrix eigenvalues, before and after changing the coefficient of xⁿ⁻¹ by ε. For n = 20 a change of about 10⁻⁷ sends ten roots into the complex plane: the roots are ill conditioned even though the polynomial looks harmless."
    >
      <XYChart
        xLabel="real part"
        yLabel="imaginary part"
        series={[
          { name: 'roots of ∏(x − k)', type: 'scatter', x: toFlat(exact.real), y: toFlat(exact.imag), slot: 0 },
          { name: 'perturbed roots', type: 'scatter', x: toFlat(perturbed.real), y: toFlat(perturbed.imag), slot: 1 },
        ]}
      />
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

export function FixedPointSpecimen() {
  const [which, setWhich] = useState('cos')
  const [omega, setOmega] = useState(1)
  const [x0, setX0] = useState(0.2)
  const map = MAPS.find((m) => m.value === which)!
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
  const curve = useMemo(() => {
    const [a, b] = map.range
    const xs = Array.from({ length: 200 }, (_, i) => a + ((b - a) * i) / 199)
    return xs
  }, [map])
  return (
    <TraceView
      title="Fixed-point iteration"
      trace={t}
      show={['|g(x) − x|', 'contraction']}
      startAtFirst
      defaultSize="L"
      controls={
        <>
          <Select label="map" value={which} onChange={setWhich} options={MAPS} />
          <Slider label="relaxation ω" value={omega} min={0.1} max={1.5} onChange={setOmega} />
          <Slider label="start x₀" value={x0} min={map.range[0]} max={map.range[1]} onChange={setX0} />
        </>
      }
      caption="The cobweb of x ← (1 − ω)x + ω g(x) up to the current step. The observed contraction ‖Δx_t‖/‖Δx_{t−1}‖ estimates |g′| at the fixed point; below 1 the iteration converges linearly at that rate. For 3.2·x(1 − x) the fixed point is repelling (|g′| = 1.2) and the iterates settle on a 2-cycle; relaxation with ω < 1 can make it attracting."
      renderState={(_, { position, trace: tr }) => {
        const xs = tr.steps.slice(0, position + 1).map((s) => toFlat(s.x)[0])
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
        const gw = curve.map((x) => (1 - omega) * x + omega * map.g(x))
        return (
          <XYChart
            equalAspect
            xLabel="x"
            yLabel="next x"
            series={[
              { name: 'relaxed map', type: 'line', x: curve, y: gw, slot: 0 },
              { name: 'y = x', type: 'line', x: curve, y: curve, muted: true },
              { name: 'cobweb', type: 'line', x: cx, y: cy, slot: 1 },
              { name: 'x_t', type: 'scatter', x: [xs[xs.length - 1]], y: [xs[xs.length - 1]], emphasis: true },
            ]}
          />
        )
      }}
    />
  )
}
