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
} from 'aifn/quadrature'
import { stream } from 'aifn/random'
import { toFlat, type Vector } from 'aifn/tensor'
import { run, trace } from 'aifn/trace'
import { useMemo, useState } from 'react'
import { Select, Slider } from '@lab/controls'
import { Figure } from '@lab/layout'
import { Readout, XYChart, type Segment, type XYSeries } from '@lab/viz'
import { TraceView } from '@lab/views'

// ---------------------------------------------------------------------------------------------------------------------
// Integrands on [0, 1] with their exact integrals.

type Integrand = { value: string; label: string; f: (x: number) => number; exact: number }

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

export function ErrorVsNSpecimen() {
  const [which, setWhich] = useState('exp')
  const g = INTEGRANDS.find((i) => i.value === which)!
  const series = useMemo<XYSeries[]>(() => {
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
      {
        name: 'trapezoid',
        type: 'line',
        showPoints: true,
        slot: 0,
        x: panels.map((n) => n + 1),
        y: panels.map((n) => err(trapezoid(g.f, 0, 1, { n }))),
      },
      {
        name: 'Simpson',
        type: 'line',
        showPoints: true,
        slot: 1,
        x: panels.map((n) => n + 1),
        y: panels.map((n) => err(simpson(g.f, 0, 1, { n }))),
      },
      {
        name: 'Gauss–Legendre',
        type: 'line',
        showPoints: true,
        slot: 2,
        x: points,
        y: points.map((n) => err(integrateGauss(g.f, 0, 1, { n }))),
      },
      {
        name: 'Romberg',
        type: 'line',
        showPoints: true,
        slot: 3,
        x: toFlat(rom.series.evaluations),
        y: toFlat(rom.series.value).map(err),
      },
      {
        name: 'adaptive Gauss–Kronrod',
        type: 'line',
        showPoints: true,
        slot: 4,
        x: toFlat(gk.series.evaluations),
        y: toFlat(gk.series.value).map(err),
      },
    ]
  }, [g])
  return (
    <Figure
      title="Quadrature error against evaluations"
      defaultSize="L"
      controls={<Select label="integrand on [0, 1]" value={which} onChange={setWhich} options={INTEGRANDS} />}
      caption="|estimate − exact| against evaluations of f, log-log. On a smooth integrand the trapezoid error falls as n⁻², Simpson's as n⁻⁴ and Gauss–Legendre's faster than any power; a kink or a singular derivative caps every fixed rule at a low order, and only adaptive subdivision recovers."
    >
      <XYChart series={series} xLog yLog xLabel="evaluations of f" yLabel="absolute error" />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Adaptive Gauss–Kronrod subdivision.

export function AdaptiveSpecimen() {
  const [which, setWhich] = useState('sqrt')
  const [logTol, setLogTol] = useState(-10)
  const g = INTEGRANDS.find((i) => i.value === which)!
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
  const curve = useMemo<XYSeries>(() => {
    const xs = Array.from({ length: 400 }, (_, i) => i / 399)
    return { name: 'f', type: 'line', x: xs, y: xs.map(g.f), slot: 0 }
  }, [g])
  return (
    <TraceView
      title="Adaptive Gauss–Kronrod subdivision"
      trace={t}
      show={['estimated error', 'actual error']}
      defaultSize="L"
      controls={
        <>
          <Select label="integrand" value={which} onChange={setWhich} options={INTEGRANDS} />
          <Slider label="log₁₀ tolerance" value={logTol} min={-14} max={-3} step={1} onChange={setLogTol} />
        </>
      }
      caption="Each step bisects the interval with the largest error estimate. The vertical bars are the interval boundaries; they crowd where f is hard (the kink, the √x singularity, the Runge peak). The estimated error (from the 7-point Gauss rule inside the 15-point Kronrod rule) bounds the actual error."
      renderState={(s: GaussKronrodState) => {
        const ys = curve.y
        const top = Math.max(...ys)
        const bottom = Math.min(0, ...ys)
        const segments: Segment[] = s.intervals.map((i) => ({ from: [i.a, bottom], to: [i.a, top] }))
        return <XYChart xLabel="x" yLabel="f(x)" segments={segments} series={[curve]} />
      }}
    />
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. Monte Carlo against quasi–Monte Carlo.

export function MonteCarloSpecimen() {
  const [d, setD] = useState(4)
  const [seed, setSeed] = useState(1)
  const results = useMemo(() => {
    // ∫ over [0, 1]^d of ∏ (π/2) sin(π x_j) = 1.
    const f = (x: Vector) => toFlat(x).reduce((p, v) => p * (Math.PI / 2) * Math.sin(Math.PI * v), 1)
    const lo = new Array<number>(d).fill(0)
    const hi = new Array<number>(d).fill(1)
    const ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    const s = stream(seed)
    const mc = ns.map((n) => integrateMonteCarlo(f, lo, hi, s.child('mc', n), { n: 8 * n }))
    const qmc = ns.map((n) => quasiMonteCarlo(f, lo, hi, s.child('qmc', n), { n, replicates: 8 }))
    const halton = ns.map((n) =>
      quasiMonteCarlo(f, lo, hi, s.child('halton', n), { n, replicates: 8, sequence: 'halton' }),
    )
    return { ns: ns.map((n) => 8 * n), mc, qmc, halton }
  }, [d, seed])
  const err = (r: { value: number }) => Math.max(Math.abs(r.value - 1), 1e-17)
  const series: XYSeries[] = [
    { name: 'Monte Carlo |error|', type: 'line', showPoints: true, slot: 0, x: results.ns, y: results.mc.map(err) },
    {
      name: 'Monte Carlo standard error',
      type: 'line',
      dashed: true,
      slot: 0,
      x: results.ns,
      y: results.mc.map((r) => r.standardError),
    },
    {
      name: 'Sobol (shifted) |error|',
      type: 'line',
      showPoints: true,
      slot: 1,
      x: results.ns,
      y: results.qmc.map(err),
    },
    {
      name: 'Sobol standard error',
      type: 'line',
      dashed: true,
      slot: 1,
      x: results.ns,
      y: results.qmc.map((r) => r.standardError),
    },
    {
      name: 'Halton (shifted) |error|',
      type: 'line',
      showPoints: true,
      slot: 2,
      x: results.ns,
      y: results.halton.map(err),
    },
  ]
  return (
    <Figure
      title="Monte Carlo against quasi–Monte Carlo"
      defaultSize="L"
      controls={
        <>
          <Slider label="dimension d" value={d} min={1} max={12} step={1} onChange={setD} />
          <Slider label="seed" value={seed} min={1} max={30} step={1} onChange={setSeed} />
        </>
      }
      readouts={
        <Readout label="largest n: MC standard error" value={results.mc.at(-1)!.standardError.toExponential(2)} />
      }
      caption="∫ ∏(π/2)sin(πx_j) dx = 1 over [0, 1]^d. Monte Carlo's error falls as n^{−1/2} in every dimension; randomly shifted Sobol and Halton points (8 shifts, which also give a standard error) fall nearly as n⁻¹ in low dimension and lose their edge as d grows."
    >
      <XYChart series={series} xLog yLog xLabel="evaluations n" yLabel="error" />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 4. Gaussian nodes and weights.

export function GaussNodesSpecimen() {
  const [family, setFamily] = useState<'legendre' | 'hermite' | 'laguerre'>('legendre')
  const [n, setN] = useState(8)
  const { nodes, weights, check } = useMemo(() => {
    const rule = family === 'legendre' ? gaussLegendre(n) : family === 'hermite' ? gaussHermite(n) : gaussLaguerre(n)
    const x = toFlat(rule.nodes)
    const w = toFlat(rule.weights)
    // ∫ w(x) x^{2n−2} dx against the exact moment, a check that the rule integrates high degrees exactly.
    const k = 2 * n - 2
    const estimate = w.reduce((s, wi, i) => s + wi * x[i] ** k, 0)
    const exact = family === 'legendre' ? 2 / (k + 1) : family === 'hermite' ? Math.exp(lgammaHalf(k)) : factorial(k)
    return { nodes: x, weights: w, check: Math.abs(estimate - exact) / exact }
  }, [family, n])
  return (
    <Figure
      title="Gaussian quadrature nodes and weights"
      controls={
        <>
          <Select
            label="rule"
            value={family}
            onChange={setFamily}
            options={[
              { value: 'legendre', label: 'Gauss–Legendre, weight 1 on [−1, 1]' },
              { value: 'hermite', label: 'Gauss–Hermite, weight e^{−x²}' },
              { value: 'laguerre', label: 'Gauss–Laguerre, weight e^{−x} on [0, ∞)' },
            ]}
          />
          <Slider label="points n" value={n} min={1} max={40} step={1} onChange={setN} />
        </>
      }
      readouts={<Readout label={`relative error of the degree-${2 * n - 2} moment`} value={check.toExponential(2)} />}
      caption="Nodes are the roots of the n-th orthogonal polynomial; weights shrink towards the ends (and, for Hermite and Laguerre, underflow far out). An n-point rule integrates every polynomial of degree ≤ 2n − 1 exactly against its weight."
    >
      <XYChart
        yLog={family !== 'legendre'}
        xLabel="node xᵢ"
        yLabel="weight wᵢ"
        series={[{ name: 'weights', type: 'bar', x: nodes, y: weights, slot: 0 }]}
      />
    </Figure>
  )
}

/** Γ((k + 1)/2) as a log, for the Hermite moments ∫ e^{−x²} x^k dx = Γ((k + 1)/2) for even k. */
function lgammaHalf(k: number): number {
  // Γ(m + ½) = (2m)! √π / (4^m m!), with m = k/2.
  const m = k / 2
  let log = 0.5 * Math.log(Math.PI)
  for (let j = 1; j <= m; j++) log += Math.log(j - 0.5)
  return log
}

function factorial(k: number): number {
  let p = 1
  for (let j = 2; j <= k; j++) p *= j
  return p
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
    <Figure title="Infinite limits and error estimates" hoverReadout={false}>
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
