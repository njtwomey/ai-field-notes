import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'

type FnId = 'exp' | 'sqrt' | 'periodic' | 'runge'

type Integrand = { label: string; f: (x: number) => number; a: number; b: number; exact: number }

const INTEGRANDS: Record<FnId, Integrand> = {
  exp: { label: 'eˣ on [0, 1]', f: Math.exp, a: 0, b: 1, exact: Math.E - 1 },
  sqrt: { label: '√x on [0, 1]', f: Math.sqrt, a: 0, b: 1, exact: 2 / 3 },
  periodic: {
    label: 'e^cos(2πx) on [0, 1]',
    f: (x) => Math.exp(Math.cos(2 * Math.PI * x)),
    a: 0,
    b: 1,
    // I₀(1), the modified Bessel function of the first kind at 1.
    exact: 1.2660658777520082,
  },
  runge: { label: '1/(1 + 25x²) on [−1, 1]', f: (x) => 1 / (1 + 25 * x * x), a: -1, b: 1, exact: 0.4 * Math.atan(5) },
}

/** Panels m = 2^k for k = 1..10; every rule below uses m + 1 function evaluations. */
const KS = Array.from({ length: 10 }, (_, i) => i + 1)
const FLOOR = 1e-17

/** Gauss–Legendre nodes and weights on [−1, 1], by Newton's method on the Legendre polynomial P_n. */
function gaussLegendre(n: number): { x: number[]; w: number[] } {
  const x: number[] = []
  const w: number[] = []
  for (let i = 1; i <= n; i++) {
    let z = Math.cos((Math.PI * (i - 0.25)) / (n + 0.5))
    let dp = 1
    for (let it = 0; it < 100; it++) {
      // Three-term recurrence for P_n(z), then P_n'(z) from P_n and P_{n−1}.
      let p0 = 1
      let p1 = z
      for (let k = 2; k <= n; k++) [p0, p1] = [p1, ((2 * k - 1) * z * p1 - (k - 1) * p0) / k]
      dp = (n * (z * p1 - p0)) / (z * z - 1)
      const step = p1 / dp
      z -= step
      if (Math.abs(step) < 1e-16) break
    }
    x.push(z)
    w.push(2 / ((1 - z * z) * dp * dp))
  }
  return { x, w }
}

function trapezoid(g: Integrand, m: number): number {
  const h = (g.b - g.a) / m
  let s = 0.5 * (g.f(g.a) + g.f(g.b))
  for (let j = 1; j < m; j++) s += g.f(g.a + j * h)
  return h * s
}

function simpson(g: Integrand, m: number): number {
  const h = (g.b - g.a) / m
  let s = g.f(g.a) + g.f(g.b)
  for (let j = 1; j < m; j++) s += (j % 2 ? 4 : 2) * g.f(g.a + j * h)
  return (h / 3) * s
}

function gauss(g: Integrand, rule: { x: number[]; w: number[] }): number {
  const half = (g.b - g.a) / 2
  const mid = (g.a + g.b) / 2
  return half * rule.x.reduce((s, z, i) => s + rule.w[i] * g.f(half * z + mid), 0)
}

/** Error against the number of function evaluations for the trapezoid rule, Simpson's rule and Gauss–Legendre. */
export function QuadratureErrors() {
  const [id, setId] = useState<FnId>('exp')
  const rules = useMemo(() => KS.map((k) => gaussLegendre(2 ** k + 1)), [])
  const g = INTEGRANDS[id]

  const { series, errs } = useMemo(() => {
    const err = (v: number) => Math.max(FLOOR, Math.abs(v - g.exact))
    const t = KS.map((k) => err(trapezoid(g, 2 ** k)))
    const s = KS.map((k) => err(simpson(g, 2 ** k)))
    const q = KS.map((_, i) => err(gauss(g, rules[i])))
    const out: XYSeries[] = [
      { name: 'trapezoid', type: 'line', x: KS, y: t, slot: 0 },
      { name: 'Simpson', type: 'line', x: KS, y: s, slot: 1 },
      { name: 'Gauss–Legendre', type: 'line', x: KS, y: q, slot: 2 },
    ]
    return { series: out, errs: { t: t[2], s: s[2], q: q[2] } }
  }, [g, rules])

  return (
    <Interactive
      title="Quadrature error against the number of evaluations"
      caption="Each rule uses 2ᵏ + 1 evaluations of f. On a log scale, the trapezoid error falls by 4 per doubling (slope h²) and Simpson's by 16 (h⁴) for smooth integrands; Gauss–Legendre falls faster than any power. √x has an infinite derivative at 0, which slows every rule. For a smooth periodic integrand over a full period the plain trapezoid rule is the best of the three."
      controls={
        <ParamChoice
          label="integrand"
          value={id}
          onChange={setId}
          options={(Object.keys(INTEGRANDS) as FnId[]).map((k) => ({ value: k, label: INTEGRANDS[k].label }))}
        />
      }
      readout={
        <>
          <Readout label="exact" value={g.exact.toPrecision(12)} />
          <Readout label="errors with 9 evaluations: trapezoid" value={formatNumber(errs.t)} />
          <Readout label="Simpson" value={formatNumber(errs.s)} />
          <Readout label="Gauss" value={formatNumber(errs.q)} />
        </>
      }
    >
      <XYChart
        height={320}
        xLabel="k (2ᵏ + 1 evaluations)"
        yLabel="absolute error"
        series={series}
        xRange={[1, 10]}
        yLog
        yRange={[FLOOR, 1]}
      />
    </Interactive>
  )
}
