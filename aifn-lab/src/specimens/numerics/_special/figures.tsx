import { grad, gradCheck } from 'aifn/foundation/autodiff'
import { entropy } from 'aifn/probability/information'
import * as S from 'aifn/numerics/special'
import { linspace, logsumexp, sum, tensor, toFlat, type Tensor, type Unary, type Value } from 'aifn/foundation/tensor'
import { useMemo, useState } from 'react'
import { Choice, Select, Slider } from '@lab/controls'
import { Figure } from '@lab/layout'
import { ChartSize, Readout, XYChart, type XYSeries } from '@lab/viz'
import { formatValue } from '@lab/views'

/** n evenly spaced points from lo to hi, as numbers. */
const grid = (lo: number, hi: number, n: number) => toFlat(linspace(lo, hi, n))

/** The site's previous normal cdf: 0.5(1 + erf(z/√2)) with Abramowitz and Stegun 7.1.26 (absolute error 1.5e-7). */
function naiveNormalCdf(z: number): number {
  const x = z / Math.SQRT2
  const a = Math.abs(x)
  const t = 1 / (1 + 0.3275911 * a)
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a)
  return 0.5 * (1 + Math.sign(x) * y)
}

/** Replace non-finite values by NaN so lines break there instead of drawing to infinity. */
const finite = (v: number) => (Number.isFinite(v) ? v : NaN)

export function NormalTailSpecimen() {
  const zs = useMemo(() => grid(-40, 0, 401), [])
  const series = useMemo((): XYSeries[] => {
    const log10 = (v: number) => finite(Math.log10(v))
    return [
      {
        name: 'log₁₀ Φ(z) from normalLogCdf',
        type: 'line',
        x: zs,
        y: zs.map((z) => S.normalLogCdf(z) / Math.LN10),
        slot: 0,
      },
      {
        name: 'log₁₀ normalCdf(z)',
        type: 'line',
        x: zs,
        y: zs.map((z) => log10(S.normalCdf(z))),
        slot: 1,
        dashed: true,
      },
      {
        name: 'log₁₀ of the old A&S 7.1.26 Φ',
        type: 'line',
        x: zs,
        y: zs.map((z) => log10(naiveNormalCdf(z))),
        slot: 2,
      },
    ]
  }, [zs])
  const errors = useMemo((): XYSeries[] => {
    const e = (z: number) => {
      const exact = Math.exp(S.normalLogCdf(z))
      const r = Math.abs(naiveNormalCdf(z) / exact - 1)
      return r > 0 && Number.isFinite(r) ? r : NaN
    }
    const zs2 = grid(-8, 0, 161)
    return [{ name: 'relative error of the old Φ', type: 'line', x: zs2, y: zs2.map(e), slot: 2 }]
  }, [])
  return (
    <Figure
      title="log₁₀ Φ(z) in the far tail, and the old Φ's relative error"
      defaultSize="L"
      readouts={
        <>
          <Readout label="Φ(−37)" value={formatValue(S.normalCdf(-37))} />
          <Readout label="log Φ(−1e5)" value={formatValue(S.normalLogCdf(-1e5))} />
          <Readout label="old Φ(−6)" value={formatValue(naiveNormalCdf(-6))} />
          <Readout label="Φ(−6)" value={formatValue(S.normalCdf(-6))} />
        </>
      }
    >
      <div className="flex flex-col gap-4">
        <ChartSize scale={0.6}>
          <XYChart series={series} xLabel="z" yLabel="log₁₀ Φ(z)" hoverGroup="normal-tail" />
        </ChartSize>
        <ChartSize scale={0.4}>
          <XYChart series={errors} xLabel="z" yLabel="relative error" yLog hoverGroup="normal-tail" />
        </ChartSize>
      </div>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

export function TruncatedMomentsSpecimen() {
  const ts = useMemo(() => grid(-30, 8, 381), [])
  const series = useMemo((): XYSeries[] => {
    // The same quantities from a naive ratio φ/Φ with the old Φ, which fails where Φ loses accuracy.
    const naiveV = (t: number) => S.normalPdf(t) / naiveNormalCdf(t)
    return [
      { name: 'v(t) + t', type: 'line', x: ts, y: ts.map((t) => S.truncatedNormalV(t) + t), slot: 0 },
      { name: 'w(t)', type: 'line', x: ts, y: toFlat(S.truncatedNormalW(tensor(ts))), slot: 1 },
      {
        name: 'w(t), naive φ/Φ with the old Φ',
        type: 'line',
        x: ts,
        y: ts.map((t) => finite(naiveV(t) * (naiveV(t) + t))),
        slot: 2,
        dashed: true,
      },
    ]
  }, [ts])
  return (
    <Figure
      title="v(t) + t and w(t)"
      readouts={
        <>
          <Readout label="v(−30)" value={formatValue(S.truncatedNormalV(-30))} />
          <Readout label="w(−30)" value={formatValue(S.truncatedNormalW(-30))} />
          <Readout label="v_draw(3, ε = 0.5)" value={formatValue(S.truncatedNormalVDraw(3, 0.5))} />
          <Readout label="w_draw(3, ε = 0.5)" value={formatValue(S.truncatedNormalWDraw(3, 0.5))} />
        </>
      }
    >
      <XYChart series={series} xLabel="t" yRange={[-0.1, 1.2]} />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

export function GammaFamilySpecimen() {
  const xs = useMemo(() => grid(-4.5, 6, 1051), [])
  const series = useMemo((): XYSeries[] => {
    const clip = (v: number) => (Math.abs(v) > 12 ? NaN : v)
    return [
      { name: 'log |Γ(x)|', type: 'line', x: xs, y: xs.map((x) => clip(S.logGamma(x))), slot: 0 },
      { name: 'ψ(x)', type: 'line', x: xs, y: xs.map((x) => clip(S.digamma(x))), slot: 1 },
      { name: 'ψ₁(x)', type: 'line', x: xs, y: xs.map((x) => clip(S.trigamma(x))), slot: 2 },
    ]
  }, [xs])
  return (
    <Figure title="log |Γ(x)|, ψ(x) and ψ₁(x)">
      <XYChart series={series} xLabel="x" yRange={[-10, 12]} />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

export function IncompleteGammaBetaSpecimen() {
  const [which, setWhich] = useState<'gamma' | 'beta' | 'student'>('gamma')
  const series = useMemo((): XYSeries[] => {
    if (which === 'gamma') {
      const xs = grid(0, 20, 401)
      return [0.5, 1, 3, 10].map((a, i) => ({
        name: `P(${a}, x)`,
        type: 'line',
        x: xs,
        y: xs.map((x) => S.regularisedGammaP(a, x)),
        slot: i,
      }))
    }
    if (which === 'beta') {
      const xs = grid(0, 1, 401)
      return (
        [
          [0.5, 0.5],
          [2, 5],
          [5, 1],
          [30, 30],
        ] as const
      ).map(([a, b], i) => ({
        name: `I_x(${a}, ${b})`,
        type: 'line',
        x: xs,
        y: xs.map((x) => S.regularisedBeta(a, b, x)),
        slot: i,
      }))
    }
    const ps = grid(0.001, 0.999, 400)
    return [0.5, 1, 3, 30, Infinity].map((v, i) => ({
      name: `t quantile, ν = ${v === Infinity ? '∞' : v}`,
      type: 'line',
      x: ps,
      y: ps.map((p) => finite(S.studentTQuantile(p, v))),
      slot: i,
    }))
  }, [which])
  return (
    <Figure
      title="Regularised incomplete functions and t quantiles"
      controls={
        <Select
          label="function"
          value={which}
          onChange={setWhich}
          options={[
            { value: 'gamma', label: 'P(a, x)' },
            { value: 'beta', label: 'I_x(a, b)' },
            { value: 'student', label: 'Student t quantile' },
          ]}
        />
      }
    >
      <XYChart
        series={series}
        xLabel={which === 'student' ? 'p' : 'x'}
        yRange={which === 'student' ? [-8, 8] : undefined}
      />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

const LOGITS = [2, 1, 0.5, -1, -3]

export function SoftmaxSpecimen() {
  const [logT, setLogT] = useState(0)
  const temperature = Math.exp(logT)
  const series = useMemo((): XYSeries[] => {
    const p = toFlat(S.softmax(tensor(LOGITS), { temperature }))
    return [{ name: 'softmax(x/T)', type: 'bar', x: LOGITS.map((_, i) => i + 1), y: p }]
  }, [temperature])
  return (
    <Figure
      title="softmax(x/T) of five logits"
      controls={
        <Slider
          label="log T"
          value={logT}
          min={-3}
          max={3}

          onChange={setLogT}
          format={(v) => `${v.toFixed(2)} (T = ${Math.exp(v).toPrecision(3)})`}
        />
      }
      readouts={
        <>
          <Readout label="logSumExp(x/T)" value={formatValue(logsumexp(tensor(LOGITS.map((x) => x / temperature))))} />
          <Readout
            label="entropy (nats)"
            value={formatValue(entropy(S.softmax(tensor(LOGITS), { temperature })) as number)}
          />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="class (logits 2, 1, 0.5, −1, −3)"
        yLabel="probability"
        yRange={[0, 1]}
        integerX
      />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

/** Plot ranges for the derivative specimen, inside each function's domain. */
const DOMAINS = {
  erf: [-3, 3],
  erfc: [-3, 3],
  erfcx: [-2, 6],
  logErfc: [-3, 10],
  erfinv: [-0.99, 0.99],
  erfcinv: [0.01, 1.99],
  normalPdf: [-4, 4],
  normalLogPdf: [-4, 4],
  normalCdf: [-4, 4],
  normalLogCdf: [-10, 4],
  normalQuantile: [0.005, 0.995],
  truncatedNormalV: [-6, 6],
  truncatedNormalW: [-6, 6],
  logGamma: [0.1, 6],
  gamma: [0.2, 5],
  digamma: [0.2, 6],
  trigamma: [0.3, 6],
  logFactorial: [0, 10],
  softplus: [-6, 6],
  sigmoid: [-6, 6],
  logSigmoid: [-6, 6],
  logit: [0.01, 0.99],
  log1mexp: [-6, -0.02],
  logExpm1: [0.02, 6],
  log1pmx: [-0.95, 4],
  binaryEntropy: [0.005, 0.995],
} satisfies Record<string, [number, number]>
type PrimitiveName = keyof typeof DOMAINS
const unaries = S as unknown as Record<PrimitiveName, Unary>

export function DerivativeKernelsSpecimen() {
  const [name, setName] = useState<PrimitiveName>('truncatedNormalV')
  const { series, worst } = useMemo(() => {
    const f = unaries[name]
    const [lo, hi] = DOMAINS[name]
    // One call on the whole grid (a tensor), and one backward sweep for f′ at every grid point: for an elementwise f,
    // the gradient of Σ f(x) is f′ at each element.
    const sumF = (v: Value) => sum(f(v))
    const derivative = grad(sumF)
    const x = linspace(lo, hi, 241)
    const ys = toFlat(f(x))
    const dys = toFlat(derivative(x) as Tensor)
    // gradCheck compares the tape's gradient with central differences, element by element, at interior grid points.
    const fdx = grid(lo, hi, 25).slice(1, -1)
    const check = gradCheck(sumF, tensor(fdx), { eps: 1e-5 })
    const xs = toFlat(x)
    const out: XYSeries[] = [
      { name, type: 'line', x: xs, y: ys.map(finite), slot: 0 },
      { name: `d${name}/dx (tape)`, type: 'line', x: xs, y: dys.map(finite), slot: 1 },
      { name: 'central difference', type: 'scatter', x: fdx, y: check.entries.map((e) => e.numeric), slot: 1 },
    ]
    return { series: out, worst: check.maxRelError }
  }, [name])
  return (
    <Figure
      title={`${name} and its derivative`}
      id="primitive-derivative"
      controls={
        <Choice label="primitive" value={name} onChange={setName} options={Object.keys(DOMAINS) as PrimitiveName[]} />
      }
      readouts={
        <Readout label="gradCheck: largest relative gap, tape vs central difference" value={formatValue(worst)} />
      }
    >
      <XYChart series={series} xLabel="x" />
    </Figure>
  )
}
