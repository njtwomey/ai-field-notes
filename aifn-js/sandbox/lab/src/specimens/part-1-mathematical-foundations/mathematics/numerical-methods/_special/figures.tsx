import { grad, gradCheck } from 'aifn/foundation/autodiff'
import { entropy } from 'aifn/probability/information'
import * as S from 'aifn/numerics/special'
import { linspace, logsumexp, sum, tensor, toFlat, type Tensor, type Unary, type Value } from 'aifn/foundation/tensor'
import { useMemo } from 'react'
import { Equation, Figure, live, tex } from '@lab/layout'
import { choice, slider, useFigureState } from '@lab/state'
import { Bars, Curve, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'
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

// ── softplus ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const SOFTPLUS_X = grid(-10, 10, 201)
const SOFTPLUS_Y = toFlat(S.softplus(linspace(-10, 10, 201)))

export function SoftplusSpecimen() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'softplus(x)' })
  return (
    <Figure
      title="softplus(x)"
      purpose="softplus(x) = log(1 + eˣ) is evaluated as max(x, 0) + log1p(e^(−|x|)), so it stays finite and accurate where eˣ overflows or 1 + eˣ rounds to 1."
      readouts={
        <>
          <Readout label="softplus(800)" value={formatValue(S.softplus(800) as number)} />
          <Readout label="log(1 + e⁸⁰⁰) (naive)" value={formatValue(Math.log(1 + Math.exp(800)))} />
          <Readout label="softplus(−40)" value={formatValue(S.softplus(-40) as number)} />
          <Readout label="log(1 + e⁻⁴⁰) (naive)" value={formatValue(Math.log(1 + Math.exp(-40)))} />
        </>
      }
      caption="For large x, softplus(x) ≈ x; for very negative x, softplus(x) ≈ eˣ, which the naive form loses entirely once 1 + eˣ rounds to 1."
    >
      <Plot x={x} y={y}>
        <Curve name="softplus" x={SOFTPLUS_X} y={SOFTPLUS_Y} />
      </Plot>
    </Figure>
  )
}

// ── Normal tail ──────────────────────────────────────────────────────────────────────────────────────────────────────

const TAIL_Z = grid(-40, 0, 401)
const ERROR_Z = grid(-8, 0, 161)

export function NormalTailSpecimen() {
  const curves = useMemo(() => {
    const log10 = (v: number) => finite(Math.log10(v))
    return {
      logCdf: TAIL_Z.map((z) => S.normalLogCdf(z) / Math.LN10),
      cdf: TAIL_Z.map((z) => log10(S.normalCdf(z))),
      old: TAIL_Z.map((z) => log10(naiveNormalCdf(z))),
      error: ERROR_Z.map((z) => {
        const r = Math.abs(naiveNormalCdf(z) / Math.exp(S.normalLogCdf(z)) - 1)
        return r > 0 && Number.isFinite(r) ? r : NaN
      }),
    }
  }, [])
  const z1 = useAxis({ label: 'z' })
  const y1 = useAxis({ label: 'log₁₀ Φ(z)' })
  const z2 = useAxis({ label: 'z' })
  const y2 = useAxis({ label: 'relative error', log: true })
  return (
    <Figure
      title="log₁₀ Φ(z) in the far tail, and the old Φ's relative error"
      purpose="normalLogCdf never underflows and normalCdf keeps full relative accuracy until it underflows near z = −37.5; the old A&S 7.1.26 formula, accurate only in absolute terms, loses relative accuracy steadily in the tail."
      defaultSize="L"
      readouts={
        <>
          <Readout label="Φ(−37)" value={formatValue(S.normalCdf(-37))} />
          <Readout label="log Φ(−1e5)" value={formatValue(S.normalLogCdf(-1e5))} />
          <Readout label="old Φ(−6)" value={formatValue(naiveNormalCdf(-6))} />
          <Readout label="Φ(−6)" value={formatValue(S.normalCdf(-6))} />
        </>
      }
      caption="Top: log₁₀ Φ on [−40, 0]; the curves overlap wherever they are defined, and the dashed normalCdf ends where Φ underflows below the smallest double. Bottom: the old formula's relative error, from about 10⁻⁷ near 0 to about 10⁻² at z = −8: its error bound is absolute, so it says less and less about a small Φ."
    >
      <Plots rows={2} heights={[3, 2]}>
        <Plot x={z1} y={y1}>
          <Curve name="log₁₀ Φ(z) from normalLogCdf" x={TAIL_Z} y={curves.logCdf} slot={0} />
          <Curve name="log₁₀ normalCdf(z)" x={TAIL_Z} y={curves.cdf} slot={1} dashed />
          <Curve name="log₁₀ of the old A&S 7.1.26 Φ" x={TAIL_Z} y={curves.old} slot={2} />
        </Plot>
        <Plot x={z2} y={y2}>
          <Curve name="relative error of the old Φ" x={ERROR_Z} y={curves.error} slot={2} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

const TRUNC_T = grid(-30, 8, 381)

export function TruncatedMomentsSpecimen() {
  const curves = useMemo(() => {
    // The same quantities from a naive ratio φ/Φ with the old Φ, which fails where Φ loses accuracy.
    const naiveV = (t: number) => S.normalPdf(t) / naiveNormalCdf(t)
    return {
      v: TRUNC_T.map((t) => S.truncatedNormalV(t) + t),
      w: toFlat(S.truncatedNormalW(tensor(TRUNC_T))),
      naive: TRUNC_T.map((t) => finite(naiveV(t) * (naiveV(t) + t))),
    }
  }, [])
  const x = useAxis({ label: 't' })
  const y = useAxis({ label: 'value', range: [-0.1, 1.2] })
  return (
    <Figure
      title="v(t) + t and w(t)"
      purpose="The truncated-normal moment functions v = φ/Φ and w = v(v + t) come from a Mills-ratio continued fraction, so they stay smooth far into the tail where the ratio φ/Φ breaks down."
      readouts={
        <>
          <Readout label="v(−30)" value={formatValue(S.truncatedNormalV(-30))} />
          <Readout label="w(−30)" value={formatValue(S.truncatedNormalW(-30))} />
          <Readout label="v_draw(3, ε = 0.5)" value={formatValue(S.truncatedNormalVDraw(3, 0.5))} />
          <Readout label="w_draw(3, ε = 0.5)" value={formatValue(S.truncatedNormalWDraw(3, 0.5))} />
        </>
      }
      caption="v(t) + t tends to 0 and w(t) to 1 as t → −∞ (v(t) ≈ −t there). The dashed naive w, computed from the old Φ, drifts below w from about t = −3 and blows up near t = −8. These functions update the moments in expectation propagation for probit factors (TrueSkill)."
    >
      <Plot x={x} y={y}>
        <Curve name="v(t) + t" x={TRUNC_T} y={curves.v} slot={0} />
        <Curve name="w(t)" x={TRUNC_T} y={curves.w} slot={1} />
        <Curve name="w(t), naive φ/Φ with the old Φ" x={TRUNC_T} y={curves.naive} slot={2} dashed />
      </Plot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

const GAMMA_X = grid(-4.5, 6, 1051)

export function GammaFamilySpecimen() {
  const curves = useMemo(() => {
    const clip = (v: number) => (Math.abs(v) > 12 ? NaN : v)
    return {
      logGamma: GAMMA_X.map((x) => clip(S.logGamma(x))),
      digamma: GAMMA_X.map((x) => clip(S.digamma(x))),
      trigamma: GAMMA_X.map((x) => clip(S.trigamma(x))),
    }
  }, [])
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'value', range: [-10, 12] })
  return (
    <Figure
      title="log |Γ(x)|, ψ(x) and ψ₁(x)"
      purpose="ψ = (log Γ)′ and ψ₁ = ψ′ are defined across the poles of Γ at 0, −1, −2, …; negative x is reached by the reflection formula."
      caption="Each function is cut off where |value| > 12 so the poles show as gaps. ψ increases between poles and crosses zero once in each interval; ψ₁ is positive everywhere it is defined."
    >
      <Plot x={x} y={y}>
        <Curve name="log |Γ(x)|" x={GAMMA_X} y={curves.logGamma} slot={0} />
        <Curve name="ψ(x)" x={GAMMA_X} y={curves.digamma} slot={1} />
        <Curve name="ψ₁(x)" x={GAMMA_X} y={curves.trigamma} slot={2} />
      </Plot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

const INCOMPLETE = [
  { value: 'gamma', label: 'P(a, x)' },
  { value: 'beta', label: 'I_x(a, b)' },
  { value: 'student', label: 'Student t quantile' },
] as const

const PURPOSES = {
  gamma: 'P(a, x) is the gamma(a) cdf at x: it rises from 0 to 1 near x = a, more steeply as a grows.',
  beta: 'I_x(a, b) is the beta(a, b) cdf at x: its shape follows where the beta density puts its mass.',
  student:
    'The t quantile is the inverse of the incomplete-beta t cdf: heavy tails at small ν, the normal quantile at ν = ∞.',
}

export function IncompleteGammaBetaSpecimen() {
  const state = useFigureState({ which: choice(INCOMPLETE, 'gamma', { label: 'function' }) })
  const which = state.which
  const curves = useMemo((): { name: string; x: number[]; y: number[] }[] => {
    if (which === 'gamma') {
      const xs = grid(0, 20, 401)
      return [0.5, 1, 3, 10].map((a) => ({ name: `P(${a}, x)`, x: xs, y: xs.map((x) => S.regularisedGammaP(a, x)) }))
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
      ).map(([a, b]) => ({ name: `I_x(${a}, ${b})`, x: xs, y: xs.map((x) => S.regularisedBeta(a, b, x)) }))
    }
    const ps = grid(0.001, 0.999, 400)
    return [0.5, 1, 3, 30, Infinity].map((v) => ({
      name: `t quantile, ν = ${v === Infinity ? '∞' : v}`,
      x: ps,
      y: ps.map((p) => finite(S.studentTQuantile(p, v))),
    }))
  }, [which])
  const x = useAxis({ label: which === 'student' ? 'p' : 'x', key: which })
  const y = useAxis({
    label: which === 'student' ? 'quantile' : 'cdf',
    range: which === 'student' ? [-8, 8] : [0, 1],
    key: which,
  })
  return (
    <Figure title="Regularised incomplete functions and t quantiles" purpose={PURPOSES[which]} state={state}>
      <Plot x={x} y={y}>
        {curves.map((c, i) => (
          <Curve key={c.name} name={c.name} x={c.x} y={c.y} slot={i} />
        ))}
      </Plot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

const LOGITS = [2, 1, 0.5, -1, -3]
const LOGIT_NAMES = LOGITS.map((l) => `x = ${l}`)
const LOGIT_INDEX = LOGITS.map((_, i) => i)

export function SoftmaxSpecimen() {
  const state = useFigureState({
    logT: slider(-3, 3, -0.5, {
      label: 'log T',
      format: (v) => `${v.toFixed(2)} (T = ${Math.exp(v).toPrecision(3)})`,
    }),
  })
  const temperature = Math.exp(state.logT)
  const p = useMemo(() => toFlat(S.softmax(tensor(LOGITS), { temperature })), [temperature])
  const lse = logsumexp(tensor(LOGITS.map((x) => x / temperature))) as number
  const x = useAxis({ label: 'class', categories: LOGIT_NAMES })
  const y = useAxis({ label: 'probability', range: [0, 1] })
  return (
    <Figure
      title="softmax(x/T) of five logits"
      purpose="Dividing the logits by a temperature T sharpens softmax towards one-hot at the largest logit as T → 0 and flattens it towards uniform as T → ∞; the order never changes."
      state={state}
      equation={
        <Equation>
          {tex`p_1 = \frac{e^{x_1/T}}{\sum_j e^{x_j/T}} = \exp\Big(\frac{2}{${live(temperature, { digits: 3 })}} - ${live(lse, { digits: 4 })}\Big) = ${live(p[0], { digits: 4, strong: true })}`}
        </Equation>
      }
      readouts={
        <Readout
          label="entropy (nats; uniform log 5 = 1.609)"
          value={formatValue(entropy(S.softmax(tensor(LOGITS), { temperature })) as number)}
        />
      }
      caption="The denominator is computed as logsumexp(x/T), so a tiny T, which makes x/T huge, never overflows. Slide log T left to concentrate the mass on x = 2, right to spread it out."
    >
      <Plot x={x} y={y}>
        <Bars name="softmax(x/T)" x={LOGIT_INDEX} y={p} />
      </Plot>
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
const PRIMITIVES = Object.keys(DOMAINS) as PrimitiveName[]

export function DerivativeKernelsSpecimen() {
  const state = useFigureState({ name: choice(PRIMITIVES, 'truncatedNormalV', { label: 'primitive' }) })
  const name = state.name as PrimitiveName
  const r = useMemo(() => {
    const f = unaries[name]
    const [lo, hi] = DOMAINS[name]
    // One call on the whole grid (a tensor), and one backward sweep for f′ at every grid point: for an elementwise f,
    // the gradient of Σ f(x) is f′ at each element.
    const sumF = (v: Value) => sum(f(v))
    const derivative = grad(sumF)
    const x = linspace(lo, hi, 241)
    // gradCheck compares the tape's gradient with central differences, element by element, at interior grid points.
    const fdx = grid(lo, hi, 25).slice(1, -1)
    const check = gradCheck(sumF, tensor(fdx), { eps: 1e-5 })
    return {
      xs: toFlat(x),
      ys: toFlat(f(x)).map(finite),
      dys: toFlat(derivative(x) as Tensor).map(finite),
      fdx,
      fdy: check.entries.map((e) => e.numeric),
      worst: check.maxRelError,
    }
  }, [name])
  const x = useAxis({ label: 'x', key: name })
  const y = useAxis({ label: 'value', key: name })
  return (
    <Figure
      title="A primitive and its derivative"
      purpose="Every one-argument primitive carries its derivative rule: grad of Σ f(x) over a grid tensor gives f′ at every point in one backward sweep, matching central differences."
      description={`${name} on [${DOMAINS[name].join(', ')}]`}
      id="primitive-derivative"
      state={state}
      readouts={
        <Readout label="gradCheck: largest relative gap, tape vs central difference" value={formatValue(r.worst)} />
      }
      caption="The dots are central differences at 23 interior points; they sit on the tape's derivative curve. Pick another primitive to check its rule."
    >
      <Plot x={x} y={y}>
        <Curve name={name} x={r.xs} y={r.ys} slot={0} />
        <Curve name={`d${name}/dx (tape)`} x={r.xs} y={r.dys} slot={1} />
        <Points name="central difference" x={r.fdx} y={r.fdy} slot={1} />
      </Plot>
    </Figure>
  )
}
