import * as D from 'aifn/probability/distributions'
import { fromRows, linspace, tensor, toFlat, unwrap, type Value } from 'aifn/foundation/tensor'
import { useMemo } from 'react'
import { Equation, Figure, live, tex } from '@lab/layout'
import { row, slider, useFigureState, useProbe, variants, type SliderDef } from '@lab/state'
import { Curve, Plot, Plots, Probe, Raster, Readout, useAxis } from '@lab/viz'
import { DistributionPanel, distributionRange, formatValue } from '@lab/views'

/** A Value as numbers. */
const numbers = (v: Value): number[] => {
  const r = unwrap(v)
  return typeof r === 'number' ? [r] : toFlat(r)
}
const num = (v: Value) => numbers(v)[0]
const grid = (lo: number, hi: number, n: number) => toFlat(linspace(lo, hi, n))

type ParamSpec = { label: string; min: number; max: number; value: number; step?: number }
type Family = { params: ParamSpec[]; make: (...v: number[]) => D.Univariate }

const FAMILIES: Record<string, Family> = {
  Normal: {
    params: [
      { label: 'loc μ', min: -3, max: 3, value: 0 },
      { label: 'scale σ', min: 0.1, max: 3, value: 1 },
    ],
    make: (m, s) => D.Normal(m, s),
  },
  LogNormal: {
    params: [
      { label: 'μ', min: -1, max: 1.5, value: 0 },
      { label: 'σ', min: 0.1, max: 1.5, value: 0.5 },
    ],
    make: (m, s) => D.LogNormal(m, s),
  },
  StudentT: {
    params: [
      { label: 'df ν', min: 0.5, max: 30, value: 3, step: 0.1 },
      { label: 'loc μ', min: -3, max: 3, value: 0 },
      { label: 'scale σ', min: 0.2, max: 3, value: 1 },
    ],
    make: (v, m, s) => D.StudentT(v, m, s),
  },
  Cauchy: {
    params: [
      { label: 'loc', min: -3, max: 3, value: 0 },
      { label: 'scale', min: 0.1, max: 3, value: 1 },
    ],
    make: (m, s) => D.Cauchy(m, s),
  },
  Laplace: {
    params: [
      { label: 'loc', min: -3, max: 3, value: 0 },
      { label: 'scale b', min: 0.1, max: 3, value: 1 },
    ],
    make: (m, s) => D.Laplace(m, s),
  },
  Logistic: {
    params: [
      { label: 'loc', min: -3, max: 3, value: 0 },
      { label: 'scale s', min: 0.1, max: 3, value: 1 },
    ],
    make: (m, s) => D.Logistic(m, s),
  },
  Uniform: {
    params: [
      { label: 'low', min: -3, max: 0, value: -1 },
      { label: 'high', min: 0.1, max: 4, value: 2 },
    ],
    make: (a, b) => D.Uniform(a, b),
  },
  Exponential: { params: [{ label: 'rate λ', min: 0.1, max: 5, value: 1 }], make: (r) => D.Exponential(r) },
  Gamma: {
    params: [
      { label: 'shape α', min: 0.2, max: 12, value: 2 },
      { label: 'rate β', min: 0.2, max: 5, value: 1 },
    ],
    make: (a, b) => D.Gamma(a, b),
  },
  InverseGamma: {
    params: [
      { label: 'shape α', min: 0.5, max: 12, value: 3 },
      { label: 'scale β', min: 0.2, max: 5, value: 2 },
    ],
    make: (a, b) => D.InverseGamma(a, b),
  },
  Beta: {
    params: [
      { label: 'a', min: 0.2, max: 10, value: 2 },
      { label: 'b', min: 0.2, max: 10, value: 5 },
    ],
    make: (a, b) => D.Beta(a, b),
  },
  ChiSquare: { params: [{ label: 'df k', min: 0.5, max: 30, value: 4, step: 0.1 }], make: (k) => D.ChiSquare(k) },
  Weibull: {
    params: [
      { label: 'shape k', min: 0.3, max: 6, value: 1.5 },
      { label: 'scale λ', min: 0.2, max: 4, value: 1 },
    ],
    make: (k, l) => D.Weibull(k, l),
  },
  Gumbel: {
    params: [
      { label: 'loc', min: -3, max: 3, value: 0 },
      { label: 'scale', min: 0.1, max: 3, value: 1 },
    ],
    make: (m, s) => D.Gumbel(m, s),
  },
  VonMises: {
    params: [
      { label: 'loc μ', min: -3, max: 3, value: 0 },
      { label: 'concentration κ', min: 0, max: 30, value: 2 },
    ],
    make: (m, k) => D.VonMises(m, k),
  },
  TruncatedNormal: {
    params: [
      { label: 'loc μ', min: -3, max: 3, value: 0 },
      { label: 'low (high = low + 2)', min: -4, max: 6, value: 1 },
    ],
    make: (m, lo) => D.TruncatedNormal(m, 1, lo, lo + 2),
  },
  Bernoulli: { params: [{ label: 'p', min: 0, max: 1, value: 0.3 }], make: (p) => D.Bernoulli(p) },
  Binomial: {
    params: [
      { label: 'n', min: 1, max: 60, value: 20, step: 1 },
      { label: 'p', min: 0, max: 1, value: 0.3 },
    ],
    make: (n, p) => D.Binomial(n, p),
  },
  Categorical: {
    params: [{ label: 'temperature', min: 0.2, max: 5, value: 1 }],
    make: (t) => D.Categorical({ logits: tensor([2, 1, 0.5, 0, -1].map((v) => v / t)) }),
  },
  Poisson: { params: [{ label: 'rate λ', min: 0.1, max: 30, value: 4, step: 0.1 }], make: (l) => D.Poisson(l) },
  Geometric: { params: [{ label: 'p', min: 0.05, max: 1, value: 0.3 }], make: (p) => D.Geometric(p) },
  NegativeBinomial: {
    params: [
      { label: 'r', min: 0.5, max: 20, value: 3 },
      { label: 'p', min: 0.05, max: 1, value: 0.4 },
    ],
    make: (r, p) => D.NegativeBinomial(r, p),
  },
  Hypergeometric: {
    params: [
      { label: 'successes K (N = 50)', min: 0, max: 50, value: 20, step: 1 },
      { label: 'draws n', min: 1, max: 50, value: 10, step: 1 },
    ],
    make: (K, n) => D.Hypergeometric(50, K, n),
  },
  DiscreteUniform: {
    params: [
      { label: 'low', min: -5, max: 5, value: 1, step: 1 },
      { label: 'size', min: 1, max: 20, value: 6, step: 1 },
    ],
    make: (a, n) => D.DiscreteUniform(a, a + n - 1),
  },
}
type FamilyName = keyof typeof FAMILIES

/** Every family as a case of one variants field: its parameters p0, p1, … in the order `make` takes them. */
const FAMILY_FIELD = variants(
  Object.fromEntries(
    Object.entries(FAMILIES).map(([name, f]) => [
      name,
      {
        label: name,
        params: Object.fromEntries(
          f.params.map((p, i): [string, SliderDef] => [
            `p${i}`,
            slider(p.min, p.max, p.value, { label: p.label, ...(p.step ? { step: p.step } : {}) }),
          ]),
        ),
      },
    ]),
  ),
  { label: '1 · family and parameters', choiceLabel: 'family', initial: 'Gamma' },
)

export function FamiliesSpecimen() {
  const state = useFigureState({ family: FAMILY_FIELD })
  const name = state.family.key as FamilyName
  const family = FAMILIES[name]
  const values = state.family.values as Record<string, number>
  const key = family.params.map((_, i) => values[`p${i}`]).join(',')
  const { d, error } = useMemo(() => {
    try {
      return { d: family.make(...key.split(',').map(Number)), error: '' }
    } catch (e) {
      return { d: null, error: (e as Error).message }
    }
  }, [family, key])
  // Axes held per family: x from the family at its default parameters, so moving a parameter changes the curve, not
  // the axes. Choosing another family refits.
  const range = useMemo(() => {
    try {
      return distributionRange(family.make(...family.params.map((p) => p.value)))
    } catch {
      return undefined
    }
  }, [family])
  return (
    <Figure
      id="families"
      title={`${name}: density, cdf and draws`}
      purpose="Each family's density (or mass), cdf, sampler and moments come from one object: the draws' histogram and empirical cdf agree with the closed forms for every parameter."
      state={state}
      caption="The density or mass (line or points) over a histogram of 2,000 draws, the cdf against the empirical cdf of the draws, and the moments from the closed forms. The axes hold while a parameter moves (x from the family's default parameters; the density axis grows to fit any taller curve); choosing a family, or the fit button, refits."
    >
      {d ? (
        <DistributionPanel distribution={d} range={range} rescaleOnChange={false} axisKey={name} />
      ) : (
        <div className="text-sm text-muted-foreground">{error}</div>
      )}
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

const MVN_X = grid(-4, 4, 161)
const MVN_Y = grid(-4, 6, 101)

export function MultivariateNormalSpecimen() {
  const state = useFigureState({
    joint: row('1 · the joint', {
      rho: slider(-0.95, 0.95, 0.7, { label: 'correlation ρ' }),
      sd2: slider(0.3, 2.5, 1.5, { label: 'sd of x₂' }),
    }),
    // The query point (x₁, x₂): x₂ is the observed value being conditioned on, x₁ where the densities are read.
    query: row('2 · the query', {
      observed: slider(-3, 5, 1, { label: 'observed x₂' }),
      x1: slider(-4, 4, 0.5, { label: 'query x₁' }),
    }),
  })
  const { rho, sd2 } = state.joint
  const probe = useProbe({ x: state.bind('query.x1'), y: state.bind('query.observed'), label: 'x₁', yLabel: 'x₂' })
  const query = probe.x ?? state.query.x1
  const observed = probe.y ?? state.query.observed
  const mvn = useMemo(
    () =>
      D.MultivariateNormal(tensor([0, 1]), {
        covariance: fromRows([
          [1, rho * sd2],
          [rho * sd2, sd2 * sd2],
        ]),
      }),
    [rho, sd2],
  )
  const conditional = useMemo(() => mvn.condition([1], tensor([observed])), [mvn, observed])
  const z = useMemo(() => {
    const points: number[] = []
    for (const y of MVN_Y) for (const x of MVN_X) points.push(x, y)
    const p = numbers(mvn.prob(tensor(points, [MVN_X.length * MVN_Y.length, 2])))
    return MVN_Y.map((_, i) => MVN_X.map((_, j) => p[i * MVN_X.length + j]))
  }, [mvn])
  // E[x₁ | x₂] = μ₁ + ρ σ₁/σ₂ (x₂ − μ₂): the line every conditional mean lies on (σ₁ = 1, μ = (0, 1)).
  const regression = useMemo(() => {
    const slope = rho / sd2
    return { x: [slope * (-4 - 1), slope * (6 - 1)], y: [-4, 6] }
  }, [rho, sd2])
  const slice = useMemo(() => {
    // The conditional density against the joint density along the line x₂ = observed, renormalised numerically.
    const points = MVN_X.flatMap((x) => [x, observed])
    const joint = numbers(mvn.prob(tensor(points, [MVN_X.length, 2])))
    const h = MVN_X[1] - MVN_X[0]
    const total = joint.reduce((a, b) => a + b, 0) * h
    return {
      cond: numbers(conditional.prob(tensor(MVN_X.map((x) => [x])))),
      joint: joint.map((v) => v / total),
    }
  }, [mvn, conditional, observed])
  const marginal = mvn.marginal([0])
  // Densities at the query point: the joint, the marginal of the observed coordinate, and the conditional.
  const joint = num(mvn.prob(tensor([query, observed])))
  const pObserved = num(mvn.marginal([1]).prob(tensor([observed])))
  const pConditional = num(conditional.prob(tensor([query])))
  const x = useAxis({ label: 'x₁', range: [-4, 4] })
  const y = useAxis({ label: 'x₂', range: [-4, 6] })
  // The conditional's peak depends on ρ only, so this range holds while the query moves and the slices are live.
  const py = useAxis({ label: 'p(x₁ | x₂)', range: [0, 1.15 / Math.sqrt(2 * Math.PI * (1 - rho * rho))] })
  return (
    <Figure
      title="Conditioning a bivariate normal"
      purpose="Conditioning a Gaussian on x₂ gives another Gaussian in closed form, whose density is the joint density sliced at x₂ and divided by p(x₂)."
      defaultSize="L"
      state={state}
      equation={
        <Equation>
          {tex`p(x_1 \mid x_2) = \frac{p(x_1, x_2)}{p(x_2)} = \frac{${live(joint, { digits: 4 })}}{${live(pObserved, { digits: 4 })}} = ${live(pConditional, { digits: 4, strong: true })}`}
        </Equation>
      }
      readouts={{
        conditional: (
          <>
            <Readout label="mean" value={formatValue(num(conditional.mean()))} />
            <Readout label="variance" value={formatValue(num(conditional.variance()))} />
            <Readout label="closed form 1 − ρ²" value={formatValue(1 - rho * rho)} />
          </>
        ),
        joint: (
          <>
            <Readout label="marginal variance of x₁" value={formatValue(num(marginal.variance()))} />
            <Readout label="entropy (nats)" value={formatValue(num(mvn.entropy()))} />
          </>
        ),
      }}
      caption="Press or drag on the heatmap to move the query point: its height is the observed x₂ being conditioned on, its position along the line is x₁, where the densities are read (x₁ can also be dragged on the lower chart). The lower chart is the closed-form conditional against the joint along the dashed line, renormalised: they coincide. The thin line is the regression line E[x₁ | x₂], on which every conditional mean lies; the conditional variance 1 − ρ² does not depend on x₂."
    >
      <Plots rows={2} heights={[3, 2]}>
        <Plot x={x} y={y}>
          <Raster x={MVN_X} y={MVN_Y} z={z} valueLabel="density" />
          <Curve name="E[x₁ | x₂]" x={regression.x} y={regression.y} slot={2} thin />
          <Curve name="conditioned on" x={[-4, 4]} y={[observed, observed]} slot={1} dashed live />
          <Probe probe={probe} />
        </Plot>
        <Plot x={x} y={py}>
          <Curve name="condition([1], x₂): closed form" x={MVN_X} y={slice.cond} slot={0} live />
          <Curve name="joint along x₂, renormalised" x={MVN_X} y={slice.joint} slot={1} dashed live />
          <Probe probe={{ ...probe, y: undefined }} at={pConditional} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

export function MixtureSpecimen() {
  const state = useFigureState({
    weight: slider(0, 1, 0.3, { label: 'weight of component 0' }),
    gap: slider(0, 6, 3, { label: 'separation' }),
  })
  const { weight, gap } = state
  const components = useMemo(() => [D.Normal(-gap / 2, 0.6), D.Normal(gap / 2, 1)], [gap])
  const mixture = useMemo(() => D.Mixture([weight, 1 - weight], components), [weight, components])
  return (
    <Figure
      id="mixture"
      title="A two-component normal mixture"
      purpose="A mixture's density is the weighted sum of its components' (a log-sum-exp in log space) and its cdf the weighted sum of cdfs; its quantile has no closed form and is found by bisection."
      state={state}
      readouts={<Readout label="median (numerical quantile)" value={formatValue(num(mixture.quantile(0.5)))} />}
      caption="Components N(−gap/2, 0.6²) and N(gap/2, 1). The mean and variance follow from the law of total variance: separating the components adds to the variance even with fixed component variances. The entropy has no closed form (—)."
    >
      <DistributionPanel distribution={mixture} range={[-6, 6]} rescaleOnChange={false} />
    </Figure>
  )
}
