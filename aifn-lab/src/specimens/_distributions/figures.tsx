import * as D from 'aifn/distributions'
import { fromRows, linspace, tensor, toFlat, unwrap, type Value } from 'aifn/tensor'
import { useCallback, useMemo, useState } from 'react'
import { Choice, Slider } from '@lab/controls'
import { Figure } from '@lab/layout'
import { ChartSize, Heatmap, Readout, XYChart, type Handle, type HeatmapOverlay, type XYSeries } from '@lab/viz'
import { DistributionView, distributionRange, formatValue } from '@lab/views'

/** A Value as numbers. */
const numbers = (v: Value): number[] => {
  const r = unwrap(v)
  return typeof r === 'number' ? [r] : toFlat(r)
}
const num = (v: Value) => numbers(v)[0]
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
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
      { label: 'df ν', min: 0.5, max: 30, value: 3 },
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
  ChiSquare: { params: [{ label: 'df k', min: 0.5, max: 30, value: 4 }], make: (k) => D.ChiSquare(k) },
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
  Poisson: { params: [{ label: 'rate λ', min: 0.1, max: 30, value: 4 }], make: (l) => D.Poisson(l) },
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

export function FamiliesSpecimen() {
  const [name, setName] = useState<FamilyName>('Gamma')
  const [values, setValues] = useState<Record<string, number[]>>({})
  const family = FAMILIES[name]
  const current = values[name] ?? family.params.map((p) => p.value)
  const set = (i: number) => (v: number) =>
    setValues((all) => ({ ...all, [name]: current.map((c, j) => (j === i ? v : c)) }))
  const { d, error } = useMemo(() => {
    try {
      return { d: family.make(...current), error: '' }
    } catch (e) {
      return { d: null, error: (e as Error).message }
    }
  }, [family, current])
  // Axes held per family: x from the family at its default parameters, so moving a parameter changes the curve, not
  // the axes. Choosing another family refits.
  const range = useMemo(() => {
    try {
      return distributionRange(family.make(...family.params.map((p) => p.value)))
    } catch {
      return undefined
    }
  }, [family])
  const controls = (
    <>
      <Choice
        label="family"
        value={name}
        onChange={(v) => setName(v as FamilyName)}
        options={Object.keys(FAMILIES) as FamilyName[]}
      />
      {family.params.map((p, i) => (
        <Slider
          key={`${name}-${p.label}`}
          label={p.label}
          value={current[i]}
          min={p.min}
          max={p.max}
          step={p.step}
          onChange={set(i)}
        />
      ))}
    </>
  )
  if (!d)
    return (
      <Figure title={name} controls={controls}>
        <div className="text-sm text-muted-foreground">{error}</div>
      </Figure>
    )
  return (
    <DistributionView
      id="families"
      title={`${name}: density, cdf and draws`}
      distribution={d}
      controls={controls}
      range={range}
      rescaleOnChange={false}
      axisKey={name}
      caption="The density or mass (line or points) over a histogram of 2,000 draws, the cdf against the empirical cdf of the draws, and the moments from the closed forms. The axes hold while a parameter moves (x from the family's default parameters; the density axis grows to fit any taller curve); choosing a family, or the fit button, refits."
    />
  )
}

// ---------------------------------------------------------------------------------------------------------------------

export function MultivariateNormalSpecimen() {
  const [rho, setRho] = useState(0.7)
  const [sd2, setSd2] = useState(1.5)
  const [observed, setObserved] = useState(1)
  // The query point (x₁, x₂): x₂ is the observed value being conditioned on, x₁ where the densities are read.
  const [query, setQuery] = useState(0.5)
  const setPoint = useCallback(([x1, x2]: [number, number]) => {
    setQuery(clamp(x1, -4, 4))
    setObserved(clamp(x2, -3, 5))
  }, [])
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
  const xs = useMemo(() => grid(-4, 4, 161), [])
  const ys = useMemo(() => grid(-4, 6, 101), [])
  const z = useMemo(() => {
    const points: number[] = []
    for (const y of ys) for (const x of xs) points.push(x, y)
    const p = numbers(mvn.prob(tensor(points, [xs.length * ys.length, 2])))
    return ys.map((_, i) => xs.map((_, j) => p[i * xs.length + j]))
  }, [mvn, xs, ys])
  const overlay = useMemo((): HeatmapOverlay[] => {
    // E[x₁ | x₂] = μ₁ + ρ σ₁/σ₂ (x₂ − μ₂): the line every conditional mean lies on (σ₁ = 1, μ = (0, 1)).
    const slope = rho / sd2
    return [
      { name: `x₂ = ${formatValue(observed)}`, type: 'line', x: [-4, 4], y: [observed, observed], slot: 1 },
      { name: 'E[x₁ | x₂]', type: 'line', x: [slope * (-4 - 1), slope * (6 - 1)], y: [-4, 6], slot: 2, thin: true },
    ]
  }, [observed, rho, sd2])
  const heatmapHandles = useMemo(
    (): Handle[] => [{ kind: 'point', at: [query, observed], label: '(x₁, x₂)', onDrag: setPoint }],
    [query, observed, setPoint],
  )
  const sliceHandles = useMemo(
    (): Handle[] => [{ kind: 'x', at: query, label: 'x₁', onDrag: (x) => setQuery(clamp(x, -4, 4)) }],
    [query],
  )
  const slice = useMemo((): XYSeries[] => {
    // The conditional density against the joint density along the line x₂ = observed, renormalised numerically.
    const points = xs.flatMap((x) => [x, observed])
    const joint = numbers(mvn.prob(tensor(points, [xs.length, 2])))
    const h = xs[1] - xs[0]
    const total = joint.reduce((a, b) => a + b, 0) * h
    const cond = numbers(conditional.prob(tensor(xs.map((x) => [x]))))
    return [
      { name: 'condition([1], x₂): closed form', type: 'line', x: xs, y: cond, slot: 0 },
      {
        name: 'joint along x₂, renormalised',
        type: 'line',
        x: xs,
        y: joint.map((v) => v / total),
        slot: 1,
        dashed: true,
      },
    ]
  }, [mvn, conditional, xs, observed])
  const marginal = mvn.marginal([0])
  // Densities at the query point: the joint, the marginal of the observed coordinate, and the conditional.
  const joint = num(mvn.prob(tensor([query, observed])))
  const pObserved = num(mvn.marginal([1]).prob(tensor([observed])))
  const pConditional = num(conditional.prob(tensor([query])))
  return (
    <Figure
      title="Conditioning a bivariate normal"
      defaultSize="L"
      controls={
        <>
          <Slider label="correlation ρ" value={rho} min={-0.95} max={0.95} onChange={setRho} />
          <Slider label="sd of x₂" value={sd2} min={0.3} max={2.5} onChange={setSd2} />
          <Slider label="observed x₂" value={observed} min={-3} max={5} onChange={setObserved} />
          <Slider label="query x₁" value={query} min={-4} max={4} onChange={setQuery} />
        </>
      }
      readouts={
        <>
          <Readout label="query (x₁, x₂)" value={`(${formatValue(query)}, ${formatValue(observed)})`} />
          <Readout label="joint p(x₁, x₂)" value={formatValue(joint)} />
          <Readout label="marginal p(x₂)" value={formatValue(pObserved)} />
          <Readout label="conditional p(x₁ | x₂)" value={formatValue(pConditional)} />
          <Readout label="p(x₁, x₂) / p(x₂)" value={formatValue(joint / pObserved)} />
          <Readout label="conditional mean" value={formatValue(num(conditional.mean()))} />
          <Readout label="conditional variance" value={formatValue(num(conditional.variance()))} />
          <Readout label="closed form 1 − ρ²" value={formatValue(1 - rho * rho)} />
          <Readout label="marginal variance of x₁" value={formatValue(num(marginal.variance()))} />
          <Readout label="entropy (nats)" value={formatValue(num(mvn.entropy()))} />
        </>
      }
      caption="Click or drag on the heatmap to move the query point: its height is the observed x₂ being conditioned on, its position along the line is x₁, where the densities are read (x₁ can also be dragged on the lower chart). condition(indices, values) returns the Gaussian of the remaining coordinates; its density matches the joint sliced at the observed value and renormalised, so p(x₁ | x₂) = p(x₁, x₂) / p(x₂). The thin line is the regression line E[x₁ | x₂], on which every conditional mean lies."
    >
      <div className="flex flex-col gap-4">
        <ChartSize scale={0.6}>
          <Heatmap
            x={xs}
            y={ys}
            z={z}
            xLabel="x₁"
            yLabel="x₂"
            overlay={overlay}
            handles={heatmapHandles}
            valueLabel="density"
            rescaleOnChange={false}
            holdFit="union"
          />
        </ChartSize>
        <ChartSize scale={0.4}>
          <XYChart
            series={slice}
            handles={sliceHandles}
            xLabel="x₁"
            yLabel="p(x₁ | x₂)"
            rescaleOnChange={false}
            holdFit="union"
          />
        </ChartSize>
      </div>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

export function MixtureSpecimen() {
  const [weight, setWeight] = useState(0.3)
  const [gap, setGap] = useState(3)
  const components = useMemo(() => [D.Normal(-gap / 2, 0.6), D.Normal(gap / 2, 1)], [gap])
  const mixture = useMemo(() => D.Mixture([weight, 1 - weight], components), [weight, components])
  return (
    <DistributionView
      id="mixture"
      title="A two-component normal mixture"
      distribution={mixture}
      range={[-6, 6]}
      rescaleOnChange={false}
      controls={
        <>
          <Slider label="weight of component 0" value={weight} min={0} max={1} onChange={setWeight} />
          <Slider label="separation" value={gap} min={0} max={6} onChange={setGap} />
        </>
      }
      readouts={<Readout label="median (numerical quantile)" value={formatValue(num(mixture.quantile(0.5)))} />}
      caption="logProb is a log-sum-exp over the components, the cdf the weighted sum, the quantile a bisection on the cdf; the mean and variance follow from the law of total variance. The entropy has no closed form (—)."
    />
  )
}
