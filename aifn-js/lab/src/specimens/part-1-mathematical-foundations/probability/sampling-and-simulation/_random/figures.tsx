import {
  Beta,
  Binomial,
  ChiSquare,
  Exponential,
  Gamma,
  Normal,
  Poisson,
  StudentT,
  type Univariate,
} from 'aifn/probability/distributions'
import { cholesky } from 'aifn/numerics/linalg'
import * as R from 'aifn/probability/samplers'
import * as Random from 'aifn/foundation/random'
import { correlation } from 'aifn/probability/stats'
import { fromRows, toFlat, toRows } from 'aifn/foundation/tensor'
import { useMemo } from 'react'
import { Figure } from '@lab/layout'
import { choice, row, slider, useFigureState, variants } from '@lab/state'
import { Curve, Plot, Points, Raster, Readout, useAxis } from '@lab/viz'
import { SamplesPanel } from '@lab/views'

const SIZES = [
  { value: 500, label: '500' },
  { value: 5000, label: '5 000' },
  { value: 50000, label: '50 000' },
] as const

/** A sampler from aifn/random and the distribution (from aifn/distributions) its draws should follow. */
type Continuous = {
  label: string
  draw: (s: Random.Stream) => number
  distribution: Univariate<number>
  range?: [number, number]
}

const CONTINUOUS = {
  normal: { label: 'normal(1, 2)', draw: (s) => Random.normal(s, 1, 2), distribution: Normal<number, number>(1, 2) },
  exponential: { label: 'exponential(1)', draw: (s) => Random.exponential(s), distribution: Exponential<number>(1) },
  gammaSmall: {
    label: 'gamma(0.5)',
    draw: (s) => R.gammaVariate(s, 0.5),
    distribution: Gamma<number, number>(0.5, 1),
    range: [0, 4],
  },
  gamma: { label: 'gamma(3)', draw: (s) => R.gammaVariate(s, 3), distribution: Gamma<number, number>(3, 1) },
  betaU: {
    label: 'beta(0.5, 0.5)',
    draw: (s) => R.beta(s, 0.5, 0.5),
    distribution: Beta<number, number>(0.5, 0.5),
    range: [0, 1],
  },
  beta: { label: 'beta(2, 5)', draw: (s) => R.beta(s, 2, 5), distribution: Beta<number, number>(2, 5), range: [0, 1] },
  studentT: { label: 'Student t(3)', draw: (s) => R.studentT(s, 3), distribution: StudentT<number>(3) },
  chiSquare: { label: 'χ²(4)', draw: (s) => R.chiSquare(s, 4), distribution: ChiSquare<number>(4) },
} satisfies Record<string, Continuous>
type ContinuousName = keyof typeof CONTINUOUS

const CONTINUOUS_NAMES = (Object.keys(CONTINUOUS) as ContinuousName[]).map((value) => ({
  value,
  label: CONTINUOUS[value].label,
}))

export function ContinuousSamplersSpecimen() {
  const state = useFigureState({
    name: choice(CONTINUOUS_NAMES, 'gammaSmall', { label: 'sampler' }),
    size: choice(SIZES, 5000, { label: 'draws' }),
    seed: slider(0, 50, 0, { step: 1, label: 'seed' }),
  })
  const { name, size, seed } = state
  const d: Continuous = CONTINUOUS[name]
  const samples = useMemo(() => {
    const s = Random.child(Random.stream(seed), 'samplers', name)
    return Float64Array.from({ length: size }, () => d.draw(s))
  }, [d, name, size, seed])
  const { density, cdf } = useMemo(
    () => ({ density: (x: number) => d.distribution.prob(x), cdf: (x: number) => d.distribution.cdf(x) }),
    [d],
  )
  return (
    <Figure
      title="Draws against the density"
      purpose="Each sampler's histogram matches its exact density, and the Kolmogorov–Smirnov distance shrinks like 1/√n as the draws grow."
      description={`${d.label}: ${size} draws from aifn/random, under the density from aifn/distributions.`}
      state={state}
      caption="gamma(0.5) is the hard case: Marsaglia–Tsang needs shape ≥ 1, so a shape below 1 draws gamma(shape + 1) and multiplies by U^(1/shape); the density's pole at 0 is matched. Raise the draws to 50 000 and the KS distance falls by about √10."
    >
      <SamplesPanel
        samples={samples}
        density={density}
        cdf={cdf}
        range={d.range}
        reference={{ mean: d.distribution.mean(), variance: d.distribution.variance() }}
      />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

export function DiscreteSamplersSpecimen() {
  const state = useFigureState({
    sampler: variants(
      {
        poisson: {
          label: 'Poisson',
          params: { lambda: slider(0.2, 80, 12, { label: 'λ (inversion below 10, PTRS above)' }) },
        },
        binomial: {
          label: 'binomial',
          params: { n: slider(1, 400, 40, { step: 1, label: 'n' }), p: slider(0.01, 0.99, 0.3, { label: 'p' }) },
        },
      },
      { label: '1 · sampler', choiceLabel: 'sampler' },
    ),
    seed: slider(0, 50, 0, { step: 1, label: 'seed' }),
  })
  const { sampler, seed } = state
  const samples = useMemo(() => {
    const s = Random.child(Random.stream(seed), 'discrete', sampler.key)
    return Float64Array.from({ length: 5000 }, () =>
      sampler.key === 'poisson'
        ? R.poisson(s, sampler.values.lambda)
        : R.binomial(s, sampler.values.n, sampler.values.p),
    )
  }, [sampler, seed])
  const distribution = useMemo(
    () => (sampler.key === 'poisson' ? Poisson(sampler.values.lambda) : Binomial(sampler.values.n, sampler.values.p)),
    [sampler],
  )
  const pmf = useMemo(() => (k: number) => distribution.prob(k), [distribution])
  const title =
    sampler.key === 'poisson'
      ? `Poisson(${sampler.values.lambda}) draws`
      : `binomial(${sampler.values.n}, ${sampler.values.p}) draws`
  return (
    <Figure
      title="Poisson and binomial draws against the mass function"
      purpose="Relative frequencies of 5 000 integer draws match the mass function, whichever algorithm the parameters select."
      description={title}
      state={state}
      caption="Poisson uses inversion (walking up the cdf) below λ = 10, where few steps are needed, and the PTRS rejection sampler above; drag λ across 10 and the frequencies stay on the mass function. Binomial uses inversion for small n·min(p, 1 − p) and the beta order-statistic recursion otherwise."
    >
      <SamplesPanel
        kind="discrete"
        samples={samples}
        pmf={pmf}
        reference={{ mean: distribution.mean(), variance: distribution.variance() }}
      />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

const STREAM_LABELS = ['s', 's/0', 's/1', 's/x', 's/x:', 'seed + 1']

/** Streams whose pairwise correlations the independence specimen shows. */
function namedStreams(seed: number): Random.Stream[] {
  const s = Random.stream(seed)
  return [
    Random.stream(seed),
    Random.child(s, 0),
    Random.child(s, 1),
    Random.child(s, 'x'),
    Random.child(s, 'x', ''),
    Random.stream(seed + 1),
  ]
}

const PAIRS = [
  { value: 'parent-child', label: 's vs s/0' },
  { value: 'siblings', label: 's/0 vs s/1' },
  { value: 'lag', label: 'successive draws' },
] as const
const STREAM_INDEX = STREAM_LABELS.map((_, i) => i)

export function StreamIndependenceSpecimen() {
  const pairState = useFigureState({
    pair: choice(PAIRS, 'siblings', { label: 'pairs' }),
    seed: slider(0, 50, 0, { step: 1, label: 'seed' }),
  })
  const corrState = useFigureState({ seed: slider(0, 50, 0, { step: 1, label: 'seed' }) })
  const { pair, seed } = pairState
  const n = 4000
  const { x, y } = useMemo(() => {
    const s = Random.stream(seed)
    const a = pair === 'siblings' ? Random.child(s, 0) : s
    const b = pair === 'parent-child' ? Random.child(s, 0) : pair === 'siblings' ? Random.child(s, 1) : null
    const xs = new Float64Array(n)
    const ys = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      xs[i] = Random.uniform(a)
      ys[i] = b ? Random.uniform(b) : Random.uniform(a)
    }
    return { x: xs, y: ys }
  }, [seed, pair])
  const cseed = corrState.seed
  const correlations = useMemo(() => {
    const draws = namedStreams(cseed).map((t) => Float64Array.from({ length: 20000 }, () => Random.uniform(t)))
    return draws.map((a) => draws.map((b) => correlation(a, b)))
  }, [cseed])
  const unchanged = useMemo(() => {
    const before = Random.child(Random.stream(cseed), 'chain', 1)
    const first = [Random.uniform(before), Random.uniform(before)]
    const parent = Random.stream(cseed)
    for (let i = 0; i < 10_000; i++) Random.randomBits(parent, 1)
    const after = Random.child(parent, 'chain', 1)
    return first[0] === Random.uniform(after) && first[1] === Random.uniform(after)
  }, [cseed])
  let worst = 0
  correlations.forEach((r, i) => r.forEach((v, j) => i !== j && (worst = Math.max(worst, Math.abs(v)))))
  const cx = useAxis({ label: 'stream', categories: STREAM_LABELS })
  const cy = useAxis({ label: 'stream', categories: STREAM_LABELS })
  return (
    <>
      <Figure
        title="Pairs of uniforms"
        purpose="Two streams derived from one seed, or successive draws of one stream, give pairs that fill the unit square evenly: no pattern links them."
        state={pairState}
        caption="4 000 pairs. A parent and its child, two siblings and lag-one pairs all look alike: the counter-based generator hashes each stream's key, so related names give unrelated numbers."
      >
        <SamplesPanel kind="scatter" x={x} y={y} xRange={[0, 1]} yRange={[0, 1]} xLabel="first" yLabel="second" />
      </Figure>
      <Figure
        title="Correlations between six streams"
        purpose="Streams named apart are uncorrelated: every off-diagonal correlation is within sampling noise of zero, about 1/√n."
        description="20 000 uniforms from each stream; s is stream(seed), s/k its child k."
        state={corrState}
        readouts={
          <>
            <Readout label="largest |correlation| between distinct streams" value={worst.toFixed(4)} />
            <Readout label="4/√n" value={(4 / Math.sqrt(20000)).toFixed(4)} />
            <Readout label="s/chain:1 unchanged after the parent draws 10 000 words" value={unchanged ? 'yes' : 'NO'} />
          </>
        }
        caption="The colour scale ends at ±0.05, so the diagonal (1) is saturated and anything visible off it would be a real dependence; the cells are pale. A child's numbers do not depend on how far its parent has drawn."
      >
        <Plot x={cx} y={cy}>
          <Raster
            x={STREAM_INDEX}
            y={STREAM_INDEX}
            z={correlations}
            scale="diverging"
            range={[-0.05, 0.05]}
            valueLabel="correlation"
          />
        </Plot>
      </Figure>
    </>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

const TRIANGLE_X = [0, 1, 0.5, 0]
const TRIANGLE_Y = [0, 0, Math.sqrt(3) / 2, 0]
/** Barycentric coordinates: corners (0, 0), (1, 0) and (½, √3/2) for components 1, 2 and 3. */
const bary = (q: readonly number[]): [number, number] => [q[1] + q[2] / 2, (q[2] * Math.sqrt(3)) / 2]

export function DirichletSpecimen() {
  const state = useFigureState({
    alpha: row('concentrations α', {
      a1: slider(0.02, 10, 0.4, { label: 'α₁ (bottom left)' }),
      a2: slider(0.02, 10, 2, { label: 'α₂ (bottom right)' }),
      a3: slider(0.02, 10, 5, { label: 'α₃ (top)' }),
    }),
  })
  const { a1, a2, a3 } = state.alpha
  const { x, y } = useMemo(() => {
    const s = Random.child(Random.stream('dirichlet'), a1, a2, a3)
    // 3000 draws in one call: a [3000, 3] tensor, row i the i-th draw.
    const p = toRows(R.dirichlet(s, [a1, a2, a3], { shape: [3000] }))
    const xy = p.map(bary)
    return { x: xy.map((q) => q[0]), y: xy.map((q) => q[1]) }
  }, [a1, a2, a3])
  const total = a1 + a2 + a3
  const mean = bary([a1 / total, a2 / total, a3 / total])
  const xa = useAxis({ label: 'barycentric x', range: [-0.05, 1.05] })
  const ya = useAxis({ label: 'barycentric y', range: [-0.05, 0.92], equal: xa })
  return (
    <Figure
      title="Dirichlet(α) draws on the simplex"
      purpose="Each draw is a probability vector, a point in the triangle; concentrations below 1 push draws to the edges and corners, large ones pull them to the mean α/Σα."
      state={state}
      readouts={<Readout label="mean α/Σα" value={`(${[a1, a2, a3].map((a) => (a / total).toFixed(3)).join(', ')})`} />}
      caption="3 000 draws. With α₁ = 0.4 many draws hug the edge opposite corner 1 (component 1 near 0) or the corner itself. Set all three to 1 for the uniform distribution on the triangle; raise them together and the cloud tightens about the mean."
    >
      <Plot x={xa} y={ya}>
        <Curve name="simplex" x={TRIANGLE_X} y={TRIANGLE_Y} muted silent />
        <Points name="draws" x={x} y={y} thin size={3} />
        <Points name="mean α/Σα" x={[mean[0]]} y={[mean[1]]} emphasis size={10} />
      </Plot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

export function MultivariateNormalSpecimen() {
  const state = useFigureState({
    cov: row('covariance', {
      s1: slider(0.2, 2.5, 1.5, { label: 'σ₁' }),
      s2: slider(0.2, 2.5, 0.7, { label: 'σ₂' }),
      rho: slider(-0.99, 0.99, 0.6, { label: 'ρ' }),
    }),
  })
  const { s1, s2, rho } = state.cov
  const { x, y, ex, ey, L } = useMemo(() => {
    const covariance = fromRows([
      [s1 * s1, rho * s1 * s2],
      [rho * s1 * s2, s2 * s2],
    ])
    // The same factor the sampler uses: L with covariance = L Lᵀ.
    const L = toRows(cholesky(covariance).L)
    // 3000 draws in one call, factoring the covariance with aifn/linalg's cholesky.
    const rows = toRows(R.multivariateNormal(Random.stream('mvn'), [0, 0], { covariance }, { shape: [3000] }))
    // The 2σ ellipse: L applied to a circle of radius 2.
    const t = Array.from({ length: 121 }, (_, i) => (2 * Math.PI * i) / 120)
    return {
      x: rows.map((r) => r[0]),
      y: rows.map((r) => r[1]),
      ex: t.map((a) => 2 * L[0][0] * Math.cos(a)),
      ey: t.map((a) => 2 * (L[1][0] * Math.cos(a) + L[1][1] * Math.sin(a))),
      L,
    }
  }, [s1, s2, rho])
  const xa = useAxis({ label: 'x₁', range: [-6, 6] })
  const ya = useAxis({ label: 'x₂', range: [-6, 6], equal: xa })
  return (
    <Figure
      title="Draws with the 2σ ellipse"
      purpose="μ + L z turns independent standard normals z into draws with covariance L Lᵀ: the Cholesky factor maps the radius-2 circle onto the 2σ ellipse."
      state={state}
      readouts={
        <>
          <Readout label="L" value={`[[${L[0][0].toFixed(3)}, 0], [${L[1][0].toFixed(3)}, ${L[1][1].toFixed(3)}]]`} />
        </>
      }
      caption="3 000 draws with mean 0. The ellipse is L applied to the circle of radius 2, so it holds the same share of the draws as the circle holds of standard normal pairs: 1 − e⁻² ≈ 86.5%."
    >
      <Plot x={xa} y={ya}>
        <Points name="draws" x={x} y={y} thin size={3} />
        <Curve name="2σ ellipse (L · circle)" x={ex} y={ey} emphasis />
      </Plot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

const STEPS = 200
let walkCalls = 0
/** One Gaussian random walk of STEPS steps on its own stream. Module-level, so replicate's cache persists. */
function walk(s: Random.Stream): number[] {
  walkCalls++
  const z = toFlat(Random.normals(s, STEPS))
  const path = [0]
  for (let i = 0; i < STEPS; i++) path.push(path[i] + z[i])
  return path
}
const WALK_STEPS = Array.from({ length: STEPS + 1 }, (_, i) => i)

export function ReplicateSpecimen() {
  const state = useFigureState({ n: slider(1, 60, 5, { step: 1, label: 'replicates' }) })
  const n = state.n
  const { paths, computed } = useMemo(() => {
    const before = walkCalls
    const out = Random.replicate(n, Random.stream('walks'), walk)
    return { paths: out, computed: walkCalls - before }
  }, [n])
  const x = useAxis({ label: 'step' })
  const y = useAxis({ label: 'position', hold: 'union' })
  return (
    <Figure
      title="Gaussian random walks"
      purpose="Walk k always draws from child(stream('walks'), k), so adding replicates computes only the new walks and never changes the old ones."
      state={state}
      readouts={
        <>
          <Readout label="walks computed on this change" value={computed} />
          <Readout label="walk k uses" value="child(stream('walks'), k)" />
        </>
      }
      caption="Raise the replicates: the drawn walks stay where they were and 'walks computed' counts only the new ones. Lower it and raise it again: the cache returns the same walks without computing any."
    >
      <Plot x={x} y={y} legend={false}>
        {paths.map((p, k) => (
          <Curve key={k} name="walks" x={WALK_STEPS} y={p} thin slot={0} silent />
        ))}
      </Plot>
    </Figure>
  )
}
