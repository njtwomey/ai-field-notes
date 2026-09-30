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
import { useMemo, useState } from 'react'
import { Select, Slider } from '@lab/controls'
import { Figure } from '@lab/layout'
import { Heatmap, Readout, XYChart, type XYSeries } from '@lab/viz'
import { SamplesView } from '@lab/views'

const SIZES = [
  { value: '500', label: '500' },
  { value: '5000', label: '5 000' },
  { value: '50000', label: '50 000' },
] as const
type Size = (typeof SIZES)[number]['value']

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

export function ContinuousSamplersSpecimen() {
  const [name, setName] = useState<ContinuousName>('gammaSmall')
  const [size, setSize] = useState<Size>('5000')
  const [seed, setSeed] = useState(0)
  const d: Continuous = CONTINUOUS[name]
  const samples = useMemo(() => {
    const s = Random.child(Random.stream(seed), 'samplers', name)
    return Float64Array.from({ length: Number(size) }, () => d.draw(s))
  }, [d, name, size, seed])
  const { density, cdf } = useMemo(
    () => ({ density: (x: number) => d.distribution.prob(x), cdf: (x: number) => d.distribution.cdf(x) }),
    [d],
  )
  return (
    <SamplesView
      title={`${d.label} draws against the density`}
      controls={
        <>
          <Select
            label="sampler"
            value={name}
            onChange={setName}
            options={(Object.keys(CONTINUOUS) as ContinuousName[]).map((value) => ({
              value,
              label: CONTINUOUS[value].label,
            }))}
          />
          <Select label="draws" value={size} onChange={setSize} options={SIZES} />
          <Slider label="seed" value={seed} min={0} max={50} step={1} onChange={setSeed} />
        </>
      }
      samples={samples}
      density={density}
      cdf={cdf}
      range={d.range}
      reference={{ mean: d.distribution.mean(), variance: d.distribution.variance() }}
    />
  )
}

// ---------------------------------------------------------------------------------------------------------------------

export function DiscreteSamplersSpecimen() {
  const [kind, setKind] = useState<'poisson' | 'binomial'>('poisson')
  const [lambda, setLambda] = useState(12)
  const [n, setN] = useState(40)
  const [p, setP] = useState(0.3)
  const [seed, setSeed] = useState(0)
  const samples = useMemo(() => {
    const s = Random.child(Random.stream(seed), 'discrete', kind)
    return Float64Array.from({ length: 5000 }, () => (kind === 'poisson' ? R.poisson(s, lambda) : R.binomial(s, n, p)))
  }, [kind, lambda, n, p, seed])
  const distribution = useMemo(() => (kind === 'poisson' ? Poisson(lambda) : Binomial(n, p)), [kind, lambda, n, p])
  const pmf = useMemo(() => (k: number) => distribution.prob(k), [distribution])
  return (
    <SamplesView
      kind="discrete"
      title={kind === 'poisson' ? `Poisson(${lambda}) draws` : `binomial(${n}, ${p}) draws`}
      controls={
        <>
          <Select
            label="sampler"
            value={kind}
            onChange={setKind}
            options={[
              { value: 'poisson', label: 'Poisson' },
              { value: 'binomial', label: 'binomial' },
            ]}
          />
          {kind === 'poisson' ? (
            <Slider
              label="λ (inversion below 10, PTRS above)"
              value={lambda}
              min={0.2}
              max={80}

              onChange={setLambda}
            />
          ) : (
            <>
              <Slider label="n" value={n} min={1} max={400} step={1} onChange={setN} />
              <Slider label="p" value={p} min={0.01} max={0.99} onChange={setP} />
            </>
          )}
          <Slider label="seed" value={seed} min={0} max={50} step={1} onChange={setSeed} />
        </>
      }
      samples={samples}
      pmf={pmf}
      reference={{ mean: distribution.mean(), variance: distribution.variance() }}
    />
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

export function StreamIndependenceSpecimen() {
  const [seed, setSeed] = useState(0)
  const [pair, setPair] = useState<'parent-child' | 'siblings' | 'lag'>('siblings')
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
  const correlations = useMemo(() => {
    const draws = namedStreams(seed).map((t) => Float64Array.from({ length: 20000 }, () => Random.uniform(t)))
    return draws.map((a) => draws.map((b) => correlation(a, b)))
  }, [seed])
  const unchanged = useMemo(() => {
    const before = Random.child(Random.stream(seed), 'chain', 1)
    const first = [Random.uniform(before), Random.uniform(before)]
    const parent = Random.stream(seed)
    for (let i = 0; i < 10_000; i++) Random.randomBits(parent, 1)[0]
    const after = Random.child(parent, 'chain', 1)
    return first[0] === Random.uniform(after) && first[1] === Random.uniform(after)
  }, [seed])
  const idx = useMemo(() => STREAM_LABELS.map((_, i) => i), [])
  let worst = 0
  correlations.forEach((row, i) => row.forEach((r, j) => i !== j && (worst = Math.max(worst, Math.abs(r)))))
  return (
    <>
      <SamplesView
        kind="scatter"
        title="Pairs of uniforms"
        controls={
          <>
            <Select
              label="pairs"
              value={pair}
              onChange={setPair}
              options={[
                { value: 'parent-child', label: 's vs s/0' },
                { value: 'siblings', label: 's/0 vs s/1' },
                { value: 'lag', label: 'successive draws' },
              ]}
            />
            <Slider label="seed" value={seed} min={0} max={50} step={1} onChange={setSeed} />
          </>
        }
        x={x}
        y={y}
        xRange={[0, 1]}
        yRange={[0, 1]}
        xLabel="first"
        yLabel="second"
      />
      <Figure
        title="Correlations between six streams"
        description={`Streams ${STREAM_LABELS.map((l, i) => `${i}: ${l}`).join(', ')}; 20 000 uniforms each, seed ${seed}.`}
        readouts={
          <>
            <Readout label="largest |correlation| between distinct streams" value={worst.toFixed(4)} />
            <Readout label="4/√n" value={(4 / Math.sqrt(20000)).toFixed(4)} />
            <Readout label="s/chain:1 unchanged after the parent draws 10 000 words" value={unchanged ? 'yes' : 'NO'} />
          </>
        }
      >
        <Heatmap
          x={idx}
          y={idx}
          z={correlations}
          scale="diverging"
          range={[-0.05, 0.05]}
          xLabel="stream"
          yLabel="stream"
          valueLabel="correlation"
        />
      </Figure>
    </>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

const TRIANGLE: XYSeries = {
  name: 'simplex',
  type: 'line',
  x: [0, 1, 0.5, 0],
  y: [0, 0, Math.sqrt(3) / 2, 0],
  muted: true,
}

export function DirichletSpecimen() {
  const [a1, setA1] = useState(0.4)
  const [a2, setA2] = useState(2)
  const [a3, setA3] = useState(5)
  const { x, y } = useMemo(() => {
    const s = Random.child(Random.stream('dirichlet'), a1, a2, a3)
    // 3000 draws in one call: a [3000, 3] tensor, row i the i-th draw.
    const p = toRows(R.dirichlet(s, [a1, a2, a3], { shape: [3000] }))
    // Barycentric coordinates: corners (0, 0), (1, 0) and (½, √3/2) for components 1, 2 and 3.
    const xs = p.map((q) => q[1] + q[2] / 2)
    const ys = p.map((q) => (q[2] * Math.sqrt(3)) / 2)
    return { x: xs, y: ys }
  }, [a1, a2, a3])
  const overlay = useMemo(() => [TRIANGLE], [])
  return (
    <SamplesView
      kind="scatter"
      title="Dirichlet(α) draws on the simplex"
      controls={
        <>
          <Slider label="α₁ (bottom left)" value={a1} min={0.02} max={10} onChange={setA1} />
          <Slider label="α₂ (bottom right)" value={a2} min={0.02} max={10} onChange={setA2} />
          <Slider label="α₃ (top)" value={a3} min={0.02} max={10} onChange={setA3} />
        </>
      }
      x={x}
      y={y}
      overlay={overlay}
      xRange={[-0.05, 1.05]}
      yRange={[-0.05, 0.92]}
    />
  )
}

// ---------------------------------------------------------------------------------------------------------------------

export function MultivariateNormalSpecimen() {
  const [s1, setS1] = useState(1.5)
  const [s2, setS2] = useState(0.7)
  const [rho, setRho] = useState(0.6)
  const { x, y, overlay } = useMemo(() => {
    const covariance = fromRows([
      [s1 * s1, rho * s1 * s2],
      [rho * s1 * s2, s2 * s2],
    ])
    // The same factor the sampler uses: L with covariance = L Lᵀ.
    const L = toRows(cholesky(covariance).L)
    // 3000 draws in one call, factoring the covariance with aifn/linalg's cholesky.
    const draws = R.multivariateNormal(Random.stream('mvn'), [0, 0], { covariance }, { shape: [3000] })
    const rows = toRows(draws)
    const xs = rows.map((r) => r[0])
    const ys = rows.map((r) => r[1])
    // The 2σ ellipse: L applied to a circle of radius 2.
    const t = Array.from({ length: 121 }, (_, i) => (2 * Math.PI * i) / 120)
    const ellipse: XYSeries = {
      name: '2σ ellipse (L · circle)',
      type: 'line',
      x: t.map((a) => 2 * L[0][0] * Math.cos(a)),
      y: t.map((a) => 2 * (L[1][0] * Math.cos(a) + L[1][1] * Math.sin(a))),
      emphasis: true,
    }
    return { x: xs, y: ys, overlay: [ellipse] }
  }, [s1, s2, rho])
  return (
    <SamplesView
      kind="scatter"
      title="Draws with the 2σ ellipse"
      controls={
        <>
          <Slider label="σ₁" value={s1} min={0.2} max={2.5} onChange={setS1} />
          <Slider label="σ₂" value={s2} min={0.2} max={2.5} onChange={setS2} />
          <Slider label="ρ" value={rho} min={-0.99} max={0.99} onChange={setRho} />
        </>
      }
      x={x}
      y={y}
      overlay={overlay}
      xRange={[-6, 6]}
      yRange={[-6, 6]}
    />
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

export function ReplicateSpecimen() {
  const [n, setN] = useState(5)
  const { paths, computed } = useMemo(() => {
    const before = walkCalls
    const out = Random.replicate(n, Random.stream('walks'), walk)
    return { paths: out, computed: walkCalls - before }
  }, [n])
  const steps = useMemo(() => Array.from({ length: STEPS + 1 }, (_, i) => i), [])
  const series = useMemo(
    (): XYSeries[] => paths.map((y) => ({ name: 'walks', type: 'line', x: steps, y, thin: true, slot: 0 })),
    [paths, steps],
  )
  return (
    <Figure
      title="Gaussian random walks"
      controls={<Slider label="replicates" value={n} min={1} max={60} step={1} onChange={setN} />}
      readouts={
        <>
          <Readout label="walks computed on this change" value={computed} />
          <Readout label="walk k uses" value="child(stream('walks'), k)" />
        </>
      }
    >
      <XYChart series={series} xLabel="step" yLabel="position" />
    </Figure>
  )
}
