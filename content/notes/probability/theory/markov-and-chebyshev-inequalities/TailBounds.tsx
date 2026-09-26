import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { defaults, distribution } from '@/lib/distributions'
import { linspace } from '@/lib/math'

type Bound = 'markov' | 'chebyshev'
type DistId = 'exponential' | 'poisson' | 'binomial'

const DISTRIBUTIONS: { value: DistId; label: string }[] = [
  { value: 'exponential', label: 'exponential' },
  { value: 'poisson', label: 'Poisson' },
  { value: 'binomial', label: 'binomial' },
]

/** Multiples of the mean (Markov) or of the standard deviation (Chebyshev) plotted on the bound chart. */
const MULTIPLES = linspace(0.25, 6, 116)

/**
 * The exact tail probability of a chosen distribution against the Markov or Chebyshev bound, as the threshold moves.
 * Thresholds are in multiples of the mean (Markov: a = c·E[X], bound 1/c) or of σ (Chebyshev: bound 1/k²).
 */
export function TailBounds() {
  const [bound, setBound] = useState<Bound>('markov')
  const [id, setId] = useState<DistId>('exponential')
  const multiple = useParam(2, { min: 0.25, max: 6, step: 0.05 })

  const d = distribution(id)
  const params = useMemo(() => defaults(d), [d])
  const mu = d.mean(params)
  const sigma = Math.sqrt(d.variance(params))

  // P(X ≥ t) and P(X ≤ t), exact for discrete and continuous laws.
  const upper = (t: number) =>
    d.discrete ? (Math.ceil(t) - 1 < 0 ? 1 : 1 - d.cdf(Math.ceil(t) - 1, params)) : 1 - d.cdf(t, params)
  const lower = (t: number) => (d.discrete ? (Math.floor(t) < 0 ? 0 : d.cdf(Math.floor(t), params)) : d.cdf(t, params))
  const exact = (c: number) => (bound === 'markov' ? upper(c * mu) : lower(mu - c * sigma) + upper(mu + c * sigma))
  const limit = (c: number) => Math.min(1, bound === 'markov' ? 1 / c : 1 / (c * c))

  const c = multiple.value
  const cuts = bound === 'markov' ? [c * mu] : [mu - c * sigma, mu + c * sigma]
  const inTail = (x: number) => (bound === 'markov' ? x >= cuts[0] : x <= cuts[0] || x >= cuts[1])

  const densitySeries = ((): XYSeries[] => {
    const [lo, hi] = d.range(params)
    if (d.discrete) {
      const ks = Array.from({ length: hi - lo + 1 }, (_, i) => lo + i)
      const tail = ks.filter(inTail)
      const body = ks.filter((k) => !inTail(k))
      return [
        { name: 'pmf', type: 'bar', x: body, y: body.map((k) => d.density(k, params)), slot: 0 },
        { name: 'tail', type: 'bar', x: tail, y: tail.map((k) => d.density(k, params)), slot: 1 },
      ]
    }
    const xs = linspace(lo, hi, 300)
    // One shaded region per side, so the two Chebyshev tails are not joined across the middle.
    const regions = [xs.filter((x) => x <= cuts[0] && bound === 'chebyshev'), xs.filter((x) => x >= cuts.at(-1)!)]
    return [
      { name: 'pdf', type: 'line', x: xs, y: xs.map((x) => d.density(x, params)), slot: 0 },
      ...regions
        .filter((r) => r.length > 0)
        .map((r): XYSeries => ({
          name: 'tail',
          type: 'line',
          x: r,
          y: r.map((x) => d.density(x, params)),
          area: true,
          slot: 1,
        })),
    ]
  })()

  const boundSeries: XYSeries[] = [
    { name: 'exact tail probability', type: 'line', x: MULTIPLES, y: MULTIPLES.map(exact), slot: 1 },
    {
      name: bound === 'markov' ? 'Markov bound 1/c' : 'Chebyshev bound 1/k²',
      type: 'line',
      x: MULTIPLES,
      y: MULTIPLES.map(limit),
      dashed: true,
      slot: 2,
    },
  ]

  // The threshold itself is draggable: on the density at a (or at μ ± kσ), and on the bound chart at c (or k).
  const densityHandles: Handle[] = cuts.map((x) => ({
    kind: 'x',
    at: x,
    label: bound === 'markov' ? 'a' : x < mu ? 'μ − kσ' : 'μ + kσ',
    onDrag: (v) => multiple.set(bound === 'markov' ? v / mu : Math.abs(v - mu) / sigma),
  }))
  const boundHandles: Handle[] = [{ kind: 'x', at: c, label: bound === 'markov' ? 'c' : 'k', onDrag: multiple.set }]

  const p = exact(c)
  const b = limit(c)
  return (
    <Interactive
      title="Tail bounds against exact tails"
      caption="Left: the distribution, with the tail event shaded. Right: the exact tail probability and the bound as the threshold moves. Markov uses a = c·E[X] and gives 1/c; Chebyshev uses μ ± kσ and gives 1/k². Drag a threshold line on either chart. The bounds hold for every distribution, so for any particular one they are loose."
      controls={
        <>
          <ParamChoice
            label="bound"
            value={bound}
            onChange={setBound}
            options={[
              { value: 'markov', label: 'Markov' },
              { value: 'chebyshev', label: 'Chebyshev' },
            ]}
          />
          <ParamChoice label="distribution" value={id} onChange={setId} options={DISTRIBUTIONS} />
          <ParamSlider label={bound === 'markov' ? 'threshold c (× mean)' : 'threshold k (× σ)'} param={multiple} />
        </>
      }
      readout={
        <>
          <Readout label="mean" value={formatNumber(mu)} />
          <Readout label="σ" value={formatNumber(sigma)} />
          <Readout label="exact" value={formatNumber(p)} />
          <Readout label="bound" value={formatNumber(b)} />
          <Readout label="bound ÷ exact" value={p > 0 ? formatNumber(b / p) : '∞'} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          height={300}
          xLabel="x"
          yLabel={d.discrete ? 'P(X = x)' : 'density'}
          series={densitySeries}
          yRange={[0, undefined]}
          handles={densityHandles}
        />
        <XYChart
          height={300}
          xLabel={bound === 'markov' ? 'c = a / E[X]' : 'k'}
          yLabel="probability"
          series={boundSeries}
          yRange={[0, 1]}
          handles={boundHandles}
        />
      </div>
    </Interactive>
  )
}
