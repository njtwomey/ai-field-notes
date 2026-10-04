import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
  variants,
} from 'aifn-render'
import { Binomial, Exponential, Poisson } from 'aifn/probability/distributions'
import { linspace, toFlat } from 'aifn/foundation/tensor'

type DistId = 'exponential' | 'poisson' | 'binomial'

/** Three laws with their plotted x range; discrete laws are drawn as bars on the integers of the range. */
const LAWS = {
  exponential: { label: 'exponential', law: Exponential(1), discrete: false, range: [0, 6] },
  poisson: { label: 'Poisson', law: Poisson(4), discrete: true, range: [0, 14] },
  binomial: { label: 'binomial', law: Binomial(20, 0.3), discrete: true, range: [0, 20] },
} as const

/** Multiples of the mean (Markov) or of the standard deviation (Chebyshev) plotted on the bound chart. */
const MULTIPLES = toFlat(linspace(0.25, 6, 116))

/**
 * The exact tail probability of a chosen distribution against the Markov or Chebyshev bound, as the threshold moves.
 * Thresholds are in multiples of the mean (Markov: a = c·E[X], bound 1/c) or of σ (Chebyshev: bound 1/k²).
 */
export function TailBounds() {
  const state = useFigureState({
    bound: variants(
      {
        markov: { label: 'Markov', params: { c: slider(0.25, 6, 2, { step: 0.05, label: 'threshold c (× mean)' }) } },
        chebyshev: {
          label: 'Chebyshev',
          params: { c: slider(0.25, 6, 2, { step: 0.05, label: 'threshold k (× σ)' }) },
        },
      },
      { choiceLabel: 'bound' },
    ),
    id: choice<DistId>(
      (Object.keys(LAWS) as DistId[]).map((k) => ({ value: k, label: LAWS[k].label })),
      'exponential',
      { label: 'distribution' },
    ),
  })
  const bound = state.bound.key
  const setMultiple = (v: number) => state.set('bound.c', v)
  const { law, discrete, range } = LAWS[state.id]
  const d = {
    discrete,
    cdf: (x: number) => law.cdf(x),
    density: (x: number) => law.prob(x),
  }
  const mu = law.mean()
  const sigma = Math.sqrt(law.variance())

  // P(X ≥ t) and P(X ≤ t), exact for discrete and continuous laws.
  const upper = (t: number) => (d.discrete ? (Math.ceil(t) - 1 < 0 ? 1 : 1 - d.cdf(Math.ceil(t) - 1)) : 1 - d.cdf(t))
  const lower = (t: number) => (d.discrete ? (Math.floor(t) < 0 ? 0 : d.cdf(Math.floor(t))) : d.cdf(t))
  const exact = (c: number) => (bound === 'markov' ? upper(c * mu) : lower(mu - c * sigma) + upper(mu + c * sigma))
  const limit = (c: number) => Math.min(1, bound === 'markov' ? 1 / c : 1 / (c * c))

  const c = state.bound.values.c
  const cuts = bound === 'markov' ? [c * mu] : [mu - c * sigma, mu + c * sigma]
  const inTail = (x: number) => (bound === 'markov' ? x >= cuts[0] : x <= cuts[0] || x >= cuts[1])

  const densitySeries = ((): SeriesSpec[] => {
    const [lo, hi] = range
    if (d.discrete) {
      const ks = Array.from({ length: hi - lo + 1 }, (_, i) => lo + i)
      const tail = ks.filter(inTail)
      const body = ks.filter((k) => !inTail(k))
      return [
        { name: 'pmf', type: 'bar', x: body, y: body.map((k) => d.density(k)), slot: 0 },
        { name: 'tail', type: 'bar', x: tail, y: tail.map((k) => d.density(k)), slot: 1 },
      ]
    }
    const xs = toFlat(linspace(lo, hi, 300))
    // One shaded region per side, so the two Chebyshev tails are not joined across the middle.
    const regions = [xs.filter((x) => x <= cuts[0] && bound === 'chebyshev'), xs.filter((x) => x >= cuts.at(-1)!)]
    return [
      { name: 'pdf', type: 'line', x: xs, y: xs.map((x) => d.density(x)), slot: 0 },
      ...regions
        .filter((r) => r.length > 0)
        .map((r): SeriesSpec => ({
          name: 'tail',
          type: 'line',
          x: r,
          y: r.map((x) => d.density(x)),
          area: true,
          slot: 1,
        })),
    ]
  })()

  const boundSeries = [
    { name: 'exact tail probability', x: MULTIPLES, y: MULTIPLES.map(exact), slot: 1 },
    {
      name: bound === 'markov' ? 'Markov bound 1/c' : 'Chebyshev bound 1/k²',
      x: MULTIPLES,
      y: MULTIPLES.map(limit),
      dashed: true,
      slot: 2,
    },
  ] as const

  // The threshold itself is draggable: on the density at a (or at μ ± kσ), and on the bound chart at c (or k).
  const densityHandles: Handle[] = cuts.map((x) => ({
    kind: 'x',
    at: x,
    label: bound === 'markov' ? 'a' : x < mu ? 'μ − kσ' : 'μ + kσ',
    onDrag: (v) => setMultiple(bound === 'markov' ? v / mu : Math.abs(v - mu) / sigma),
  }))

  const p = exact(c)
  const b = limit(c)
  const xAxis = useAxis({ label: 'x', hold: 'union' })
  const yAxis = useAxis({ label: d.discrete ? 'P(X = x)' : 'density', range: [0, undefined], hold: 'union' })
  const xAxis2 = useAxis({ label: bound === 'markov' ? 'c = a / E[X]' : 'k', hold: 'union' })
  const yAxis2 = useAxis({ label: 'probability', range: [0, 1] })
  return (
    <Figure
      title="Tail bounds against exact tails"
      caption="Left: the distribution, with the tail event shaded. Right: the exact tail probability and the bound as the threshold moves. Markov uses a = c·E[X] and gives 1/c; Chebyshev uses μ ± kσ and gives 1/k². Drag a threshold line on either chart. The bounds hold for every distribution, so for any particular one they are loose."
      state={state}
      readouts={
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
        <Plot x={xAxis} y={yAxis} height={300}>
          {seriesLayers(densitySeries)}
          {(densityHandles ?? []).map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          <Curve {...boundSeries[0]} />
          <Curve {...boundSeries[1]} />
          <Handle kind="x" at={c} label={bound === 'markov' ? 'c' : 'k'} onDrag={setMultiple} />
        </Plot>
      </div>
    </Figure>
  )
}
