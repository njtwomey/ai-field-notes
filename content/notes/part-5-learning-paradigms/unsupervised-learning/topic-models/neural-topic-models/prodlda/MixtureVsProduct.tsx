import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  MathText,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'

const WORDS = ['goal', 'match', 'team', 'league', 'club', 'transfer', 'fee', 'deal', 'market', 'shares', 'bank', 'rate']
const X = WORDS.map((_, i) => i + 1)
const Y_RANGE: [number, number] = [0, 0.6]

/** A topic as a bump of log-probability over the word axis: log β ∝ −(i − centre)² / (2 width²). */
function logTopic(centre: number, width: number): number[] {
  return X.map((i) => -((i - centre) ** 2) / (2 * width * width))
}

function softmax(logits: number[]): number[] {
  const m = Math.max(...logits)
  const e = logits.map((l) => Math.exp(l - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

const perplexity = (p: number[]) => 2 ** -p.reduce((h, q) => (q > 0 ? h + q * Math.log2(q) : h), 0)

/** A mixture of two topics against their weighted product, over a twelve-word vocabulary. */
export function MixtureVsProduct() {
  const state = useFigureState({
    theta: float(0.5, { min: 0, max: 1, step: 0.01, label: 'θ₁ (weight of topic 1)' }),
    width: float(1.6, { min: 0.8, max: 3, step: 0.1, label: 'topic width' }),
  })

  const { sport, finance, mixture, product } = useMemo(() => {
    const lA = logTopic(3, state.width)
    const lB = logTopic(9, state.width)
    const sport = softmax(lA)
    const finance = softmax(lB)
    // LDA: average the probabilities. ProdLDA with B = log β: average the logits, then renormalise.
    const mixture = sport.map((p, i) => state.theta * p + (1 - state.theta) * finance[i])
    const product = softmax(lA.map((l, i) => state.theta * l + (1 - state.theta) * lB[i]))
    return { sport, finance, mixture, product }
  }, [state.theta, state.width])

  const experts: SeriesSpec[] = [
    { name: 'topic 1 (sport)', type: 'line', x: X, y: sport, slot: 0, dashed: true },
    { name: 'topic 2 (finance)', type: 'line', x: X, y: finance, slot: 1, dashed: true },
  ]
  const mixSeries: SeriesSpec[] = [{ name: 'mixture', type: 'bar', x: X, y: mixture, slot: 2 }, ...experts]
  const prodSeries: SeriesSpec[] = [{ name: 'product', type: 'bar', x: X, y: product, slot: 3 }, ...experts]
  const top = (p: number[]) => WORDS[p.indexOf(Math.max(...p))]

  const xAxis = useAxis({ label: 'word', hold: 'union' })
  const yAxis = useAxis({ label: 'p(w | θ)', range: Y_RANGE })
  const xAxis2 = useAxis({ label: 'word', hold: 'union' })
  const yAxis2 = useAxis({ label: 'p(w | θ)', range: Y_RANGE })
  return (
    <Figure
      title="Mixture against product of experts"
      state={state}
      caption={
        <MathText
          text={`Two topics over twelve words: ${WORDS.map((w, i) => `${i + 1} ${w}`).join(', ')}. Left: LDA's word distribution $\\theta_1\\betavec_1 + \\theta_2\\betavec_2$ spreads over both topics' words. Right: ProdLDA's $\\operatorname{softmax}(\\theta_1\\log\\betavec_1 + \\theta_2\\log\\betavec_2)$ concentrates on the words both topics give some weight, here the football transfer market, and is about as narrow as a single topic.`}
        />
      }

      readouts={
        <>
          <Readout label="perplexity of one topic" value={formatNumber(perplexity(sport))} />
          <Readout label="mixture" value={formatNumber(perplexity(mixture))} />
          <Readout label="product" value={formatNumber(perplexity(product))} />
          <Readout label="most probable word under the product" value={top(product)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={260}>
          {seriesLayers(mixSeries)}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={260}>
          {seriesLayers(prodSeries)}
        </Plot>
      </div>
    </Figure>
  )
}
