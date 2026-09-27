import { useMemo, useState } from 'react'
import { MathText } from '@/components/content/MathText'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'

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
  const [theta, setTheta] = useState(0.5)
  const [width, setWidth] = useState(1.6)

  const { sport, finance, mixture, product } = useMemo(() => {
    const lA = logTopic(3, width)
    const lB = logTopic(9, width)
    const sport = softmax(lA)
    const finance = softmax(lB)
    // LDA: average the probabilities. ProdLDA with B = log β: average the logits, then renormalise.
    const mixture = sport.map((p, i) => theta * p + (1 - theta) * finance[i])
    const product = softmax(lA.map((l, i) => theta * l + (1 - theta) * lB[i]))
    return { sport, finance, mixture, product }
  }, [theta, width])

  const experts: XYSeries[] = [
    { name: 'topic 1 (sport)', type: 'line', x: X, y: sport, slot: 0, dashed: true },
    { name: 'topic 2 (finance)', type: 'line', x: X, y: finance, slot: 1, dashed: true },
  ]
  const mixSeries: XYSeries[] = [{ name: 'mixture', type: 'bar', x: X, y: mixture, slot: 2 }, ...experts]
  const prodSeries: XYSeries[] = [{ name: 'product', type: 'bar', x: X, y: product, slot: 3 }, ...experts]
  const top = (p: number[]) => WORDS[p.indexOf(Math.max(...p))]

  return (
    <Interactive
      title="Mixture against product of experts"
      caption={
        <MathText
          text={`Two topics over twelve words: ${WORDS.map((w, i) => `${i + 1} ${w}`).join(', ')}. Left: LDA's word distribution $\\theta_1\\betavec_1 + \\theta_2\\betavec_2$ spreads over both topics' words. Right: ProdLDA's $\\operatorname{softmax}(\\theta_1\\log\\betavec_1 + \\theta_2\\log\\betavec_2)$ concentrates on the words both topics give some weight, here the football transfer market, and is about as narrow as a single topic.`}
        />
      }
      controls={
        <>
          <ParamSlider label="θ₁ (weight of topic 1)" value={theta} onChange={setTheta} min={0} max={1} step={0.01} />
          <ParamSlider label="topic width" value={width} onChange={setWidth} min={0.8} max={3} step={0.1} />
        </>
      }
      readout={
        <>
          <Readout label="perplexity of one topic" value={formatNumber(perplexity(sport))} />
          <Readout label="mixture" value={formatNumber(perplexity(mixture))} />
          <Readout label="product" value={formatNumber(perplexity(product))} />
          <Readout label="most probable word under the product" value={top(product)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <XYChart series={mixSeries} xLabel="word" yLabel="p(w | θ)" yRange={Y_RANGE} height={260} />
        <XYChart series={prodSeries} xLabel="word" yLabel="p(w | θ)" yRange={Y_RANGE} height={260} />
      </div>
    </Interactive>
  )
}
