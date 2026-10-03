import { useMemo } from 'react'
import { MathText } from 'aifn-render'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'

type Word = { w: string; x: number; y: number; g: number }

// A hand-placed two-dimensional embedding space: three semantic regions and function words near the origin.
const WORDS: Word[] = [
  { w: 'goal', x: 2.6, y: 1.9, g: 0 },
  { w: 'match', x: 2.2, y: 2.4, g: 0 },
  { w: 'striker', x: 3.0, y: 2.3, g: 0 },
  { w: 'league', x: 2.0, y: 1.6, g: 0 },
  { w: 'coach', x: 1.6, y: 2.1, g: 0 },
  { w: 'stadium', x: 2.7, y: 1.3, g: 0 },
  { w: 'bank', x: -2.4, y: 1.7, g: 1 },
  { w: 'loan', x: -2.9, y: 1.2, g: 1 },
  { w: 'shares', x: -2.1, y: 2.3, g: 1 },
  { w: 'market', x: -1.6, y: 1.8, g: 1 },
  { w: 'profit', x: -2.6, y: 2.6, g: 1 },
  { w: 'interest', x: -3.0, y: 0.6, g: 1 },
  { w: 'rain', x: 0.2, y: -2.8, g: 2 },
  { w: 'storm', x: 0.8, y: -2.5, g: 2 },
  { w: 'wind', x: -0.5, y: -2.4, g: 2 },
  { w: 'forecast', x: -0.1, y: -1.9, g: 2 },
  { w: 'cloud', x: 0.6, y: -3.1, g: 2 },
  { w: 'flood', x: -0.9, y: -2.9, g: 2 },
  { w: 'transfer', x: 0.3, y: 2.4, g: 0 },
  { w: 'sponsor', x: 0.1, y: 1.7, g: 1 },
  { w: 'the', x: 0.2, y: 0.1, g: 3 },
  { w: 'of', x: -0.2, y: 0.3, g: 3 },
  { w: 'and', x: 0.1, y: -0.3, g: 3 },
  { w: 'said', x: -0.4, y: -0.1, g: 3 },
]
const GROUPS = ['sport', 'finance', 'weather', 'function words']
const RANGE: [number, number] = [-4, 4]
const TOP = 5

/** ETM topic–word distribution: β_kv ∝ exp(ρ_vᵀ α_k), here with a two-dimensional topic vector α_k. */
export function EtmTopicVector() {
  const ax = useParam(2.2, { min: -4, max: 4, step: 0.05 })
  const ay = useParam(1.8, { min: -4, max: 4, step: 0.05 })

  const beta = useMemo(() => {
    const logits = WORDS.map((d) => d.x * ax.value + d.y * ay.value)
    const m = Math.max(...logits)
    const e = logits.map((l) => Math.exp(l - m))
    const s = e.reduce((a, b) => a + b, 0)
    return e.map((v) => v / s)
  }, [ax.value, ay.value])

  const order = beta.map((p, i) => [p, i] as const).sort((a, b) => b[0] - a[0])
  const top = order.slice(0, TOP).map(([, i]) => i)
  const entropy = -beta.reduce((h, p) => (p > 0 ? h + p * Math.log(p) : h), 0)

  const series: XYSeries[] = [
    {
      name: 'words',
      type: 'scatter',
      x: WORDS.map((d) => d.x),
      y: WORDS.map((d) => d.y),
      group: WORDS.map((d) => d.g),
      groupNames: GROUPS,
    },
    {
      name: `top ${TOP} words`,
      type: 'scatter',
      x: top.map((i) => WORDS[i].x),
      y: top.map((i) => WORDS[i].y),
      emphasis: true,
    },
  ]
  const handles: Handle[] = [
    {
      kind: 'point',
      at: [ax.value, ay.value],
      label: 'topic vector α',
      onDrag: ([x, y]) => {
        ax.set(x)
        ay.set(y)
      },
    },
  ]

  return (
    <Interactive
      title="A topic is a vector in word-embedding space"
      caption={
        <MathText text="Each point is a word embedding $\rhovec_v$; the arrow is a topic embedding $\alphavec_k$. Topic $k$ gives word $v$ probability $\beta_{kv} \propto \exp(\rhovec_v^\top\alphavec_k)$, so the top words (ink diamonds) are those furthest along the arrow's direction. Drag the arrow's tip. A longer arrow makes the topic sharper; a short one spreads it over the whole vocabulary. Function words sit near the origin, so their inner product with any topic vector is near zero and they rarely top a topic." />
      }
      controls={
        <>
          <ParamSlider label="α₁" param={ax} />
          <ParamSlider label="α₂" param={ay} />
        </>
      }
      readout={
        <>
          <Readout
            label="top words"
            value={order
              .slice(0, TOP)
              .map(([p, i]) => `${WORDS[i].w} ${p.toFixed(3)}`)
              .join(', ')}
          />
          <Readout label="perplexity of the topic" value={formatNumber(Math.exp(entropy))} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-md">
        <XYChart
          series={series}
          vectors={[{ from: [0, 0], to: [ax.value, ay.value] }]}
          handles={handles}
          xRange={RANGE}
          yRange={RANGE}
          xLabel="embedding dimension 1"
          yLabel="embedding dimension 2"
          equalAspect
        />
      </div>
    </Interactive>
  )
}
