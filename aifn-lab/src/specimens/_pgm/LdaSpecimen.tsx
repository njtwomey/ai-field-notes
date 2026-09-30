import { expandModel, infer, ldaEstimates, ldaModel, nestedValues, sampleModel, type LdaState } from 'aifn/pgm'
import { stream } from 'aifn/random'
import { trace, type Algorithm } from 'aifn/trace'
import { toFlat, toRows } from 'aifn/tensor'
import { useMemo, useState } from 'react'
import { Player, Slider, useParam } from '@lab/controls'
import { Figure } from '@lab/layout'
import { ChartSize, Heatmap, Readout, XYChart, formatNumber } from '@lab/viz'
import { Columns } from './common'

const K = 6
const V = 9
/** Griffiths & Steyvers' bars: on a 3 × 3 grid of words, three row topics and three column topics. */
const BARS = [
  ...[0, 1, 2].map((r) => Array.from({ length: V }, (_, w) => (Math.floor(w / 3) === r ? 1 / 3 : 0))),
  ...[0, 1, 2].map((c) => Array.from({ length: V }, (_, w) => (w % 3 === c ? 1 / 3 : 0))),
]
const MODEL = ldaModel()

/** Topics side by side as 3 × 3 images, separated by empty columns: a 3 × (4K − 1) grid. */
function mosaic(topics: number[][]): number[][] {
  return [0, 1, 2].map((r) =>
    topics.flatMap((t, k) => [...[0, 1, 2].map((c) => t[r * 3 + c]), ...(k < topics.length - 1 ? [NaN] : [])]),
  )
}
const COLS = Array.from({ length: 4 * K - 1 }, (_, i) => i)
const ROWS = [0, 1, 2]

/** Collapsed Gibbs for LDA, chosen by `infer` from the model description, on a corpus drawn from the bars topics. */
export function LdaSpecimen() {
  const docs = useParam(100, { min: 10, max: 200, step: 10 })
  const seed = useParam(1, { min: 1, max: 20, step: 1 })
  const corpus = useMemo(() => {
    const bindings = { sizes: { K, V, D: docs.value, N: 40 }, constants: { α: 0.3, β: 0.1 } }
    const values = sampleModel(stream('bars').child(seed.value), MODEL, { ...bindings, given: { φ: BARS } })
    return nestedValues(expandModel(MODEL, bindings), 'w', values) as number[][]
  }, [docs.value, seed.value])
  const inference = useMemo(
    () => infer(MODEL, { sizes: { K, V }, constants: { α: 0.3, β: 0.1 }, data: { w: corpus } }),
    [corpus],
  )
  const run = useMemo(
    () =>
      trace(inference.algorithm as Algorithm<unknown, LdaState>, inference.options, 150, {
        stream: stream('lda').child(seed.value),
        record: { ll: (s) => s.logLikelihood },
      }),
    [inference, seed.value],
  )
  const [step, setStep] = useState(150)
  const s = run.steps[Math.min(step, run.steps.length - 1)]
  const phi = toRows(ldaEstimates(s).topicWord)
  const ll = toFlat(run.series.ll)
  const heat = (topics: number[][]) => (
    <ChartSize scale={0.4}>
      <Heatmap x={COLS} y={ROWS} z={mosaic(topics)} range={[0, 0.4]} valueLabel="p(word | topic)" equalAspect />
    </ChartSize>
  )
  return (
    <Figure
      title="Collapsed Gibbs sampling for LDA on the bars corpus"
      description="infer() recognises LDA's shape and runs its registered collapsed Gibbs engine; the estimated topics turn into the bars."
      defaultSize="L"
      controls={
        <>
          <Slider label="documents" param={docs} />
          <Slider label="seed" param={seed} />
          <div className="col-span-full">
            <Player label="sweep" value={step} onChange={setStep} count={run.steps.length} />
          </div>
        </>
      }
      readouts={
        <>
          <Readout label="engine" value={inference.engine} />
          <Readout label="sweep" value={step} />
          <Readout label="log p(w | z)" value={formatNumber(s.logLikelihood)} />
        </>
      }
      caption="Each document mixes the six bar topics with weights from Dir(0.3) and has 40 words, drawn by sampleModel from the LDA description with φ held at the bars. Each sweep resamples every word's topic from p(z = k | rest) ∝ (n_dk + α)(n_kw + β)/(n_k + Vβ). The topics are unordered, so the learned ones match the true ones up to a permutation."
    >
      <Columns
        widths={[3, 2]}
        panels={[
          {
            title: 'true topics (top) and estimated topics φ at this sweep (bottom)',
            body: (
              <div className="flex h-full flex-col gap-2">
                {heat(BARS)}
                {heat(phi)}
              </div>
            ),
          },
          {
            title: 'log p(w | z) by sweep',
            body: (
              <ChartSize scale={0.85}>
                <XYChart
                  series={[
                    { name: 'log p(w | z)', type: 'line', x: run.index, y: ll, slot: 0 },
                    {
                      name: 'this sweep',
                      type: 'scatter',
                      x: [step],
                      y: [ll[Math.min(step, ll.length - 1)]],
                      emphasis: true,
                    },
                  ]}
                  xLabel="sweep"
                  yLabel="log p(w | z)"
                />
              </ChartSize>
            ),
          },
        ]}
      />
    </Figure>
  )
}
