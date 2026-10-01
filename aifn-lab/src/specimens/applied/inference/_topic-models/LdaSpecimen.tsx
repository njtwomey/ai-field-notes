import { expandModel, nestedValues, sampleModel } from 'aifn/inference/model'
import { infer } from 'aifn/inference/engines'
import { ldaEngines, ldaEstimates, ldaModel, ldaOptions, type LdaState } from 'aifn-applied/inference/topic-models'
import { child, stream } from 'aifn/foundation/random'
import { trace, type Algorithm } from 'aifn/foundation/trace'
import { toFlat, toRows } from 'aifn/foundation/tensor'
import { useMemo, useState } from 'react'
import { Player } from '@lab/controls'
import { ControlRow, Dashboard, DashboardCell, DashboardRow, Figure } from '@lab/layout'
import { number, row, slider, useFigureState } from '@lab/state'
import { Curve, formatNumber, Plot, Plots, Points, Raster, Readout, useAxis } from '@lab/viz'

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
const TRUE_MOSAIC = mosaic(BARS)
const COLS = Array.from({ length: 4 * K - 1 }, (_, i) => i)
const ROWS = [0, 1, 2]

/** Collapsed Gibbs for LDA, chosen by `infer` from the model description, on a corpus drawn from the bars topics. */
export function LdaSpecimen() {
  const state = useFigureState({
    corpus: row('1 · corpus', {
      docs: slider(10, 200, 100, { label: 'documents', step: 10 }),
      seed: number(1, { min: 1, max: 20, step: 1, label: 'seed' }),
    }),
  })
  const { docs, seed } = state.corpus
  const corpus = useMemo(() => {
    const bindings = { sizes: { K, V, D: docs, N: 40 }, constants: { α: 0.3, β: 0.1 } }
    const values = sampleModel(child(stream('bars'), seed), MODEL, { ...bindings, given: { φ: BARS } })
    return nestedValues(expandModel(MODEL, bindings), 'w', values) as number[][]
  }, [docs, seed])
  const inference = useMemo(() => {
    const bindings = { sizes: { K, V }, constants: { α: 0.3, β: 0.1 }, data: { w: corpus } }
    // `infer` picks LDA's collapsed Gibbs engine from the model's shape when given LDA's engine table.
    return { ...infer(MODEL, bindings, { engines: ldaEngines }), options: ldaOptions(MODEL, bindings) }
  }, [corpus])
  const run = useMemo(
    () =>
      trace(inference.algorithm as Algorithm<void, LdaState>, undefined, 150, {
        stream: child(stream('lda'), seed),
        record: { ll: (s) => s.logLikelihood },
      }),
    [inference, seed],
  )
  const [step, setStep] = useState(0)
  const s = run.steps[Math.min(step, run.steps.length - 1)]
  const phi = useMemo(() => mosaic(toRows(ldaEstimates(s, inference.options).topicWord) as number[][]), [s, inference])
  const ll = useMemo(() => ({ x: Array.from(run.index), y: toFlat(run.series.ll) }), [run])
  const now = { x: [step], y: [ll.y[Math.min(step, ll.y.length - 1)]] }
  const tx = useAxis({ label: 'true topics' })
  const ty = useAxis({ label: 'row', equal: tx })
  const ex = useAxis({ label: 'estimated topics at this sweep' })
  const ey = useAxis({ label: 'row', equal: ex })
  const sweepAxis = useAxis({ label: 'sweep', hold: 'initial', key: run })
  const llAxis = useAxis({ label: 'log p(w | z)', hold: 'initial', key: run })
  return (
    <Figure
      title="Collapsed Gibbs sampling for LDA on the bars corpus"
      purpose="infer() recognises LDA's shape and runs its registered collapsed Gibbs engine; sweep by sweep the estimated topics turn into the bars."
      state={state}
      defaultSize="L"
      controls={
        <ControlRow label="2 · sweeps">
          <Player label="sweep" value={step} onChange={setStep} count={run.steps.length} />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="engine" value={inference.engine} />
          <Readout label="sweep" value={step} />
          <Readout label="log p(w | z)" value={formatNumber(s.logLikelihood)} />
        </>
      }
      caption="Left: the six true topics (top) and the estimates at this sweep (bottom), each a 3 × 3 image of word probabilities; right: log p(w | z) by sweep. Each document mixes the six bar topics with weights from Dir(0.3) and has 40 words, drawn by sampleModel from the LDA description with φ held at the bars. Each sweep resamples every word's topic from p(z = k | rest) ∝ (n_dk + α)(n_kw + β)/(n_k + Vβ). The topics are unordered, so the learned ones match the true ones up to a permutation."
    >
      <Dashboard>
        <DashboardRow minHeight={320}>
          <DashboardCell ratio={1.5}>
            <Plots rows={2}>
              <Plot x={tx} y={ty}>
                <Raster x={COLS} y={ROWS} z={TRUE_MOSAIC} range={[0, 0.4]} valueLabel="p(word | topic)" />
              </Plot>
              <Plot x={ex} y={ey}>
                <Raster x={COLS} y={ROWS} z={phi} range={[0, 0.4]} valueLabel="p(word | topic)" />
              </Plot>
            </Plots>
          </DashboardCell>
          <DashboardCell>
            <Plot x={sweepAxis} y={llAxis} legend={false}>
              <Curve name="log p(w | z)" x={ll.x} y={ll.y} slot={0} />
              <Points name="this sweep" x={now.x} y={now.y} emphasis />
            </Plot>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}
