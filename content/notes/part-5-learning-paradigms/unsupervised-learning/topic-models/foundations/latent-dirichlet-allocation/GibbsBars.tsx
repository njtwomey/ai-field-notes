import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  int,
  Player,
  Plot,
  Raster,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { K, TRUE_TOPICS, V, corpus, distanceToTruth, initialise, logLikelihood, sweep, topicWord } from './gibbs'

const WORDS = Array.from({ length: V }, (_, w) => w + 1)
const TOPICS = Array.from({ length: K }, (_, k) => k + 1)
/** Sweeps computed up front for the walk-through. */
const SWEEPS = 60

/** Topic labels are arbitrary; greedily pair each learned topic with its closest unused true topic for display. */
function alignToTruth(phi: number[][]): number[][] {
  const pairs = phi.flatMap((row, k) =>
    TRUE_TOPICS.map((t, j) => ({ k, j, d: row.reduce((s, v, w) => s + Math.abs(v - t[w]), 0) })),
  )
  pairs.sort((a, b) => a.d - b.d)
  const slot = new Array<number>(K).fill(-1)
  const used = new Set<number>()
  for (const { k, j } of pairs) {
    if (slot[j] !== -1 || used.has(k)) continue
    slot[j] = k
    used.add(k)
  }
  return slot.map((k) => phi[k])
}

/** What the figure shows after each sweep: the aligned topic–word table, the log-likelihood, the distance to the truth. */
type Snapshot = { shown: number[][]; ll: number; distance: number }

export function GibbsBars() {
  const docs = useMemo(() => corpus(200, 40, 5), [])
  const state = useFigureState({
    alpha: slider(0.05, 2, 0.3, { step: 0.05, label: 'α (document–topic prior)' }),
    beta: slider(0.01, 1, 0.1, { step: 0.01, label: 'β (topic–word prior)' }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })
  const { alpha, beta, seed } = state
  const trace = useMemo(() => {
    const snap = (s: ReturnType<typeof initialise>): Snapshot => {
      const phi = topicWord(s, beta)
      return { shown: alignToTruth(phi), ll: logLikelihood(s, beta), distance: distanceToTruth(phi) }
    }
    let s = initialise(docs, seed)
    const out = [snap(s)]
    for (let i = 0; i < SWEEPS; i++) {
      s = sweep(docs, s, alpha, beta)
      out.push(snap(s))
    }
    return out
  }, [docs, seed, alpha, beta])
  // Sweeps made: the walk-through position. It counts sweeps, so it keeps its meaning when the priors change.
  const [step, setStep] = useState(0)
  const now = trace[step]
  const sweeps = useMemo(() => trace.slice(0, step + 1).map((_, i) => i), [trace, step])
  const history = useMemo(() => trace.slice(0, step + 1).map((t) => t.ll), [trace, step])

  const xAxis = useAxis({ label: 'word' })
  const yAxis = useAxis({ label: 'true topic' })
  const xAxis2 = useAxis({ label: 'word' })
  const yAxis2 = useAxis({ label: 'learned topic' })
  const xAxis3 = useAxis({ label: 'sweep', range: [0, SWEEPS] })
  const yAxis3 = useAxis({ label: 'log p(w | z)', hold: 'union' })
  return (
    <Figure
      title="Collapsed Gibbs sampling recovers the bars topics"
      state={state}
      caption="The vocabulary is nine words on a 3 × 3 grid. The six true topics (left) are the three rows and three columns of the grid. 200 documents of 40 words were generated from them. Starting from random assignments, each sweep resamples every token's topic. Play through the first sweeps: the learned topic–word distributions (middle, matched to the true topics for display) sharpen into bars within about 20 sweeps, and the log-likelihood (right) levels off."
      controls={
        <Player value={step} onChange={setStep} count={SWEEPS + 1} label="sweeps" format={(k) => `${k} of ${SWEEPS}`} />
      }
      readouts={
        <>
          <Readout label="sweeps" value={step} />
          <Readout label="log p(w | z)" value={formatNumber(now.ll)} />
          <Readout label="mean distance to nearest true topic" value={formatNumber(now.distance)} />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-3">
        <Plot x={xAxis} y={yAxis} height={280}>
          <Raster x={WORDS} y={TOPICS} z={TRUE_TOPICS} range={[0, 0.4]} valueLabel={'φ'} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={280}>
          <Raster x={WORDS} y={TOPICS} z={now.shown} range={[0, 0.4]} valueLabel={'φ'} />
        </Plot>
        <Plot x={xAxis3} y={yAxis3} height={280}>
          <Curve name="log-likelihood" x={sweeps} y={history} slot={0} />
        </Plot>
      </div>
    </Figure>
  )
}
