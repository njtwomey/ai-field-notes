import { useMemo, useState } from 'react'
import { Heatmap, Interactive, ParamSlider, Readout, StepControls, XYChart, formatNumber } from 'aifn-render'
import { useDerivedState } from '@/lib/use-derived-state'
import {
  K,
  TRUE_TOPICS,
  V,
  corpus,
  distanceToTruth,
  initialise,
  logLikelihood,
  sweep,
  topicWord,
  type State,
} from './gibbs'

const WORDS = Array.from({ length: V }, (_, w) => w + 1)
const TOPICS = Array.from({ length: K }, (_, k) => k + 1)
const RUN = 25

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

type Run = State & { history: number[] }

export function GibbsBars() {
  const docs = useMemo(() => corpus(200, 40, 5), [])
  const [alpha, setAlpha] = useState(0.3)
  const [beta, setBeta] = useState(0.1)
  const [seed, setSeed] = useState(1)
  const initial = useMemo((): Run => {
    const s = initialise(docs, seed)
    return { ...s, history: [logLikelihood(s, beta)] }
  }, [docs, seed, beta])
  const [state, setState, reset] = useDerivedState(initial)
  const advance = (s: Run, sweeps: number): Run => {
    let next: Run = s
    for (let i = 0; i < sweeps; i++) {
      const t = sweep(docs, next, alpha, beta)
      next = { ...t, history: [...next.history, logLikelihood(t, beta)] }
    }
    return next
  }
  const phi = topicWord(state, beta)
  const shown = alignToTruth(phi)
  const sweeps = state.history.map((_, i) => i)

  return (
    <Interactive
      title="Collapsed Gibbs sampling recovers the bars topics"
      caption="The vocabulary is nine words on a 3 × 3 grid. The six true topics (left) are the three rows and three columns of the grid. 200 documents of 40 words were generated from them. Starting from random assignments, each sweep resamples every token's topic. Step through the first sweeps: the learned topic–word distributions (middle, matched to the true topics for display) sharpen into bars within about 20 sweeps, and the log-likelihood (right) levels off."
      controls={
        <>
          <ParamSlider
            label="α (document–topic prior)"
            value={alpha}
            onChange={setAlpha}
            min={0.05}
            max={2}
            step={0.05}
          />
          <ParamSlider label="β (topic–word prior)" value={beta} onChange={setBeta} min={0.01} max={1} step={0.01} />
          <ParamSlider label="seed" value={seed} onChange={setSeed} min={1} max={10} step={1} />
          <StepControls
            onStep={() => setState((s) => advance(s, 1))}
            onRun={() => setState((s) => advance(s, RUN))}
            onReset={reset}
            done={state.sweep >= 500}
          />
        </>
      }
      readout={
        <>
          <Readout label="sweeps" value={state.sweep} />
          <Readout label="log p(w | z)" value={formatNumber(state.history[state.history.length - 1])} />
          <Readout label="mean distance to nearest true topic" value={formatNumber(distanceToTruth(phi))} />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-3">
        <Heatmap
          x={WORDS}
          y={TOPICS}
          z={TRUE_TOPICS}
          xLabel="word"
          yLabel="true topic"
          range={[0, 0.4]}
          valueLabel="φ"
          height={280}
        />
        <Heatmap
          x={WORDS}
          y={TOPICS}
          z={shown}
          xLabel="word"
          yLabel="learned topic"
          range={[0, 0.4]}
          valueLabel="φ"
          height={280}
        />
        <XYChart
          height={280}
          xLabel="sweep"
          yLabel="log p(w | z)"
          series={[{ name: 'log-likelihood', type: 'line', x: sweeps, y: state.history, slot: 0 }]}
        />
      </div>
    </Interactive>
  )
}
