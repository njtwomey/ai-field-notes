import { useMemo } from 'react'
import { Figure, float, formatNumber, int, Plot, Points, Raster, Readout, useAxis, useFigureState } from 'aifn-render'
import { forwardBackward, viterbi, type Mat, type Vec } from '../_shared/chain-crf'
import { normal, stream, uniform } from 'aifn/foundation/random'

const N = 30
const K = 3
const POSITIONS = Array.from({ length: N }, (_, n) => n + 1)
const LABELS = Array.from({ length: K }, (_, k) => k)

const softmax = (z: Vec) => {
  const m = Math.max(...z)
  const e = z.map((v) => Math.exp(v - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}
const argmax = (v: Vec) => v.indexOf(Math.max(...v))

/**
 * Stage one scores each position on its own; stage two is a chain CRF whose node potentials are those scores and whose
 * edge potential couples neighbours. Forward–backward then propagates each position's uncertainty along the chain.
 */
export function ScoreSmoothing() {
  const state = useFigureState({
    signal: float(1.2, { min: 0.2, max: 3, step: 0.1, label: 'stage-one signal strength' }),
    stay: float(0.85, { min: 0.34, max: 0.98, step: 0.01, label: 'P(label stays the same)' }),
    coupling: float(1, { min: 0, max: 3, step: 0.1, label: 'CRF coupling (× matched weight)' }),
    seed: int(4, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const r = useMemo(() => {
    const g = stream(state.seed)
    // True labels: a Markov chain that stays in its label with probability `stay`.
    const truth: number[] = [Math.floor(uniform(g) * K)]
    for (let n = 1; n < N; n++) {
      const prev = truth[n - 1]
      truth.push(uniform(g) < state.stay ? prev : (prev + 1 + Math.floor(uniform(g) * (K - 1))) % K)
    }
    // Stage one: per-position class probabilities from noisy scores, independent across positions.
    const scores: Vec[] = truth.map((t) => LABELS.map((k) => (k === t ? state.signal : 0) + normal(g)))
    const stage1 = scores.map(softmax)
    // Stage two: ψ_n = stage-one probabilities; Ψ(u, v) = exp(w [u = v]). The generating chain's own log-odds of
    // staying is the natural weight; the coupling slider scales it (0 = no structure, 1 = matched, > 1 = over-smooth).
    const matched = Math.log(state.stay / ((1 - state.stay) / (K - 1)))
    const w = state.coupling * matched
    const edge: Mat = LABELS.map((u) => LABELS.map((v) => Math.exp(u === v ? w : 0)))
    const edges = Array.from({ length: N - 1 }, () => edge)
    const inference = forwardBackward(stage1, edges)
    const path = viterbi(stage1, edges).path
    const accuracy = (pred: number[]) => pred.filter((p, n) => p === truth[n]).length / N
    return {
      truth,
      stage1,
      marginals: inference.marginals,
      acc1: accuracy(stage1.map(argmax)),
      accMarginal: accuracy(inference.marginals.map(argmax)),
      accViterbi: accuracy(path),
      w,
    }
  }, [state.signal, state.stay, state.coupling, state.seed])

  // Raster rows are labels and columns positions: z[k][n].
  const grid = (probs: Vec[]) => LABELS.map((k) => probs.map((p) => p[k]))
  // Both panels share the position and label axes.
  const xAxis = useAxis({ label: 'position n' })
  const yAxis = useAxis({ label: 'label', integer: true })
  const panel = (title: string, probs: Vec[]) => (
    <div className="min-w-0 space-y-1">
      <div className="text-center text-xs text-muted-foreground">{title}</div>
      <Plot x={xAxis} y={yAxis} height={220}>
        <Raster x={POSITIONS} y={LABELS} z={grid(probs)} range={[0, 1]} valueLabel="probability" />
        <Points name="true label" x={POSITIONS} y={r.truth} emphasis live />
      </Plot>
    </div>
  )

  return (
    <Figure
      title="Scores in, uncertainty propagated"
      state={state}
      caption="Stage one classifies each position on its own, giving noisy class probabilities (top). Stage two is a chain CRF whose node potentials are those probabilities and whose edge potential rewards staying in the same label; forward–backward turns them into marginals that use the whole sequence (bottom). Diamonds mark the true labels. Isolated stage-one mistakes are corrected by their neighbours. Here the CRF weights are set from the generating chain rather than learned, and the coupling slider scales them: 0 removes the structure, values above 1 over-smooth and erase short runs."
      readouts={
        <>
          <Readout label="stage one alone" value={`${formatNumber(100 * r.acc1)}%`} />
          <Readout label="CRF marginals" value={`${formatNumber(100 * r.accMarginal)}%`} />
          <Readout label="CRF Viterbi path" value={`${formatNumber(100 * r.accViterbi)}%`} />
          <Readout label="edge weight w" value={formatNumber(r.w)} />
        </>
      }
    >
      <div className="space-y-4">
        {panel('stage one: per-position class probabilities', r.stage1)}
        {panel('stage two: CRF marginals P(Y_n | whole sequence)', r.marginals)}
      </div>
    </Figure>
  )
}
