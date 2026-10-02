import { useMemo } from 'react'
import {
  Heatmap,
  Interactive,
  ParamSlider,
  Readout,
  formatNumber,
  useParam,
  type HeatmapOverlay,
} from 'aifn-render'
import { rng } from '@/lib/math'
import { forwardBackward, viterbi, type Mat, type Vec } from '../_shared/chain-crf'

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
  const signal = useParam(1.2, { min: 0.2, max: 3, step: 0.1 })
  const stay = useParam(0.85, { min: 0.34, max: 0.98, step: 0.01 })
  const coupling = useParam(1, { min: 0, max: 3, step: 0.1 })
  const seed = useParam(4, { min: 1, max: 30, step: 1 })

  const r = useMemo(() => {
    const g = rng(seed.value)
    // True labels: a Markov chain that stays in its label with probability `stay`.
    const truth: number[] = [Math.floor(g.uniform() * K)]
    for (let n = 1; n < N; n++) {
      const prev = truth[n - 1]
      truth.push(g.uniform() < stay.value ? prev : (prev + 1 + Math.floor(g.uniform() * (K - 1))) % K)
    }
    // Stage one: per-position class probabilities from noisy scores, independent across positions.
    const scores: Vec[] = truth.map((t) => LABELS.map((k) => (k === t ? signal.value : 0) + g.normal()))
    const stage1 = scores.map(softmax)
    // Stage two: ψ_n = stage-one probabilities; Ψ(u, v) = exp(w [u = v]). The generating chain's own log-odds of
    // staying is the natural weight; the coupling slider scales it (0 = no structure, 1 = matched, > 1 = over-smooth).
    const matched = Math.log(stay.value / ((1 - stay.value) / (K - 1)))
    const w = coupling.value * matched
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
  }, [signal.value, stay.value, coupling.value, seed.value])

  // Heatmap rows are labels and columns positions: z[k][n].
  const grid = (probs: Vec[]) => LABELS.map((k) => probs.map((p) => p[k]))
  const truthMarks: HeatmapOverlay[] = [
    { name: 'true label', type: 'scatter', x: POSITIONS, y: r.truth, emphasis: true },
  ]
  const panel = (title: string, probs: Vec[]) => (
    <div className="min-w-0 space-y-1">
      <div className="text-center text-xs text-muted-foreground">{title}</div>
      <Heatmap
        x={POSITIONS}
        y={LABELS}
        z={grid(probs)}
        range={[0, 1]}
        xLabel="position n"
        yLabel="label"
        valueLabel="probability"
        overlay={truthMarks}
        height={220}
      />
    </div>
  )

  return (
    <Interactive
      title="Scores in, uncertainty propagated"
      caption="Stage one classifies each position on its own, giving noisy class probabilities (top). Stage two is a chain CRF whose node potentials are those probabilities and whose edge potential rewards staying in the same label; forward–backward turns them into marginals that use the whole sequence (bottom). Diamonds mark the true labels. Isolated stage-one mistakes are corrected by their neighbours. Here the CRF weights are set from the generating chain rather than learned, and the coupling slider scales them: 0 removes the structure, values above 1 over-smooth and erase short runs."
      controls={
        <>
          <ParamSlider label="stage-one signal strength" param={signal} />
          <ParamSlider label="P(label stays the same)" param={stay} />
          <ParamSlider label="CRF coupling (× matched weight)" param={coupling} />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
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
    </Interactive>
  )
}
