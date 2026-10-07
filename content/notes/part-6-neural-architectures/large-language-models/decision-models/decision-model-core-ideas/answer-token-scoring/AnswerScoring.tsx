import { useMemo } from 'react'
import { Bars, Figure, float, formatNumber, Plot, Readout, setting, slider, useAxis, useFigureState } from 'aifn-render'

/** Next-token logits at the answer boundary for tokens that are not answer codes; fixed, as a trained model sets them. */
const OTHERS = [
  { token: 'The', logit: 1.2 },
  { token: '{', logit: 0.4 },
  { token: 'Based', logit: 0.9 },
  { token: 'I', logit: -0.3 },
  { token: '"', logit: 0.1 },
] as const

function softmax(z: readonly number[], t: number): number[] {
  const m = Math.max(...z)
  const e = z.map((v) => Math.exp((v - m) / t))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

/**
 * One question with options A, B, C (and an optional "none" option D). The left panel is the model's next-token
 * distribution over a small vocabulary; the right panel is the decision: a softmax over the code logits alone.
 */
export function AnswerScoring() {
  const state = useFigureState({
    zA: slider(-2, 6, 3.5, { step: 0.1, label: 'logit of A' }),
    zB: slider(-2, 6, 2.6, { step: 0.1, label: 'logit of B' }),
    zC: slider(-2, 6, 0.5, { step: 0.1, label: 'logit of C' }),
    none: setting(false, 'add a "none" option D'),
    zD: slider(-2, 6, 3.0, { step: 0.1, label: 'logit of D' }),
    t: float(1, { min: 0.25, max: 4, step: 0.05, label: 'temperature T', suggestions: [0.5, 1, 2.179] }),
  })
  const codes = useMemo(() => (state.none ? ['A', 'B', 'C', 'D'] : ['A', 'B', 'C']), [state.none])
  const codeLogits = useMemo(
    () => (state.none ? [state.zA, state.zB, state.zC, state.zD] : [state.zA, state.zB, state.zC]),
    [state.none, state.zA, state.zB, state.zC, state.zD],
  )

  const vocab = useMemo(() => [...codes, ...OTHERS.map((o) => o.token)], [codes])
  const full = useMemo(() => softmax([...codeLogits, ...OTHERS.map((o) => o.logit)], 1), [codeLogits])
  const q = useMemo(() => softmax(codeLogits, state.t), [codeLogits, state.t])

  const n = codes.length
  const k = q.indexOf(Math.max(...q))
  const mass = full.slice(0, n).reduce((a, b) => a + b, 0)
  const entropy = -q.reduce((a, p) => a + (p > 0 ? p * Math.log(p) : 0), 0)
  const typesafe = (q[k] - 1 / n) / (1 - 1 / n)
  const entropyConf = 1 - entropy / Math.log(n)

  const vx = useAxis({ label: 'next token', categories: vocab, key: n })
  const vy = useAxis({ label: 'probability over the vocabulary (T = 1)', range: [0, 1] })
  const cx = useAxis({ label: 'option code', categories: codes, key: n })
  const cy = useAxis({ label: 'decision probability q', range: [0, 1] })
  const codeIdx = useMemo(() => codes.map((_, i) => i), [codes])
  const otherIdx = useMemo(() => OTHERS.map((_, i) => n + i), [n])

  return (
    <Figure
      title="Reading a decision from next-token logits"
      state={state}
      caption="Left: the model's next-token distribution at the end of the prompt, where the answer would begin. Answer codes are coloured; other tokens are grey. Right: the decision, a softmax over the code logits alone, divided by T. The grey tokens enter only the left panel: however much probability they take, the decision on the right does not change. Adding a 'none' option takes probability from A, B and C because the decision always sums to 1 over the options offered."
      readouts={
        <>
          <Readout label="answer" value={codes[k]} />
          <Readout label="mass on codes (T = 1)" value={formatNumber(mass)} />
          <Readout label="(p_max − 1/n)/(1 − 1/n)" value={formatNumber(typesafe)} />
          <Readout label="1 − H/ln n" value={formatNumber(entropyConf)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[3fr_2fr]">
        <Plot x={vx} y={vy} height={260}>
          <Bars name="answer codes" x={codeIdx} y={full.slice(0, n)} slot={0} />
          <Bars name="other tokens" x={otherIdx} y={full.slice(n)} muted />
        </Plot>
        <Plot x={cx} y={cy} height={260}>
          <Bars name="decision q" x={codeIdx} y={q} slot={0} />
        </Plot>
      </div>
    </Figure>
  )
}
