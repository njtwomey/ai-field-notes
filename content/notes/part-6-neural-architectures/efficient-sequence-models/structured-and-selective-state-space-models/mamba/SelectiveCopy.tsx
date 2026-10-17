import { useMemo } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Plots,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { integers, normal, permutation, stream } from 'aifn-compute/foundation/random'
import { mul, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { discretiseDiagonal, linearRecurrence, selectiveScan } from 'aifn-compute/nn/sequence'

const L = 40
const STEPS = Array.from({ length: L }, (_, t) => t + 1)

type Rule = 'zoh' | 'euler'

/** A selective-copying input: four marked tokens with values in {−2, −1, 1, 2} at seeded positions, noise elsewhere. */
function makeTokens(seed: number) {
  const r = stream(seed)
  // Marked positions among tokens 2 to L − 1 (indices 1 to L − 2), values in {−2, −1, 1, 2}.
  const order = toFlat(permutation(r, L - 2)).map((p) => p + 1)
  const marked = new Set(order.slice(0, 4))
  const VALUES = [-2, -1, 1, 2]
  const x = STEPS.map((_, i) => (marked.has(i) ? VALUES[integers(r, 4)] : 1.5 * normal(r)))
  const isMarked = STEPS.map((_, i) => marked.has(i))
  // The value a copying memory should hold: the most recent marked token (0 before the first).
  let last = 0
  const target = x.map((v, i) => (isMarked[i] ? (last = v) : last))
  return { x, isMarked, target }
}

const RULES = [
  { value: 'zoh', label: 'zero-order hold (gate form)' },
  { value: 'euler', label: 'Euler, B̄ = ΔB (Mamba kernel)' },
] as const

/**
 * One channel of a selective SSM with N = 1, A = −1, B = C = 1. The step Δ_t is large on marked tokens and small on
 * the others; a time-invariant layer uses one fixed Δ. The selective state is computed by the engine's
 * `selectiveScan` (zero-order hold for A, Euler for B, as Mamba's kernel) or, for the gate form, by `discretiseDiagonal`
 * with the zero-order hold for both and `linearRecurrence`.
 */
export function SelectiveCopy() {
  const state = useFigureState({
    keep: float(4, { min: 0.1, max: 10, scale: 'log10', label: 'Δ on marked tokens', suggestions: [0.5, 1, 4, 10] }),
    skip: float(0.01, {
      min: 0.0001,
      max: 1,
      scale: 'log10',
      label: 'Δ on other tokens',
      suggestions: [0.001, 0.01, 0.1],
    }),
    fixed: float(0.7, { min: 0.01, max: 10, scale: 'log10', label: 'Δ of the time-invariant layer' }),
    rule: choice<Rule>(RULES, 'zoh', { label: 'discretisation of B' }),
    seed: int(3, { min: 0, max: 99, label: 'token seed', suggestions: [1, 2, 3] }),
  })
  const { keep, skip, fixed, rule, seed } = state
  const tokens = useMemo(() => makeTokens(seed), [seed])

  const traces = useMemo(() => {
    const x = tensor(tokens.x.map((v) => [v]))
    const steps = tensor(tokens.isMarked.map((m) => [m ? keep : skip]))
    const ones = tensor(tokens.x.map(() => [1]))
    const a = tensor([-1])
    const b = tensor([1])
    let selective: number[]
    if (rule === 'euler') {
      selective = toFlat(selectiveScan(x, steps, tensor([[-1]]), ones, ones).outputs as Tensor)
    } else {
      const d = discretiseDiagonal(a, b, steps, 'zoh')
      selective = toFlat(linearRecurrence(d.A, mul(d.B, x)) as Tensor)
    }
    const lti = discretiseDiagonal(a, b, fixed, 'zoh')
    const invariant = toFlat(linearRecurrence(lti.A, mul(lti.B, x)) as Tensor)
    // The weight a marked token is written with: 1 − e^{−Δ} under the zero-order hold, Δ under Euler.
    const wSel = rule === 'euler' ? keep : 1 - Math.exp(-keep)
    const wLti = 1 - Math.exp(-fixed)
    const first = tokens.isMarked.indexOf(true)
    // The held value decoded by dividing by the write weight (an output map C = 1/w), against the last marked value.
    const rms = (h: number[], w: number) => {
      let s = 0
      for (let i = first; i < L; i++) s += (h[i] / w - tokens.target[i]) ** 2
      return Math.sqrt(s / (L - first))
    }
    return { selective, invariant, wSel, wLti, errSel: rms(selective, wSel), errLti: rms(invariant, wLti) }
  }, [tokens, keep, skip, fixed, rule])

  const markedX = STEPS.filter((_, i) => tokens.isMarked[i])
  const markedY = tokens.x.filter((_, i) => tokens.isMarked[i])
  const noiseX = STEPS.filter((_, i) => !tokens.isMarked[i])
  const noiseY = tokens.x.filter((_, i) => !tokens.isMarked[i])

  const xAxis = useAxis({ label: 'token t', range: [0.5, L + 0.5], integer: true })
  const tokAxis = useAxis({ label: 'x_t', hold: 'union', key: seed })
  const hAxis = useAxis({ label: 'state h_t', hold: 'union', key: `${seed}-${rule}` })

  return (
    <Figure
      title="Selection lets a recurrence copy chosen tokens"
      state={state}
      caption="Selective copying with one state (N = 1, A = −1, B = C = 1). Four marked tokens (coloured bars) sit among noise tokens, and a copying memory should hold the most recent marked value (dashed). The selective layer takes a large step Δ on marked tokens, which overwrites the state, and a small step on the others, which keeps it. The time-invariant layer uses one Δ for every token, so noise enters its state whatever Δ is: a small Δ averages marked and noise tokens alike, and a large Δ overwrites the marked value at the next noise token. The errors divide the state by the weight w with which a marked token is written, as an output map C = 1/w would. Under the Euler rule of Mamba's kernel that weight is Δ rather than 1 − e^{−Δ}, so the curve is scaled by Δ but the decoded error is almost the same."
      readouts={
        <>
          <Readout label="selective: write weight" value={formatNumber(traces.wSel)} />
          <Readout label="selective: RMS error of h_t / w" value={formatNumber(traces.errSel)} />
          <Readout label="time-invariant: gate 1 − e^{−Δ}" value={formatNumber(traces.wLti)} />
          <Readout label="time-invariant: RMS error of h_t / w" value={formatNumber(traces.errLti)} />
        </>
      }
    >
      <Plots rows={2} heights={[1, 2]}>
        <Plot x={xAxis} y={tokAxis}>
          <Bars name="noise token" x={noiseX} y={noiseY} muted />
          <Bars name="marked token" x={markedX} y={markedY} slot={2} />
        </Plot>
        <Plot x={xAxis} y={hAxis}>
          <Curve name="last marked value" x={STEPS} y={tokens.target} emphasis dashed />
          <Curve name="selective" x={STEPS} y={traces.selective} slot={0} showPoints />
          <Curve name="time-invariant" x={STEPS} y={traces.invariant} slot={1} showPoints />
        </Plot>
      </Plots>
    </Figure>
  )
}
