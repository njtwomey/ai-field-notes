import { useMemo, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Player,
  Plot,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream, uniform } from 'aifn-compute/foundation/random'

const sigmoid = (u: number) => 1 / (1 + Math.exp(-u))
/** The true probability that the answer is "yes" in context x. */
const truth = (x: number) => sigmoid(1.5 * x)

const N = 500
const BATCH = 32
const GROUP = 4
const GRID = Array.from({ length: 121 }, (_, i) => -3 + i * 0.05)
const TRUE_CURVE = GRID.map(truth)

type Reward = 'correct' | 'coupled' | 'proper' | 'supervised'

/** Where each reward's expected value is maximised, as a function of the true probability p (see the note). */
const OPTIMUM: Record<Reward, (p: number) => number> = {
  correct: (p) => (p > 0.5 ? 1 : 0),
  coupled: (p) => Math.min(1, Math.max(0, 2 * p - 0.5)),
  proper: (p) => p,
  supervised: (p) => p,
}

/**
 * Policy-gradient training of a one-token yes/no decision π(yes | x) = σ(w x + b). Each step draws a batch of
 * contexts, samples a group of answers per context, scores each answer against the stored label, and moves (w, b)
 * along the score-function gradient with a leave-one-out group baseline, plus the exact gradient of any term that
 * depends on π directly. "supervised" is log-loss training on the label, for reference.
 */
function train(reward: Reward, lr: number, steps: number, seed: number): Float64Array {
  const r = stream(seed)
  const xs = Array.from({ length: N }, () => -3 + 6 * uniform(r))
  const ys = xs.map((x) => (uniform(r) < truth(x) ? 1 : 0))
  const trace = new Float64Array(2 * (steps + 1))
  let w = 0
  let b = 0
  for (let s = 0; s <= steps; s++) {
    trace[2 * s] = w
    trace[2 * s + 1] = b
    if (s === steps) break
    let gw = 0
    let gb = 0
    for (let i = 0; i < BATCH; i++) {
      const n = Math.floor(uniform(r) * N)
      const x = xs[n]
      const y = ys[n]
      const pi = sigmoid(w * x + b)
      const dpi = pi * (1 - pi) // dπ/du
      let gu = 0 // d(objective)/du for this context
      if (reward === 'supervised') {
        gu = y - pi
      } else {
        const acts: number[] = []
        const rews: number[] = []
        for (let g = 0; g < GROUP; g++) {
          const a = uniform(r) < pi ? 1 : 0
          const c = a === y ? 1 : 0
          const pa = a === 1 ? pi : 1 - pi
          acts.push(a)
          rews.push(
            reward === 'correct' ? c : reward === 'coupled' ? c - (pa - c) ** 2 : 2 * c - pi * pi - (1 - pi) ** 2,
          )
          // Pathwise part: the coupled reward depends on π_a itself.
          if (reward === 'coupled') gu += (-2 * (pa - c) * (a === 1 ? dpi : -dpi)) / GROUP
        }
        const total = rews.reduce((t, v) => t + v, 0)
        for (let g = 0; g < GROUP; g++) {
          const adv = rews[g] - (total - rews[g]) / (GROUP - 1)
          gu += (adv * (acts[g] - pi)) / GROUP // ∂ log π(a) / ∂u = a − π
        }
        // The sharpness penalty −‖π‖² is a known function of π: its gradient is exact.
        if (reward === 'proper') gu += -2 * (2 * pi - 1) * dpi
      }
      gw += gu * x
      gb += gu
    }
    w += (lr * gw) / BATCH
    b += (lr * gb) / BATCH
  }
  return trace
}

export function RewardCalibration() {
  const state = useFigureState({
    reward: choice(
      [
        { value: 'correct', label: 'correct answer: 1, else 0' },
        { value: 'coupled', label: 'correct − (π(a) − correct)²' },
        { value: 'proper', label: '2 × correct − ‖π‖²' },
        { value: 'supervised', label: 'supervised log loss' },
      ],
      'correct',
      { label: 'reward' },
    ),
    steps: int(600, { min: 10, max: 5000, label: 'training steps', suggestions: [200, 600, 2000] }),
    lr: float(1, { min: 0.05, max: 5, step: 0.05, label: 'learning rate', suggestions: [0.3, 1, 3] }),
    seed: int(1, { min: 1, max: 999, label: 'seed' }),
  })
  const reward = state.reward as Reward
  const trace = useMemo(
    () => train(reward, state.lr, state.steps, state.seed),
    [reward, state.lr, state.steps, state.seed],
  )

  const key = `${reward}|${state.lr}|${state.steps}|${state.seed}`
  const [pos, setPos] = useState({ key, step: 0 })
  const step = pos.key === key ? Math.min(pos.step, state.steps) : 0
  const w = trace[2 * step]
  const b = trace[2 * step + 1]

  const policy = useMemo(() => GRID.map((x) => sigmoid(w * x + b)), [w, b])
  const optimum = useMemo(() => TRUE_CURVE.map(OPTIMUM[reward]), [reward])
  let acc = 0
  let brier = 0
  let gap = 0
  for (let i = 0; i < GRID.length; i++) {
    const p = TRUE_CURVE[i]
    const q = policy[i]
    acc += q > 0.5 ? p : 1 - p
    brier += p * (1 - q) ** 2 + (1 - p) * q ** 2
    gap += Math.abs(q - p)
  }
  const m = GRID.length

  const xAxis = useAxis({ label: 'context x', range: [-3, 3] })
  const yAxis = useAxis({ label: 'probability of "yes"', range: [0, 1] })
  return (
    <Figure
      title="Which rewards keep a decision calibrated"
      state={state}
      caption="A yes-or-no decision model π(yes | x) = σ(wx + b), trained by policy gradient. In each context it samples four answers, a checker says which are correct, and each answer's reward is compared with the others' (a group baseline). The muted curve is the true probability, 'the reward's optimum' is where the chosen reward's expected value peaks, and 'model π' is the model at the current step. The correctness reward alone pushes towards certainty. Adding the squared error of the model's own probability for the sampled answer still overshoots. Doubling the correctness reward and subtracting the sum of squared probabilities is the Brier score in expectation, and stays calibrated, as supervised log loss does."
      controls={
        <Player
          value={step}
          onChange={(k) => setPos({ key, step: k })}
          count={state.steps + 1}
          label="step"
          format={(k) => `step ${k} of ${state.steps}`}
        />
      }
      readouts={
        <>
          <Readout label="slope w" value={formatNumber(w)} />
          <Readout label="accuracy" value={formatNumber(acc / m)} />
          <Readout label="expected Brier" value={formatNumber(brier / m)} />
          <Readout label="mean |π − p|" value={formatNumber(gap / m)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve name="true probability p" x={GRID} y={TRUE_CURVE} muted />
        <Curve name="the reward's optimum" x={GRID} y={optimum} slot={1} />
        <Curve name="model π" x={GRID} y={policy} slot={0} live />
      </Plot>
    </Figure>
  )
}
