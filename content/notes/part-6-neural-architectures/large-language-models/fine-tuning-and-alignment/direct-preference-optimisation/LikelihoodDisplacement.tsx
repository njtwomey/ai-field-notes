import { useContext, useMemo, useState, type ReactNode } from 'react'
import {
  Annotation,
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  FrameContext,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { postTrainingTrace } from 'aifn-methods/neural/post-training'

/** The unlabelled responses: fixed features φ_k in the plane. */
const OTHERS: readonly (readonly [number, number])[] = [
  [-1.5, 1.2],
  [0, 1.6],
  [-1.6, -0.6],
  [0.2, -1.4],
  [1.8, -1.0],
  [-0.6, 0.2],
]
const OTHER_NAMES = ['A', 'B', 'C', 'D', 'E', 'F']
const NAMES = ['W', 'L', ...OTHER_NAMES]
const K = NAMES.length
/** The reference policy is uniform: θ₀ = 0. */
const THETA0 = [0, 0]
const LOG_REF = Math.log(1 / K)
const SPAN = 2.5
const AT = NAMES.map((_, k) => k)
const LEFT = AT.map((k) => k - 0.2)
const RIGHT = AT.map((k) => k + 0.2)
const TOP = 280
const BOTTOM = 200

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

export function LikelihoodDisplacement() {
  const state = useFigureState({
    wx: slider(-SPAN, SPAN, 1.4, { step: 0.05, onChart: true }),
    wy: slider(-SPAN, SPAN, 0.8, { step: 0.05, onChart: true }),
    lx: slider(-SPAN, SPAN, 1.1, { step: 0.05, onChart: true }),
    ly: slider(-SPAN, SPAN, 1.1, { step: 0.05, onChart: true }),
    loss: choice(
      [
        { value: 'dpo', label: 'DPO' },
        { value: 'ipo', label: 'IPO (β is τ)' },
        { value: 'simpo', label: 'SimPO (γ = 0)' },
      ],
      'dpo',
      { label: 'loss' },
    ),
    beta: float(0.1, { gt: 0, le: 10, scale: 'log10', suggestions: [0.01, 0.1, 0.5, 2], label: 'β' }),
    nllWeight: float(0, { ge: 0, le: 5, suggestions: [0, 0.1, 0.5, 1], label: 'NLL weight on the chosen response' }),
    optimiser: choice(
      [
        { value: 'adam', label: 'Adam' },
        { value: 'sgd', label: 'SGD' },
      ],
      'adam',
      { label: 'optimiser' },
    ),
    learningRate: float(0.01, {
      gt: 0,
      le: 10,
      scale: 'log10',
      suggestions: [0.003, 0.01, 0.03, 0.1, 1],
      label: 'learning rate',
    }),
    steps: int(300, { ge: 10, le: 2000, suggestions: [100, 300, 1000], label: 'steps' }),
  })
  const { wx, wy, lx, ly, loss, beta, nllWeight, optimiser, learningRate, steps } = state

  const run = useComputed(() => {
    const features = [[wx, wy], [lx, ly], ...OTHERS.map((p) => [p[0], p[1]])]
    const trace = postTrainingTrace(
      { features, theta0: THETA0 },
      {
        method: 'dpo',
        pairs: [{ chosen: 0, rejected: 1 }],
        beta,
        loss,
        nllWeight,
        optimiser,
        learningRate,
        steps,
        seed: 1,
      },
    )
    // The policy's mean feature Σ_k π(k) φ_k at each step: where the probability mass sits in feature space.
    const mean = trace.map((s) =>
      features.reduce((acc, phi, k) => [acc[0] + s.probs[k] * phi[0], acc[1] + s.probs[k] * phi[1]], [0, 0]),
    )
    return {
      trace,
      mean,
      x: trace.map((s) => s.step),
      logpChosen: trace.map((s) => s.logpChosen ?? NaN),
      logpRejected: trace.map((s) => s.logpRejected ?? NaN),
      margin: trace.map((s) => s.margin ?? NaN),
      accuracy: trace.map((s) => s.accuracy ?? NaN),
    }
  }, [wx, wy, lx, ly, loss, beta, nllWeight, optimiser, learningRate, steps])
  const r = run.value

  const key = `${wx}|${wy}|${lx}|${ly}|${loss}|${beta}|${nllWeight}|${optimiser}|${learningRate}|${steps}`
  const [pos, setPos] = useState({ key, step: 0 })
  const step = pos.key === key ? Math.min(pos.step, r.trace.length - 1) : 0
  const now = r.trace[step]

  const path = useMemo(
    () => ({ x: r.mean.slice(0, step + 1).map((m) => m[0]), y: r.mean.slice(0, step + 1).map((m) => m[1]) }),
    [r.mean, step],
  )
  const top = useMemo(() => {
    let best = 0
    now.probs.forEach((p, k) => {
      if (p > now.probs[best]) best = k
    })
    return best
  }, [now])
  const reference = useMemo(() => AT.map(() => 1 / K), [])

  const fx = useAxis({ label: 'feature φ₁', range: [-SPAN, SPAN] })
  const fy = useAxis({ label: 'feature φ₂', range: [-SPAN, SPAN], equal: fx })
  const bx = useAxis({ label: 'response', categories: NAMES })
  const by = useAxis({ label: 'probability', range: [0, 1] })
  const sx = useAxis({ label: 'step', integer: true, range: [0, steps] })
  const ly1 = useAxis({ label: 'log π(y)', key })
  const ly2 = useAxis({ label: 'margin, accuracy', key })

  return (
    <Figure
      title="Likelihood displacement"
      state={state}
      defaultSize="L"
      caption="A toy policy over eight responses, π_θ(k) ∝ exp(θᵀφ_k), with fixed two-dimensional features φ_k and a uniform reference (θ₀ = 0), trained on one preference pair: chosen (W) over rejected (L). Top left: the features; drag the chosen and rejected points. The black path is the policy's mean feature Σ_k π_θ(k) φ_k up to the current step (the dot). Top right: π_ref (grey) and π_θ (green) at the current step. Bottom: TRL's metrics, logps/chosen and logps/rejected (left; the dashed line is the reference log-probability, log 1/8), rewards/margins and rewards/accuracies (right). Play the steps. For this policy the gradient of log π(chosen) − log π(rejected) with respect to θ is φ_chosen − φ_rejected, so every loss here moves θ along that difference. At the defaults the two features are close and their difference points towards E, the response furthest along it. Accuracy is 1 from the first step and the margin grows, while log π(chosen) falls from −2.08 to about −6.5 by step 300 and E, which neither label mentions, takes 0.97 of the probability. Drag rejected to the origin, so that the difference points at chosen itself, and log π(chosen) rises instead, to about −0.2; drag it onto A, where the difference still points at E, and E still takes the mass. With an NLL weight of 0.1 or more, log π(chosen) rises to about −0.8 by step 300, chosen becomes the most probable response and E ends below 0.2. The rejected response, being similar, rises too, and the margin grows about six times more slowly."
      controls={
        <Player
          value={step}
          onChange={(s) => setPos({ key, step: s })}
          count={r.trace.length}
          label="step"
          format={(s) => (s === 0 ? 'start: the reference policy' : `step ${s} of ${r.trace.length - 1}`)}
        />
      }
      readouts={
        <>
          <Readout label="loss" value={formatNumber(now.loss)} />
          <Readout label="logps/chosen" value={formatNumber(now.logpChosen ?? NaN)} />
          <Readout label="logps/rejected" value={formatNumber(now.logpRejected ?? NaN)} />
          <Readout label="rewards/margins" value={formatNumber(now.margin ?? NaN)} />
          <Readout label="rewards/accuracies" value={formatNumber(now.accuracy ?? NaN)} />
          <Readout label="KL(π_θ ‖ π_ref)" value={formatNumber(now.kl)} />
          <Readout label="most probable response" value={`${NAMES[top]} (${formatNumber(now.probs[top])})`} />
        </>
      }
    >
      <div className="flex flex-col gap-3">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <FixedHeight height={TOP}>
            <Plot x={fx} y={fy}>
              <Points
                name="unlabelled responses"
                x={OTHERS.map((p) => p[0])}
                y={OTHERS.map((p) => p[1])}
                labels={OTHER_NAMES}
                muted
                size={9}
              />
              <Curve name="mean feature" x={path.x} y={path.y} emphasis width={1.5} live />
              <Points name="mean feature now" x={[r.mean[step][0]]} y={[r.mean[step][1]]} emphasis size={7} live />
              <Points name="chosen" x={[wx]} y={[wy]} labels={['W, chosen']} slot={0} size={9} live />
              <Points name="rejected" x={[lx]} y={[ly]} labels={['L, rejected']} slot={1} size={9} live />
              <Handle {...state.handle(['wx', 'wy'], { label: 'chosen' })} slot={0} />
              <Handle {...state.handle(['lx', 'ly'], { label: 'rejected' })} slot={1} />
            </Plot>
          </FixedHeight>
          <FixedHeight height={TOP}>
            <Plot x={bx} y={by}>
              <Bars name="π_ref" x={LEFT} y={reference} width={0.38} muted />
              <Bars name="π_θ" x={RIGHT} y={now.probs} width={0.38} slot={2} live />
            </Plot>
          </FixedHeight>
        </div>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <FixedHeight height={BOTTOM}>
            <Plot x={sx} y={ly1}>
              <Curve name="logps/chosen" x={r.x} y={r.logpChosen} slot={0} stale={run.stale} />
              <Curve name="logps/rejected" x={r.x} y={r.logpRejected} slot={1} stale={run.stale} />
              <Annotation y={LOG_REF} dashed muted />
              <Annotation x={step} live />
            </Plot>
          </FixedHeight>
          <FixedHeight height={BOTTOM}>
            <Plot x={sx} y={ly2}>
              <Curve name="rewards/margins" x={r.x} y={r.margin} slot={3} stale={run.stale} />
              <Curve name="rewards/accuracies" x={r.x} y={r.accuracy} slot={4} stale={run.stale} />
              <Annotation x={step} live />
            </Plot>
          </FixedHeight>
        </div>
      </div>
    </Figure>
  )
}
