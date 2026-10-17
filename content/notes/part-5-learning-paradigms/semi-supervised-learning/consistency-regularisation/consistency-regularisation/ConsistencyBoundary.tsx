import { useMemo } from 'react'
import { choice, Curve, Figure, float, Handle, int, Player, Readout, useFigureState } from 'aifn-render'
import { BoundaryMap } from '../_shared/BoundaryMap'
import {
  circle,
  classifier,
  consistencyRun,
  field,
  moonsProblem,
  MOONS_BOX,
  MOONS_GRID,
  percent,
  type RunSetup,
} from '../_shared/consistency-runs'
import { usePlayerPosition, useSlicedRun } from '../_shared/figure-helpers'
import type { ConsistencyMethod } from 'aifn-methods/neural/semi-supervised'

const METHODS = [
  { value: 'pi-model', label: 'Π-model' },
  { value: 'temporal-ensembling', label: 'temporal ensembling' },
  { value: 'supervised', label: 'labelled points only' },
] as const
type Method = (typeof METHODS)[number]['value']

const EVERY = 10

/**
 * Two moons with four labels per class: a network trained with the Π-model or temporal ensembling against the same
 * network trained on the labelled points alone, with the perturbation radius set by dragging a circle.
 */
export function ConsistencyBoundary() {
  const state = useFigureState({
    method: choice(METHODS, 'pi-model', { label: 'method' }),
    sigma: float(0.1, {
      ge: 0,
      le: 0.5,
      label: 'perturbation sd σ',
      suggestions: [0, 0.05, 0.1, 0.2, 0.4],
      onChart: true,
    }),
    weight: float(10, { ge: 0, le: 100, label: 'consistency weight λ', suggestions: [1, 10, 30] }),
    steps: int(600, { ge: 10, le: 2000, label: 'training steps', suggestions: [200, 600, 1200] }),
    rate: float(0.01, { gt: 0, le: 0.1, label: 'Adam step size', suggestions: [0.003, 0.01, 0.03] }),
    seed: int(2, { ge: 1, le: 999, label: 'data seed' }),
  })
  const { method, sigma, weight, steps, rate, seed } = state
  const problem = useMemo(() => moonsProblem(seed), [seed])
  const setup = useMemo<RunSetup>(
    () => ({
      problem,
      method: { kind: method as Method } as ConsistencyMethod,
      steps,
      rate,
      weight,
      rampUp: Math.round(steps / 3),
      weak: sigma,
      strong: sigma,
      seed,
      every: EVERY,
    }),
    [problem, method, steps, rate, weight, sigma, seed],
  )
  const baselineSetup = useMemo<RunSetup>(
    () => ({ ...setup, method: { kind: 'supervised' }, weak: 0, strong: 0, every: steps }),
    [setup, steps],
  )
  const run = useSlicedRun(setup, consistencyRun)
  const baseline = useSlicedRun(baselineSetup, consistencyRun)
  const [pos, setPos] = usePlayerPosition(setup)
  const count = run.checkpoints.length
  const at = Math.min(pos, Math.max(0, count - 1))
  const cp = run.checkpoints[at]
  const net = useMemo(() => classifier(2), [])
  const values = useMemo(() => (cp ? field(net, cp.state.params, MOONS_GRID, 2) : null), [cp, net])
  const base = baseline.done ? baseline.checkpoints[baseline.checkpoints.length - 1] : undefined

  // The perturbation disc is drawn around the first labelled point; its handle sits on the rim.
  const centre = problem.labelled[0]
  const rim = circle(centre, sigma)

  return (
    <Figure
      title="Consistency regularisation on two moons"
      state={state}
      defaultSize="L"
      caption={`Two moons, 200 points with Gaussian noise of sd 0.1, of which 8 are labelled (large marks, 4 per class) and 192 are not (small grey marks). The network has two hidden layers of 32 tanh units and is trained full batch with Adam. The background is the network's probability of the inner moon minus ½; the black curve is its decision boundary. The Π-model adds λ times the mean squared difference between the softmax outputs on two copies of each unlabelled point, each jittered by Gaussian noise of sd σ; temporal ensembling compares one jittered copy with the running average of the point's past outputs. λ ramps up over the first third of training. Drag the handle on the dashed circle to set σ (the circle has radius σ). With σ = 0 the Π-model's two passes coincide, its consistency term is zero and the run matches the labelled-only network; at σ = 0.4 the jittered copies of one moon reach the other and the gain shrinks or reverses.`}
      controls={
        <Player
          value={at}
          onChange={setPos}
          count={Math.max(1, count)}
          label="checkpoint"
          format={(a) => `step ${run.checkpoints[a]?.step ?? 0}`}
        />
      }
      readouts={
        <>
          <Readout label="step" value={cp ? `${cp.step}${run.done ? '' : ', training…'}` : '…'} />
          <Readout label="accuracy on unlabelled points" value={percent(cp?.accuracy)} />
          <Readout label="labelled points only, at the end" value={percent(base?.accuracy)} />
          <Readout
            label="consistency loss"
            value={cp && Number.isFinite(cp.state.unsupervisedLoss) ? cp.state.unsupervisedLoss.toExponential(2) : '…'}
          />
        </>
      }
    >
      <BoundaryMap problem={problem} grid={MOONS_GRID} box={MOONS_BOX} values={values} stale={!cp}>
        <Curve name="perturbation radius σ" x={rim.x} y={rim.y} emphasis dashed silent />
        <Handle
          kind="point"
          at={[centre[0] + sigma, centre[1]]}
          label="σ"
          onDrag={([x, y]) => {
            const r = Math.hypot(x - centre[0], y - centre[1])
            state.set('sigma', Math.round(Math.min(0.5, r) * 100) / 100)
          }}
        />
      </BoundaryMap>
    </Figure>
  )
}
