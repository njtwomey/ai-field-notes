import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  Handle,
  int,
  Player,
  Readout,
  setting,
  useFigureState,
  Vectors,
  type Vector,
} from 'aifn-render'
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

const EVERY = 10

/**
 * Virtual adversarial training on two moons with eight labels: the adversarial perturbation of every unlabelled point
 * at a checkpoint, and the boundary it trains.
 */
export function VirtualAdversarialDirections() {
  const state = useFigureState({
    epsilon: float(0.3, { ge: 0.01, le: 0.8, label: 'radius ε', suggestions: [0.1, 0.2, 0.3, 0.5], onChart: true }),
    iterations: int(1, { ge: 0, le: 5, label: 'power iterations (0: random direction)', suggestions: [0, 1, 2] }),
    entropy: setting(false, { label: 'add entropy minimisation' }),
    weight: float(1, { ge: 0, le: 30, label: 'VAT weight', suggestions: [0.3, 1, 3] }),
    steps: int(400, { ge: 10, le: 2000, label: 'training steps', suggestions: [100, 400, 800] }),
    rate: float(0.01, { gt: 0, le: 0.1, label: 'Adam step size', suggestions: [0.003, 0.01, 0.03] }),
    seed: int(2, { ge: 1, le: 999, label: 'data seed' }),
  })
  const { epsilon, iterations, entropy, weight, steps, rate, seed } = state
  const problem = useMemo(() => moonsProblem(seed), [seed])
  const setup = useMemo<RunSetup>(
    () => ({
      problem,
      method: { kind: 'vat', radius: epsilon, iterations, entropy },
      steps,
      rate,
      weight,
      weak: 0,
      strong: 0,
      seed,
      every: EVERY,
    }),
    [problem, epsilon, iterations, entropy, steps, rate, weight, seed],
  )
  const baselineSetup = useMemo<RunSetup>(
    () => ({ ...setup, method: { kind: 'supervised' }, every: steps }),
    [setup, steps],
  )
  const run = useSlicedRun(setup, consistencyRun)
  const baseline = useSlicedRun(baselineSetup, consistencyRun)
  const base = baseline.done ? baseline.checkpoints[baseline.checkpoints.length - 1] : undefined
  const [pos, setPos] = usePlayerPosition(setup)
  const count = run.checkpoints.length
  const at = Math.min(pos, Math.max(0, count - 1))
  const cp = run.checkpoints[at]
  const net = useMemo(() => classifier(2), [])
  const values = useMemo(() => (cp ? field(net, cp.state.params, MOONS_GRID, 2) : null), [cp, net])
  const arrows = useMemo<Vector[]>(() => {
    if (!cp || cp.step === 0) return []
    const r = cp.state.perturbation
    return problem.unlabelled.map(([x, y], i) => ({
      from: [x, y],
      to: [x + r[2 * i], y + r[2 * i + 1]],
      muted: true,
      head: 5,
      width: 1,
    }))
  }, [cp, problem])

  const centre = problem.labelled[0]
  const rim = circle(centre, epsilon)

  return (
    <Figure
      title="Virtual adversarial directions on two moons"
      state={state}
      defaultSize="L"
      caption={`Two moons with 8 labelled points (large marks) and 192 unlabelled ones; two hidden layers of 32 tanh units, trained full batch with Adam. At every step each unlabelled point x gets the perturbation r of length ε that most changes the network's predictive distribution, found by power iteration from a random direction with finite-difference length ξ = 10⁻⁶. The loss adds the mean KL divergence between the prediction at x (held fixed) and at x + r. The arrows show r at the player's checkpoint: they point across the boundary, where the prediction changes fastest. Drag the handle on the dashed circle around a labelled point to set ε. With 0 power iterations the direction stays random (random perturbation training); in two dimensions a random direction is rarely far from the adversarial one, so it trains almost as well here, and the difference grows with the input dimension.`}
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
            label="mean KL at x + r"
            value={cp && Number.isFinite(cp.state.unsupervisedLoss) ? cp.state.unsupervisedLoss.toExponential(2) : '…'}
          />
        </>
      }
    >
      <BoundaryMap problem={problem} grid={MOONS_GRID} box={MOONS_BOX} values={values} stale={!cp}>
        {arrows.length > 0 && <Vectors name="adversarial perturbation r" vectors={arrows} />}
        <Curve name="radius ε" x={rim.x} y={rim.y} emphasis dashed silent />
        <Handle
          kind="point"
          at={[centre[0] + epsilon, centre[1]]}
          label="ε"
          onDrag={([x, y]) => {
            const r = Math.hypot(x - centre[0], y - centre[1])
            state.set('epsilon', Math.round(Math.min(0.8, Math.max(0.01, r)) * 100) / 100)
          }}
        />
      </BoundaryMap>
    </Figure>
  )
}
