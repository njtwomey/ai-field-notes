import { useMemo, useState } from 'react'
import {
  ControlRow,
  Curve,
  Figure,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  Select,
  formatNumber,
  useAxis,
} from 'aifn-render'
import { stream } from 'aifn/foundation/random'
import {
  discInRing,
  odeRun,
  reflectionData,
  type OdeCheckpoint,
  type OdeModelKind,
  type OdeRun,
} from 'aifn-methods/neural/ode'

const DATASETS = [
  { value: 'disc', label: 'Disc in ring (non-homeomorphic topology in 2D)' },
  { value: 'reflection', label: '1D Reflection g(x) = -x (crosses paths)' },
]

const MODELS: { value: OdeModelKind; label: string }[] = [
  { value: 'node', label: 'Neural ODE (standard homeomorphic flow)' },
  { value: 'anode', label: 'Augmented Neural ODE (ANODE: +1 dimension)' },
  { value: 'resnet', label: 'Discrete ResNet (Euler discretisation)' },
]

const SOLVERS = [
  { value: 'rk4', label: 'Runge-Kutta 4th-order (RK4)' },
  { value: 'euler', label: 'Euler method (1st-order)' },
]

export function NeuralOdeExplorer() {
  const [dataChoice, setDataChoice] = useState<'disc' | 'reflection'>('disc')
  const [modelKind, setModelKind] = useState<OdeModelKind>('node')
  const [solver, setSolver] = useState<'rk4' | 'euler'>('rk4')
  const [steps, setSteps] = useState(80)
  const [timeIndex, setTimeIndex] = useState(0)

  // Run the training simulation
  const run: OdeRun | null = useMemo(() => {
    const s = stream('neural-ode-explorer')
    const dataset = dataChoice === 'disc' ? discInRing(s, 240) : reflectionData(40)

    const gen = odeRun(dataset, {
      kind: modelKind,
      steps,
      hidden: 32,
      augment: modelKind === 'anode' ? 1 : 0,
      solver: { method: solver },
      frames: 21,
      seed: 42,
    })

    let last: OdeRun | null = null
    for (const snapshot of gen) {
      last = snapshot
    }
    return last
  }, [dataChoice, modelKind, solver, steps])

  const lastCheckpoint: OdeCheckpoint | null = useMemo(() => {
    if (!run || run.checkpoints.length === 0) return null
    return run.checkpoints[run.checkpoints.length - 1]
  }, [run])

  const numTimes = run?.times.length ?? 1
  const currentFrame = Math.min(timeIndex, Math.max(0, numTimes - 1))
  const currentTime = run?.times[currentFrame] ?? 0

  const isOneD = dataChoice === 'reflection'

  // Points at current time frame
  const pointCoordinates = useMemo(() => {
    if (!run || !lastCheckpoint) {
      return { class0: { x: [], y: [] }, class1: { x: [], y: [] } }
    }

    const P = run.shown.length
    const S = run.stateDim
    const isOneD = run.dim === 1
    const paths = lastCheckpoint.paths

    const c0X: number[] = []
    const c0Y: number[] = []
    const c1X: number[] = []
    const c1Y: number[] = []

    for (let p = 0; p < P; p++) {
      const idx = run.shown[p]
      const label = run.labels[idx] ?? 0
      const offset = (currentFrame * P + p) * S

      if (isOneD) {
        const xVal = currentTime
        const yVal = paths[offset]
        if (label === 0) {
          c0X.push(xVal)
          c0Y.push(yVal)
        } else {
          c1X.push(xVal)
          c1Y.push(yVal)
        }
      } else {
        const xVal = paths[offset]
        const yVal = paths[offset + 1]
        if (label === 0) {
          c0X.push(xVal)
          c0Y.push(yVal)
        } else {
          c1X.push(xVal)
          c1Y.push(yVal)
        }
      }
    }

    return {
      class0: { x: c0X, y: c0Y },
      class1: { x: c1X, y: c1Y },
    }
  }, [run, lastCheckpoint, currentFrame, currentTime])

  const plotX = useAxis({
    label: isOneD ? 'Time t' : 'State dimension x₁',
    range: isOneD ? [0, 1] : [-2.5, 2.5],
  })

  const plotY = useAxis({
    label: isOneD ? 'Feature value x(t)' : 'State dimension x₂',
    range: isOneD ? [-2, 2] : [-2.5, 2.5],
  })

  const lossX = useAxis({
    label: 'Iteration',
    range: [0, steps],
  })

  const lossY = useAxis({
    label: 'Loss',
    range: [0, 1.2],
  })

  const lossSteps = useMemo(() => {
    return Array.from({ length: steps }, (_, i) => i)
  }, [steps])

  const lossValues = useMemo(() => {
    if (!run) return []
    return Array.from(run.loss.subarray(0, steps))
  }, [run, steps])

  return (
    <Figure
      title="Neural ODE and Augmented NODE Dynamics"
      purpose="Explore homeomorphic flow trajectories, topology preserving limitations, and dimensional lifting across Euler and Runge-Kutta ODE solvers."
      caption={
        'Interactive Neural ODE dynamics (Dupont et al., 2019; Chen et al., 2018). In 2D, the continuous flow of a standard Neural ODE cannot separate a disc from a surrounding ring without tearing the plane (a homeomorphism cannot change topology). Augmented Neural ODEs (ANODE) lift the dynamics into an extra dimension, allowing the inner disc to rotate out and achieve zero error.'
      }
    >
      <ControlRow label="Setup">
        <Select
          label="Problem"
          value={dataChoice}
          options={DATASETS}
          onChange={(v) => setDataChoice(v as 'disc' | 'reflection')}
        />
        <Select label="Model" value={modelKind} options={MODELS} onChange={(v) => setModelKind(v as OdeModelKind)} />
        <Select label="Solver" value={solver} options={SOLVERS} onChange={(v) => setSolver(v as 'rk4' | 'euler')} />
        <Select
          label="Iterations"
          value={String(steps)}
          options={[
            { value: '40', label: '40 iterations' },
            { value: '80', label: '80 iterations' },
            { value: '140', label: '140 iterations' },
          ]}
          onChange={(v) => setSteps(Number(v))}
        />
      </ControlRow>

      <ControlRow label="Continuous time evolution">
        <Player count={Math.max(1, numTimes)} value={currentFrame} onChange={setTimeIndex} />
      </ControlRow>

      <Plots>
        <Plot x={plotX} y={plotY} title={isOneD ? '1D flow over time t' : 'State trajectories in feature space'}>
          <Points x={pointCoordinates.class0.x} y={pointCoordinates.class0.y} slot={0} size={5} />
          <Points x={pointCoordinates.class1.x} y={pointCoordinates.class1.y} slot={1} size={5} />
        </Plot>

        <Plot x={lossX} y={lossY} title="Training loss convergence">
          <Curve x={lossSteps} y={lossValues} slot={0} />
        </Plot>
      </Plots>

      <ControlRow label="Metrics">
        <Readout
          label="Final loss"
          value={run && run.loss.length > 0 ? formatNumber(Number(run.loss[run.loss.length - 1].toFixed(4))) : '—'}
        />
        <Readout label="Time t" value={formatNumber(Number(currentTime.toFixed(2)))} />
        <Readout label="Evaluations (NFE)" value={lastCheckpoint ? lastCheckpoint.evaluations : '—'} />
        <Readout label="State dimensions" value={run ? run.stateDim : 2} />
      </ControlRow>
    </Figure>
  )
}
