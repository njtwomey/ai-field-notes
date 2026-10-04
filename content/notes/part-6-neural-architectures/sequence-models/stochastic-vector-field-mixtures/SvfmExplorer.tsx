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
import {
  endpointTask,
  svfmRun,
  type ComponentSelection,
  type SvfmCheckpoint,
  type SvfmRun,
} from 'aifn-methods/neural/ode-mixtures'
import { odeFailureCase } from 'aifn-methods/data/synthetic'
import { stream } from 'aifn/foundation/random'

const TASKS = [
  { value: 'splitting', label: '1D Trajectory Splitting (one-to-many)' },
  { value: 'crossing', label: '1D Trajectory Crossing (intersection)' },
  { value: 'scaling', label: 'Scale variance divergence' },
]

const MODELS = [
  { value: 'svfm', label: 'SVFM (Stochastic Vector Field Mixture)' },
  { value: 'node', label: 'Deterministic Neural ODE (single VF)' },
]

const SELECTIONS = [
  { value: 'pick-and-stick', label: 'Pick-and-stick (sample component at t₀)' },
  { value: 'forward-filter', label: 'Forward filter (continuous dynamic switching)' },
]

const REGULARISERS = [
  { value: 'none', label: 'Standard NLL only' },
  { value: 'transport', label: 'Transport regulariser L_T' },
  { value: 'variance', label: 'Variance regulariser L_V' },
]

export function SvfmExplorer() {
  const [task, setTask] = useState<'splitting' | 'crossing' | 'scaling'>('splitting')
  const [model, setModel] = useState<'svfm' | 'node'>('svfm')
  const [components, setComponents] = useState(2)
  const [selection, setSelection] = useState<ComponentSelection>('pick-and-stick')
  const [regulariser, setRegulariser] = useState<'none' | 'transport' | 'variance'>('none')
  const [steps, setSteps] = useState(120)
  const [timeIndex, setTimeIndex] = useState(0)

  const isMixture = model === 'svfm'

  // Run SVFM simulation
  const run: SvfmRun | null = useMemo(() => {
    const s = stream(`svfm-demo-${task}-${model}-${components}`)
    const dataset = odeFailureCase(s, { kind: task, n: 60 })
    const runData = endpointTask(dataset)

    const gen = svfmRun(runData, {
      components: isMixture ? components : 1,
      selection,
      losses: {
        transport: regulariser === 'transport',
        variance: regulariser === 'variance',
      },
      steps,
      learningRate: 0.01,
      hidden: 32,
      checkpoints: 10,
    })

    let last: SvfmRun | null = null
    for (const snap of gen) {
      last = snap
    }
    return last
  }, [task, model, components, selection, regulariser, steps, isMixture])

  const lastCheckpoint: SvfmCheckpoint | null = useMemo(() => {
    if (!run || run.checkpoints.length === 0) return null
    return run.checkpoints[run.checkpoints.length - 1]
  }, [run])

  const numFrames = run?.frameTimes.length ?? 1
  const currentFrame = Math.min(timeIndex, Math.max(0, numFrames - 1))
  const currentTime = run?.frameTimes[currentFrame] ?? 0

  // Trajectory paths over time
  const trajectories = useMemo(() => {
    if (!run || !lastCheckpoint) return []
    const P = run.shown.length
    const F = run.frameTimes.length
    const S = run.stateDim
    const paths = lastCheckpoint.paths

    const list: { key: number; x: number[]; y: number[] }[] = []
    for (let p = 0; p < P; p++) {
      const xs: number[] = []
      const ys: number[] = []
      for (let f = 0; f < F; f++) {
        xs.push(run.frameTimes[f])
        ys.push(paths[(f * P + p) * S])
      }
      list.push({ key: p, x: xs, y: ys })
    }
    return list
  }, [run, lastCheckpoint])

  // Points at the current frame
  const currentPoints = useMemo(() => {
    if (!run || !lastCheckpoint) return { x: [], y: [] }
    const P = run.shown.length
    const S = run.stateDim
    const paths = lastCheckpoint.paths

    const xs: number[] = []
    const ys: number[] = []
    for (let p = 0; p < P; p++) {
      xs.push(currentTime)
      ys.push(paths[(currentFrame * P + p) * S])
    }
    return { x: xs, y: ys }
  }, [run, lastCheckpoint, currentFrame, currentTime])

  // Target points at final time
  const targetPoints = useMemo(() => {
    if (!run) return { x: [], y: [] }
    const P = run.shown.length
    const D = run.dim
    const xs: number[] = []
    const ys: number[] = []
    for (let p = 0; p < P; p++) {
      xs.push(1.0)
      ys.push(run.targets[p * D])
    }
    return { x: xs, y: ys }
  }, [run])

  // Component probability curves over time for the mean trajectory
  const piCurves = useMemo(() => {
    if (!run || !lastCheckpoint || !isMixture) return []
    const K = run.components
    const F = run.frameTimes.length
    const P = run.shown.length
    const pi = lastCheckpoint.weights

    const curves: { name: string; x: number[]; y: number[]; colorSlot: number }[] = []
    for (let k = 0; k < K; k++) {
      const xs: number[] = []
      const ys: number[] = []
      for (let f = 0; f < F; f++) {
        let sum = 0
        for (let p = 0; p < P; p++) {
          sum += pi[(f * P + p) * K + k]
        }
        xs.push(run.frameTimes[f])
        ys.push(sum / P)
      }
      curves.push({
        name: `Component ${k + 1}`,
        x: xs,
        y: ys,
        colorSlot: k,
      })
    }
    return curves
  }, [run, lastCheckpoint, isMixture])

  const flowX = useAxis({ label: 'Time t', range: [0, 1] })
  const flowY = useAxis({ label: 'State x(t)', range: [-2.5, 2.5] })

  const piX = useAxis({ label: 'Time t', range: [0, 1] })
  const piY = useAxis({ label: isMixture ? 'Posterior weight πₖ(t)' : 'Training loss', range: [0, 1.2] })

  const lossCurve = useMemo(() => {
    if (!run) return { x: [0, 1], y: [0, 0] }
    const arr = Array.from(run.loss)
    const xs = arr.map((_, i) => (i + 1) / (arr.length || 1))
    return { x: xs, y: arr }
  }, [run])

  return (
    <Figure
      title="Stochastic Vector Field Mixtures (SVFM) Trajectory Modelling"
      purpose="Visualise multi-modal trajectory bifurcations, path crossings, and mixture component switching vs standard neural ODEs."
      caption={
        'Stochastic Vector Field Mixtures (SVFM; Twomey et al., 2020). Standard Neural ODEs are deterministic and cannot model trajectory splitting (bifurcations) or path crossing in 1D without violating Picard–Lindelöf uniqueness. SVFM parameterises a mixture of vector fields with learned switching probabilities π_k(t), allowing stochastic routing across multiple trajectories while preserving smoothness via transport and variance path regularisation.'
      }
    >
      <ControlRow label="Problem & model">
        <Select
          label="ODE failure scenario"
          value={task}
          options={TASKS}
          onChange={(v) => setTask(v as 'splitting' | 'crossing' | 'scaling')}
        />
        <Select label="Architecture" value={model} options={MODELS} onChange={(v) => setModel(v as any)} />
      </ControlRow>

      <ControlRow label="Mixture settings">
        <Select
          label="Components K"
          value={String(components)}
          options={[
            { value: '2', label: '2 components' },
            { value: '3', label: '3 components' },
            { value: '4', label: '4 components' },
          ]}
          disabled={!isMixture}
          onChange={(v) => setComponents(Number(v))}
        />
        <Select
          label="Component selection"
          value={selection}
          options={SELECTIONS}
          disabled={!isMixture}
          onChange={(v) => setSelection(v as any)}
        />
        <Select
          label="Path loss"
          value={regulariser}
          options={REGULARISERS}
          onChange={(v) => setRegulariser(v as any)}
        />
        <Select
          label="Training steps"
          value={String(steps)}
          options={[
            { value: '60', label: '60 steps' },
            { value: '120', label: '120 steps' },
            { value: '200', label: '200 steps' },
          ]}
          onChange={(v) => setSteps(Number(v))}
        />
      </ControlRow>

      <ControlRow label="Flow animation">
        <Player count={Math.max(1, numFrames)} value={currentFrame} onChange={setTimeIndex} />
      </ControlRow>

      <Plots>
        <Plot x={flowX} y={flowY} title="Continuous flow trajectories h(t)">
          {trajectories.map((traj) => (
            <Curve key={traj.key} x={traj.x} y={traj.y} slot={0} thin={true} />
          ))}
          <Points x={currentPoints.x} y={currentPoints.y} slot={1} size={5} />
          <Points x={targetPoints.x} y={targetPoints.y} slot={2} size={4} />
        </Plot>

        <Plot x={piX} y={piY} title={isMixture ? 'Component posterior weights πₖ(t)' : 'Model loss'}>
          {isMixture ? (
            piCurves.map((c) => <Curve key={c.name} x={c.x} y={c.y} slot={c.colorSlot} />)
          ) : (
            <Curve x={lossCurve.x} y={lossCurve.y} slot={0} />
          )}
        </Plot>
      </Plots>

      <ControlRow label="Diagnostics">
        <Readout
          label="Loss"
          value={run && run.loss.length > 0 ? formatNumber(Number(run.loss[run.loss.length - 1].toFixed(4))) : '—'}
        />
        <Readout label="Components" value={isMixture ? components : 1} />
        <Readout label="Time t" value={formatNumber(Number(currentTime.toFixed(2)))} />
        <Readout label="Solver steps" value={run ? run.gridTimes.length - 1 : '—'} />
      </ControlRow>
    </Figure>
  )
}
