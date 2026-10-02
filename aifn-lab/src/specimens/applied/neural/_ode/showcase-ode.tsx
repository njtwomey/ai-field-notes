import { useMemo, useState, type ReactNode } from 'react'
import type { CnfRun, LatentOdeRun, OdeRun } from 'aifn-applied/neural/ode'
import { Player, Select, Slider } from '@lab/controls'
import { Dashboard, DashboardCell, DashboardRow, Figure } from '@lab/layout'
import { choice, float, int, row, setting, toggle, useFigureState, when } from '@lab/state'
import { formatValue, TrainControls, useTrainedRun } from '@lab/views'
import { Annotation, Curve, Handle, Plot, Points, Raster, Readout, Segments, useAxis, Vectors } from '@lab/viz'
import { arrows, at, rows, thin, trails, typicalLength } from './geometry'
import {
  CLASS_DATA,
  DENSITY_DATA,
  GRADIENTS,
  LATENT_DATA,
  MODELS,
  PRESETS,
  SOLVERS,
  TASKS,
  taskOf,
  type ModelKind,
  type Result,
  type Settings,
  type TaskKind,
} from './settings'

const f3 = (v: number | undefined) =>
  v !== undefined && Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—'
const supervised = (v: { task?: unknown }) => v.task === 'classification' || v.task === 'regression'

/** The checkpoint shown belongs to the run it was picked on; a new run opens at step 0 and t = 0. */
function usePick(run: unknown, fallback = 0) {
  const [picked, setPicked] = useState<{ run: unknown; index: number } | null>(null)
  const index = picked && picked.run === run ? picked.index : fallback
  return [index, (i: number) => setPicked({ run, index: i })] as const
}

export function NeuralOdeShowcase() {
  const state = useFigureState({
    setup: row('1 · task and model', {
      task: choice(TASKS, 'classification', { label: 'task' }),
      classData: CLASS_DATA.field({ choiceLabel: 'data', initial: 'disc', when: when('task', 'classification') }),
      densityData: DENSITY_DATA.field({ choiceLabel: 'data', initial: 'moons', when: when('task', 'density') }),
      latentData: choice(LATENT_DATA, 'sine', { label: 'data', when: when('task', 'latent') }),
      model: choice(MODELS, 'node', { label: 'model', when: supervised }),
      augment: choice([1, 2], 1, { label: 'extra dimensions', when: (v) => supervised(v) && v.model === 'anode' }),
      depth: int(10, {
        ge: 1,
        le: 100,
        suggestions: [4, 10, 20],
        label: 'blocks (= Euler steps)',
        when: (v) => supervised(v) && v.model === 'resnet',
      }),
      timeDependent: setting(false, { label: 'time-dependent f(t, x)', when: supervised }),
      estimator: choice(
        [
          { value: 'exact', label: 'exact trace (2 jvps)' },
          { value: 'hutchinson', label: 'Hutchinson (1 vjp)' },
        ],
        'exact',
        { label: 'divergence', when: when('task', 'density') },
      ),
      probe: choice(['rademacher', 'gaussian'], 'rademacher', {
        label: 'probe ε',
        when: (v) => v.task === 'density' && v.estimator === 'hutchinson',
      }),
    }),
    solver: row('2 · solver and gradient', {
      gradient: choice(GRADIENTS, 'backprop', { label: 'gradient' }),
      method: choice(SOLVERS, 'rk4', { label: 'solver' }),
      stepSize: float(0.1, {
        gt: 0,
        scale: 'log10',
        suggestions: [0.05, 0.1, 0.25, 0.5],
        label: 'step size',
        when: (v) => v.method !== 'dormand-prince',
      }),
      rtol: float(1e-3, {
        gt: 0,
        le: 0.1,
        scale: 'log10',
        suggestions: [1e-2, 1e-3, 1e-4, 1e-5],
        label: 'rtol (atol = rtol/100)',
        when: when('method', 'dormand-prince'),
      }),
      checkpoints: int(1, {
        ge: 1,
        le: 50,
        suggestions: [1, 4, 10],
        label: 'adjoint checkpoints',
        when: when('gradient', 'adjoint'),
      }),
    }),
    train: row('3 · training', {
      steps: int(300, { ge: 1, suggestions: [100, 200, 300, 500], label: 'iterations' }),
      learningRate: float(0.01, {
        label: 'Adam rate',
        gt: 0,
        le: 0.1,
        scale: 'log10',
        suggestions: [0.003, 0.01, 0.02],
      }),
      kinetic: float(0, { ge: 0, step: 0.001, suggestions: [0, 0.001, 0.01, 0.05], label: 'kinetic λ' }),
      jacobian: float(0, { ge: 0, step: 0.001, suggestions: [0, 0.001, 0.01, 0.05], label: 'Jacobian λ' }),
      hidden: int(32, { ge: 2, le: 128, suggestions: [16, 32, 64], label: 'field width' }),
      seed: int(0, { label: 'seed', ge: 0, le: 9999 }),
    }),
    show: row('4 · show', { boundary: toggle(true, 'decision boundary (P = 0.5)') }),
  })
  const { setup, solver, train } = state
  const settings: Settings = {
    task: setup.task as TaskKind,
    classData: { key: setup.classData.key, values: { ...setup.classData.values } },
    densityData: { key: setup.densityData.key, values: { ...setup.densityData.values } },
    latentData: setup.latentData as 'sine' | 'spiral',
    model: setup.model as ModelKind,
    augment: Number(setup.augment),
    depth: Number(setup.depth),
    timeDependent: Boolean(setup.timeDependent),
    estimator: setup.estimator as 'exact' | 'hutchinson',
    probe: setup.probe as 'rademacher' | 'gaussian',
    gradient: solver.gradient as 'backprop' | 'adjoint',
    method: solver.method as Settings['method'],
    stepSize: Number(solver.stepSize),
    rtol: Number(solver.rtol),
    checkpoints: Number(solver.checkpoints),
    steps: Number(train.steps),
    learningRate: train.learningRate,
    kinetic: Number(train.kinetic),
    jacobian: Number(train.jacobian),
    hidden: Number(train.hidden),
    seed: train.seed,
  }
  const trained = useTrainedRun(settings, taskOf)
  const raw = trained.run.value
  const result: Result | null = raw && trained.trained ? ({ task: trained.trained.task, run: raw } as Result) : null

  const [preset, setPreset] = useState<string>('')
  const presets = (
    <div className="pb-2">
      <Select
        label="preset (sets the controls; then press Train)"
        value={preset}
        onChange={(key) => {
          setPreset(key)
          const p = PRESETS[key]
          if (!p) return
          state.reset()
          for (const [path, v] of Object.entries(p.values)) state.set(path, v)
        }}
        options={[
          { value: '', label: 'choose…' },
          ...Object.entries(PRESETS).map(([value, p]) => ({ value, label: p.label })),
        ]}
      />
    </div>
  )
  const done = raw?.done ?? 0
  const total = trained.trained?.steps ?? settings.steps
  const controls = (
    <>
      {presets}
      <TrainControls run={trained as never} progress={done / total} progressText={`${done} / ${total} iterations`} />
    </>
  )
  const common = {
    runKey: trained.trained,
    error: raw?.error ?? null,
    pending: !trained.trained ? 'press Train to start' : !raw ? 'training…' : undefined,
  }
  const task = trained.trained?.task ?? settings.task
  // Every view's hooks run on every render (each with no run unless it is the task's), so the figures never remount.
  const flow = useFlowView({
    boundary: state.show.boundary,
    ...common,
    run: result && (result.task === 'classification' || result.task === 'regression') ? result.run : null,
  })
  const density = useCnfView({ ...common, run: result?.task === 'density' ? result.run : null })
  const latent = useLatentView({ ...common, run: result?.task === 'latent' ? result.run : null })
  const view = task === 'density' ? density : task === 'latent' ? latent : flow
  return (
    <>
      <Figure
        title={TITLE}
        id="the-flow"
        purpose={PURPOSE}
        state={state}
        defaultSize="XL"
        controls={controls}
        readouts={view.readouts}
        caption={view.caption}
      >
        {view.body}
      </Figure>
      <TrainingFigure training={view.training} />
    </>
  )
}

type ViewProps<R> = {
  run: R | null
  runKey: unknown
  error: string | null
  pending?: string
}

const TITLE = 'The flow'

/** What a task contributes to the two figures (the figures themselves stay mounted across tasks, keeping their ids). */
type View = { readouts: ReactNode; caption: ReactNode; body: ReactNode; training: ReactNode }

/** The training curves; dragging or clicking the iteration marker picks the model the flow figure shows. */
function TrainingFigure({ training }: { training: ReactNode }) {
  return (
    <Figure
      title="Training"
      id="training"
      purpose="The loss, the work of each iteration and the gradient checks over the run; the iteration marker picks the model shown above."
      defaultSize="XL"
      caption={
        <>
          Left: the minibatch loss. Middle: function evaluations per iteration, forward and backward (the adjoint&apos;s
          backward solve, or backprop&apos;s replay of the recorded steps, which equals the forward count). Right: the
          diagnostics recorded at each checkpoint. Drag the iteration marker, or click a chart, to show that
          checkpoint&apos;s model in the flow figure; it starts at the last checkpoint.
        </>
      }
    >
      <Dashboard>{training}</Dashboard>
    </Figure>
  )
}
const PURPOSE =
  'A network that is the flow of a learned vector field: trajectories cannot cross, extra dimensions let them pass, the solver and the gradient method trade work, memory and accuracy, and the same flow transports a density.'

// ── Shared training panels ────────────────────────────────────────────────────────────────────────────────────────

function useTrainingRow({
  loss,
  lossLabel,
  nfeForward,
  nfeBackward,
  steps,
  marker,
  extra,
}: {
  loss: Float64Array | null
  lossLabel: string
  nfeForward: Float64Array | null
  nfeBackward: Float64Array | null
  steps: number
  marker: ReactNode
  extra?: ReactNode
}) {
  const it = useAxis({ label: 'iteration', range: [0, steps], key: steps, integer: true })
  const lossAxis = useAxis({ label: lossLabel, hold: 'union', key: steps })
  const nfeAxis = useAxis({ label: 'function evaluations', range: [0, undefined], hold: 'union', key: steps })
  const l = useMemo(() => (loss ? thin(loss) : null), [loss])
  const f = useMemo(() => (nfeForward ? thin(nfeForward) : null), [nfeForward])
  const b = useMemo(() => (nfeBackward ? thin(nfeBackward) : null), [nfeBackward])
  return (
    <DashboardRow key="training">
      <DashboardCell>
        <Plot x={it} y={lossAxis} title="training loss (minibatch)">
          {l && <Curve name={lossLabel} x={l.x} y={l.y} slot={2} />}
          {marker}
        </Plot>
      </DashboardCell>
      <DashboardCell>
        <Plot x={it} y={nfeAxis} title="work per iteration">
          {f && <Curve name="forward NFE" x={f.x} y={f.y} slot={3} />}
          {b && <Curve name="backward NFE" x={b.x} y={b.y} slot={4} dashed />}
          {marker}
        </Plot>
      </DashboardCell>
      {extra}
    </DashboardRow>
  )
}

const meanOf = (a: Float64Array | undefined, from = 1) => {
  if (!a || a.length <= from) return NaN
  let s = 0
  for (let i = from; i < a.length; i++) s += a[i]
  return s / (a.length - from)
}

function useCheckpoint<C extends { step: number }>(run: { checkpoints: C[] } | null, runKey: unknown) {
  const shots = run?.checkpoints ?? []
  // The model shown defaults to the latest checkpoint (the trained model once the run ends).
  const [picked, pick] = usePick(runKey, Infinity)
  const index = Math.min(picked, Math.max(0, shots.length - 1))
  const shot: C | undefined = shots[index]
  const pickStep = (step: number) => {
    let best = 0
    shots.forEach((c, i) => {
      if (Math.abs(c.step - step) < Math.abs(shots[best].step - step)) best = i
    })
    pick(best)
  }
  const marker = shot ? <Handle kind="x" at={shot.step} onDrag={pickStep} label={`iteration ${shot.step}`} /> : null
  // Which model the flow shows: a steppable slider over the checkpoints (no playback: time t is the animation).
  const player = (
    <Slider
      label="model shown (training iteration)"
      value={index}
      onChange={(v) => pick(Math.round(v))}
      min={0}
      max={Math.max(1, shots.length - 1)}
      step={1}
      steppable
      disabled={shots.length < 2}
      format={(i) => `iteration ${shots[Math.round(i)]?.step ?? 0}`}
    />
  )
  return { shot, index, marker, player }
}

function TimePlayer({
  frame,
  setFrame,
  times,
}: {
  frame: number
  setFrame: (f: number) => void
  times: Float64Array | null
}) {
  return (
    <Player
      label="time t"
      value={frame}
      onChange={setFrame}
      count={Math.max(1, times?.length ?? 1)}
      format={(i) => `t = ${f3(times?.[i] ?? 0)}`}
    />
  )
}

// ── Classification and regression ─────────────────────────────────────────────────────────────────────────────────

function useFlowView({ run, runKey, error, pending, boundary }: ViewProps<OdeRun> & { boundary: boolean }): View {
  const { shot, marker, player } = useCheckpoint(run, runKey)
  const [frame, setFrame] = usePick(runKey)
  const T = run?.times.length ?? 1
  const f = Math.min(frame, T - 1)
  const P = run?.shown.length ?? 0
  const D = run?.stateDim ?? 2
  const oneD = run?.dim === 1
  const groups = run?.task === 'classification' ? run.classes : 2
  // Each shown point's group: its class, or the sign of x(0) for the reflection.
  const group = useMemo(() => {
    if (!run) return new Int32Array(0)
    return Int32Array.from(run.shown, (i) =>
      run.task === 'classification' ? run.labels[i] : run.data[i * run.dim] < 0 ? 0 : 1,
    )
  }, [run])
  const groupNames =
    run?.task === 'classification'
      ? Array.from({ length: run.classes }, (_, k) => `class ${k}`)
      : ['x(0) < 0', 'x(0) > 0']

  // Axes held for the run: the plane of the first two state coordinates, or (t, x) for a 1-d state.
  const box = run?.box ?? 2
  // Held for the run: the trajectories' extent over every time and checkpoint (98% of the coordinates, so a few
  // escaping points do not shrink the rest), between the data's box and twice it (the field's grid).
  const extent = useMemo(() => {
    if (!run) return box
    const all: number[] = []
    for (const c of run.checkpoints)
      for (let i = 0; i < c.paths.length; i += 3) if (Number.isFinite(c.paths[i])) all.push(Math.abs(c.paths[i]))
    all.sort((p, q) => p - q)
    const m = all.length ? all[Math.floor(0.98 * (all.length - 1))] : box
    return Math.ceil(Math.min(Math.max(m * 1.1, box), 2 * box) * 2) / 2
  }, [run, box])
  const key = runKey
  const px = useAxis({
    label: oneD && D === 1 ? 't' : 'z₁',
    range: oneD && D === 1 ? [0, 1] : [-extent, extent],
    key: [key, extent],
  })
  const py = useAxis({
    label: oneD && D === 1 ? 'x(t)' : oneD ? 'z₂ (augmented)' : 'z₂',
    range: [-extent, extent],
    key: [key, extent],
    equal: oneD && D === 1 ? undefined : px,
  })
  const qx = useAxis({ label: 'x₁', range: [-box, box], key })
  const qy = useAxis({ label: oneD ? 'g(x)' : 'x₂', range: [-box, box], key, equal: oneD ? undefined : qx })
  const ax = useAxis({ label: 'z₁', range: [-extent, extent], key: [key, extent] })
  const aa = useAxis({ label: 'a (augmented)', range: [-extent, extent], key: [key, extent], equal: ax })

  const scale = useMemo(() => {
    // One scale per checkpoint (over its frames), so an early, weak field still shows its directions.
    if (!run || !shot?.field) return 1
    const cell = Math.abs(run.fieldAxis[1] - run.fieldAxis[0])
    return (0.7 * cell) / typicalLength(shot.field)
  }, [run, shot])
  const vectors = useMemo(() => {
    if (!run || !shot?.field) return []
    const field = shot.field[shot.field.length === 1 ? 0 : f]
    if (D === 1) {
      // Direction field (1, f(t, x)) on the (t, x) grid, scaled per axis.
      const tAxis = run.fieldTimes
      const xAxis = run.fieldAxis
      const dt = tAxis[1] - tAxis[0]
      const dx = xAxis[1] - xAxis[0]
      let m = 0
      for (let i = 1; i < field.length; i += 2) m = Math.max(m, Math.abs(field[i]))
      const out = []
      for (let i = 0; i < xAxis.length; i++)
        for (let j = 0; j < tAxis.length; j++) {
          const v = field[2 * (i * tAxis.length + j) + 1]
          const len = Math.hypot(1, v / Math.max(1e-9, m / (dx / dt)))
          const ux = (0.8 * dt) / len
          const uy = ((0.8 * dt) / len) * (v / Math.max(1e-9, m / (dx / dt))) * (dx / dt)
          out.push({
            from: [tAxis[j], xAxis[i]] as [number, number],
            to: [tAxis[j] + ux, xAxis[i] + uy] as [number, number],
            head: 6,
            width: 1,
            muted: true,
          })
        }
      return out
    }
    return arrows(field, run.fieldAxis, run.fieldAxis, scale)
  }, [run, shot, f, D, scale])

  // Trails and points: for a 1-d state, x(t) against t.
  const drawn = useMemo(() => {
    if (!run || !shot) return null
    if (D === 1) {
      const segs: { from: [number, number]; to: [number, number] }[][] = Array.from({ length: groups }, () => [])
      for (let k = 0; k < f; k++)
        for (let i = 0; i < P; i++)
          segs[group[i]].push({
            from: [run.times[k], shot.paths[k * P + i]],
            to: [run.times[k + 1], shot.paths[(k + 1) * P + i]],
          })
      const now = {
        x: Array.from({ length: P }, () => run.times[f]),
        y: Array.from({ length: P }, (_, i) => shot.paths[f * P + i]),
      }
      return { segs, now }
    }
    return { segs: trails(shot.paths, P, D, f, 0, 1, group, groups), now: at(shot.paths, P, D, f, 0, 1) }
  }, [run, shot, f, D, P, group, groups])
  const augmented = useMemo(() => {
    if (!run || !shot || run.kind !== 'anode' || D < 3) return null
    return { segs: trails(shot.paths, P, D, f, 0, D - 1, group, groups), now: at(shot.paths, P, D, f, 0, D - 1) }
  }, [run, shot, f, D, P, group, groups])
  const data = useMemo(() => {
    if (!run) return null
    const n = run.data.length / run.dim
    return run.dim === 2
      ? {
          x: Array.from({ length: n }, (_, i) => run.data[2 * i]),
          y: Array.from({ length: n }, (_, i) => run.data[2 * i + 1]),
          g: Array.from(run.labels),
        }
      : { x: Array.from(run.data), y: Array.from(run.targets), g: Array.from(run.data, (v) => (v < 0 ? 0 : 1)) }
  }, [run])
  const decision = useMemo(() => {
    if (!run || !shot?.decision) return null
    return run.task === 'classification' && run.dim === 2 ? rows(shot.decision, run.decisionAxis.length) : null
  }, [run, shot])
  const gradients = useMemo(() => {
    if (!run) return null
    const withGrad = run.checkpoints.filter((c) => c.gradient)
    return {
      x: withGrad.map((c) => c.step),
      rel: withGrad.map((c) => Math.max(1e-12, c.gradient!.relativeError)),
      rec: withGrad.map((c) => Math.max(1e-12, c.gradient!.reconstructionError)),
    }
  }, [run])
  const stepAxis = useAxis({ label: 'iteration', range: [0, run?.steps ?? 1], key, integer: true })
  const errAxis = useAxis({ label: 'error', log: true, hold: 'union', key })

  const g = shot?.gradient
  const flowTitle =
    D === 1
      ? 'x(t) through the 1-d flow'
      : run?.kind === 'sonode'
        ? 'positions x(t) (state is x and velocity)'
        : 'the flow at time t'
  const training = useTrainingRow({
    loss: run?.loss ?? null,
    lossLabel: run?.task === 'regression' ? 'squared error' : 'cross-entropy',
    nfeForward: run?.nfeForward ?? null,
    nfeBackward: run?.nfeBackward ?? null,
    steps: run?.steps ?? 1,
    marker,
    extra: (
      <DashboardCell>
        <Plot
          x={stepAxis}
          y={errAxis}
          title={run?.kind === 'resnet' ? 'no ODE: no adjoint' : 'adjoint against backprop'}
        >
          {gradients && gradients.x.length > 0 && (
            <Curve name="gradient relative error" x={gradients.x} y={gradients.rel} slot={5} showPoints />
          )}
          {gradients && gradients.x.length > 0 && (
            <Curve name="x(0) reconstruction error" x={gradients.x} y={gradients.rec} slot={6} showPoints dashed />
          )}
          {marker}
        </Plot>
      </DashboardCell>
    ),
  })
  return {
    readouts: (
      <>
        <Readout label="iteration" value={shot ? shot.step : '—'} />
        <Readout label={run?.task === 'regression' ? 'MSE' : 'accuracy'} value={f3(shot?.metric)} />
        <Readout
          label="NFE forward / backward"
          value={run ? `${f3(meanOf(run.nfeForward))} / ${f3(meanOf(run.nfeBackward))}` : '—'}
        />
        <Readout label="ms per iteration" value={f3(meanOf(run?.wallMs))} />
        <Readout label="adjoint vs backprop" value={g ? f3(g.relativeError) : '—'} />
        <Readout
          label="memory, backprop : adjoint"
          value={g ? `${Math.round(g.backpropMemory / g.adjointMemory)} : 1` : '—'}
        />
      </>
    ),
    caption: (
      <>
        aifn odeRun (aifn-applied/neural/ode) on the core OdeBlock (aifn/nn/layers) and odeFlow (aifn/dynamics/ode),
        trained in the worker by Adam on minibatches of 128. Top left: the learned field (arrows, one scale per
        checkpoint; per frame when f depends on t), the trails of {P || 160} points from t = 0 and their positions at t
        (class colours as in the data; for g(x) = −x the sign of x(0)). A 1-d neural ODE cannot map x to −x: its
        trajectories x(t) never cross, so the flow is increasing. A disc inside a ring defeats a plain 2-d NODE for the
        same reason: it can only squeeze the ring between the finite points, with a contorted flow; the augmented
        NODE&apos;s extra coordinate a (middle) lifts the disc over the ring. Right: P(class 1) over the input plane
        (with the decision boundary on, the ink line is its 0.5 contour), or the fitted g. Below, the training figure:
        loss, function evaluations per iteration (forward, and backward: the adjoint&apos;s backward solve, or
        backprop&apos;s replay of the recorded steps), and at each checkpoint the adjoint gradient&apos;s relative error
        against backprop&apos;s and the error of x(0) reconstructed by integrating backwards (checkpoints shrink it).
        Play time t; the slider (or the iteration marker in the training figure) picks the model.{' '}
        {error ? `Run stopped: ${error}` : ''}
      </>
    ),
    body: (
      <>
        <Dashboard>
          <DashboardRow ratio={1.3}>
            <DashboardCell>
              <Plot x={px} y={py} title={pending ?? flowTitle}>
                {vectors.length > 0 && <Vectors vectors={vectors} />}
                {drawn?.segs.map((s, k) => s.length > 0 && <Segments key={k} segments={s} slot={k} width={1} />)}
                {drawn && (
                  <Points
                    name="points"
                    x={drawn.now.x}
                    y={drawn.now.y}
                    group={Array.from(group)}
                    groupNames={groupNames}
                    size={5}
                  />
                )}
              </Plot>
            </DashboardCell>
            {augmented && (
              <DashboardCell>
                <Plot x={ax} y={aa} title="the augmented coordinate a against z₁">
                  {augmented.segs.map((s, k) => s.length > 0 && <Segments key={k} segments={s} slot={k} width={1} />)}
                  <Points
                    name="points"
                    x={augmented.now.x}
                    y={augmented.now.y}
                    group={Array.from(group)}
                    groupNames={groupNames}
                    size={5}
                  />
                </Plot>
              </DashboardCell>
            )}
            <DashboardCell>
              <Plot
                x={qx}
                y={qy}
                title={run?.task === 'regression' ? 'g(x) = −x and the fit' : 'P(class 1) over the inputs'}
              >
                {decision && run && (
                  <Raster
                    x={run.decisionAxis}
                    y={run.decisionAxis}
                    z={decision}
                    scale="diverging"
                    range={[0, 1]}
                    valueLabel="P(class 1)"
                    fillOpacity={0.7}
                    boundary={boundary ? 0.5 : false}
                  />
                )}
                {data && run?.task === 'classification' && (
                  <Points name="data" x={data.x} y={data.y} group={data.g} groupNames={groupNames} thin />
                )}
                {data && run?.task === 'regression' && (
                  <Points name="targets" x={data.x} y={data.y} group={data.g} groupNames={groupNames} thin />
                )}
                {run?.task === 'regression' && shot?.decision && (
                  <Curve name="fit" x={Array.from(run.decisionAxis)} y={Array.from(shot.decision)} emphasis />
                )}
              </Plot>
            </DashboardCell>
          </DashboardRow>
        </Dashboard>
        {player}
        <TimePlayer frame={f} setFrame={setFrame} times={run?.times ?? null} />
      </>
    ),
    training,
  }
}

// ── Continuous normalising flow ───────────────────────────────────────────────────────────────────────────────────

function useCnfView({ run, runKey, error, pending }: ViewProps<CnfRun>): View {
  const { shot, marker, player } = useCheckpoint(run, runKey)
  const [frame, setFrame] = usePick(runKey)
  const T = run?.times.length ?? 1
  const f = Math.min(frame, T - 1)
  const box = run?.box ?? 3
  const x1 = useAxis({ label: 'x₁', range: [-box, box], key: runKey })
  const x2 = useAxis({ label: 'x₂', range: [-box, box], key: runKey, equal: x1 })
  const g = run?.gridAxis.length ?? 0
  const density = useMemo(() => (shot && g ? rows(shot.density[f], g, Math.exp) : null), [shot, f, g])
  const top = useMemo(() => {
    if (!run) return 0.2
    let m = 0
    for (const c of run.checkpoints) for (const d of c.density) for (const v of d) m = Math.max(m, Math.exp(v))
    return m || 0.2
  }, [run])
  const scale = useMemo(() => {
    if (!run || !shot) return 1
    const cell = Math.abs(run.fieldAxis[1] - run.fieldAxis[0])
    return (0.7 * cell) / typicalLength(shot.field)
  }, [run, shot])
  const vectors = useMemo(
    () => (run && shot ? arrows(shot.field[f], run.fieldAxis, run.fieldAxis, scale) : []),
    [run, shot, f, scale],
  )
  const samples = useMemo(() => {
    if (!run || !shot) return null
    const S = shot.samples.length / (2 * T)
    return at(shot.samples, S, 2, f, 0, 1)
  }, [run, shot, f, T])
  const data = useMemo(() => {
    if (!run) return null
    const n = run.data.length / 2
    return {
      x: Array.from({ length: n }, (_, i) => run.data[2 * i]),
      y: Array.from({ length: n }, (_, i) => run.data[2 * i + 1]),
    }
  }, [run])
  const hutch = useMemo(() => {
    if (!run) return null
    const c = run.checkpoints
    return {
      x: c.map((k) => k.step),
      exact: c.map((k) => k.hutchinson.exact),
      mean: c.map((k) => k.hutchinson.mean),
      bars: c.map((k) => ({
        from: [k.step, k.hutchinson.mean - k.hutchinson.sd] as [number, number],
        to: [k.step, k.hutchinson.mean + k.hutchinson.sd] as [number, number],
      })),
    }
  }, [run])
  const stepAxis = useAxis({ label: 'iteration', range: [0, run?.steps ?? 1], key: runKey, integer: true })
  const nllAxis = useAxis({ label: 'NLL (nats)', hold: 'union', key: runKey })
  const training = useTrainingRow({
    loss: run?.loss ?? null,
    lossLabel: 'NLL',
    nfeForward: run?.nfeForward ?? null,
    nfeBackward: run?.nfeBackward ?? null,
    steps: run?.steps ?? 1,
    marker,
    extra: (
      <DashboardCell>
        <Plot x={stepAxis} y={nllAxis} title="exact trace against Hutchinson">
          {hutch && hutch.x.length > 0 && <Curve name="exact" x={hutch.x} y={hutch.exact} slot={2} showPoints />}
          {hutch && hutch.x.length > 0 && <Points name="Hutchinson mean" x={hutch.x} y={hutch.mean} slot={5} />}
          {hutch && hutch.x.length > 0 && <Segments segments={hutch.bars} slot={5} width={2} />}
          {marker}
        </Plot>
      </DashboardCell>
    ),
  })
  return {
    readouts: (
      <>
        <Readout label="iteration" value={shot ? shot.step : '—'} />
        <Readout label="NLL (nats per point)" value={f3(shot?.nll)} />
        <Readout
          label="Hutchinson NLL ± sd"
          value={shot ? `${f3(shot.hutchinson.mean)} ± ${f3(shot.hutchinson.sd)}` : '—'}
        />
        <Readout
          label="NFE forward / backward"
          value={run ? `${f3(meanOf(run.nfeForward))} / ${f3(meanOf(run.nfeBackward))}` : '—'}
        />
        <Readout label="ms per iteration" value={f3(meanOf(run?.wallMs))} />
      </>
    ),
    caption: (
      <>
        aifn cnfRun: a continuous normalising flow (FFJORD) with a time-dependent MLP field f(t, x), trained in the
        worker by maximum likelihood. log p₁(x) = log 𝒩(z(0)) + ∫₀¹ tr(∂f/∂x) dt along the trajectory solved from x at t
        = 1 back to t = 0 (core augmentedDynamics; the trace exact by two forward products, or Hutchinson&apos;s
        εᵀ(∂f/∂x)ε from one vjp). Left: the density p_t on a grid (one colour scale per run) with the field at t.
        Middle: 600 base samples carried by the flow from the Gaussian at t = 0 to the data (grey) at t = 1. Below, the
        training figure: the minibatch NLL, function evaluations per iteration, and at each checkpoint the exact NLL of
        128 points against Hutchinson&apos;s estimate over six probes (mean ± sd). Play time t; the slider (or the
        iteration marker in the training figure) picks the model. {error ? `Run stopped: ${error}` : ''}
      </>
    ),
    body: (
      <>
        <Dashboard>
          <DashboardRow ratio={1.3}>
            <DashboardCell>
              <Plot x={x1} y={x2} title={pending ?? 'density pₜ and the field'}>
                {density && run && (
                  <Raster
                    x={run.gridAxis}
                    y={run.gridAxis}
                    z={density}
                    scale="sequential"
                    range={[0, top]}
                    valueLabel="pₜ(x)"
                  />
                )}
                {vectors.length > 0 && <Vectors vectors={vectors} />}
              </Plot>
            </DashboardCell>
            <DashboardCell>
              <Plot x={x1} y={x2} title="samples carried from 𝒩(0, I)">
                {data && <Points name="data" x={data.x} y={data.y} muted thin />}
                {samples && <Points name="samples at t" x={samples.x} y={samples.y} slot={0} thin />}
              </Plot>
            </DashboardCell>
          </DashboardRow>
        </Dashboard>
        {player}
        <TimePlayer frame={f} setFrame={setFrame} times={run?.times ?? null} />
      </>
    ),
    training,
  }
}

// ── Latent ODE ────────────────────────────────────────────────────────────────────────────────────────────────────

function useLatentView({ run, runKey, error, pending }: ViewProps<LatentOdeRun>): View {
  const { shot, marker, player } = useCheckpoint(run, runKey)
  const [frame, setFrame] = usePick(runKey)
  const T = run?.times.length ?? 1
  const f = Math.min(frame, T - 1)
  const D = run?.dim ?? 1
  const S = run ? run.mask.length / T : 0
  const duration = run ? run.times[T - 1] : 4
  const tAxis = useAxis({ label: 't', range: [0, duration], key: runKey })
  const xAxis = useAxis({ label: D === 1 ? 'x(t)' : 'x₂', hold: 'initial', key: runKey })
  const p1 = useAxis({ label: 'x₁', range: [-1.6, 1.6], key: runKey })
  const p2 = useAxis({ label: 'x₂', range: [-1.6, 1.6], key: runKey, equal: p1 })
  const z1 = useAxis({ label: 'z₁', hold: 'initial', key: runKey })
  const z2 = useAxis({ label: 'z₂', hold: 'initial', key: runKey, equal: z1 })
  const traj = useMemo(() => {
    if (!run) return []
    return Array.from({ length: S }, (_, i) => {
      const val = (a: Float64Array, j: number, k: number) => a[(i * T + j) * D + k]
      const obs = Array.from({ length: T }, (_, j) => j).filter((j) => run.mask[i * T + j] > 0)
      const t = Array.from(run.times)
      return {
        truth: {
          x: D === 1 ? t : t.map((_, j) => val(run.truth, j, 0)),
          y: t.map((_, j) => val(run.truth, j, D === 1 ? 0 : 1)),
        },
        obs: {
          x: obs.map((j) => (D === 1 ? run.times[j] : val(run.observations, j, 0))),
          y: obs.map((j) => val(run.observations, j, D === 1 ? 0 : 1)),
        },
        pred: shot
          ? {
              x: t.slice(0, f + 1).map((tt, j) => (D === 1 ? tt : val(shot.predictions, j, 0))),
              y: t.slice(0, f + 1).map((_, j) => val(shot.predictions, j, D === 1 ? 0 : 1)),
            }
          : null,
      }
    })
  }, [run, shot, f, S, T, D])
  const latents = useMemo(() => {
    if (!run || !shot) return []
    const L = run.latent
    return Array.from({ length: S }, (_, i) => ({
      x: Array.from({ length: f + 1 }, (_, j) => shot.latents[(j * S + i) * L]),
      y: Array.from({ length: f + 1 }, (_, j) => shot.latents[(j * S + i) * L + 1]),
    }))
  }, [run, shot, f, S])
  const quiver = useMemo(() => {
    if (!run || !shot?.field || run.fieldAxis.length < 2) return []
    const cell = Math.abs(run.fieldAxis[1] - run.fieldAxis[0])
    return arrows(shot.field, run.fieldAxis, run.fieldAxis, (0.7 * cell) / typicalLength([shot.field]))
  }, [run, shot])
  const errors = useMemo(() => {
    if (!run) return null
    const c = run.checkpoints
    return { x: c.map((k) => k.step), i: c.map((k) => k.interpolation), e: c.map((k) => k.extrapolation) }
  }, [run])
  const stepAxis = useAxis({ label: 'iteration', range: [0, run?.steps ?? 1], key: runKey, integer: true })
  const mseAxis = useAxis({ label: 'MSE against the truth', log: true, hold: 'union', key: runKey })
  const window = run ? run.times[run.observedUntil - 1] : 2
  const training = useTrainingRow({
    loss: run?.loss ?? null,
    lossLabel: 'negative ELBO per point',
    nfeForward: run?.nfeForward ?? null,
    nfeBackward: run?.nfeBackward ?? null,
    steps: run?.steps ?? 1,
    marker,
    extra: (
      <DashboardCell>
        <Plot x={stepAxis} y={mseAxis} title="interpolation and extrapolation">
          {errors && errors.x.length > 0 && (
            <Curve name="interpolation" x={errors.x} y={errors.i} slot={2} showPoints />
          )}
          {errors && errors.x.length > 0 && (
            <Curve name="extrapolation" x={errors.x} y={errors.e} slot={5} showPoints dashed />
          )}
          {marker}
        </Plot>
      </DashboardCell>
    ),
  })
  return {
    readouts: (
      <>
        <Readout label="iteration" value={shot ? shot.step : '—'} />
        <Readout label="interpolation MSE" value={f3(shot?.interpolation)} />
        <Readout label="extrapolation MSE" value={f3(shot?.extrapolation)} />
        <Readout
          label="NFE forward / backward"
          value={run ? `${f3(meanOf(run.nfeForward))} / ${f3(meanOf(run.nfeBackward))}` : '—'}
        />
        <Readout label="ms per iteration" value={f3(meanOf(run?.wallMs))} />
      </>
    ),
    caption: (
      <>
        aifn latentOdeRun: a latent ODE (Rubanova et al., 2019), minimal. A GRU reads each trajectory&apos;s observed
        points backwards (irregular: about 40% of the grid times in [0, {f3(window)}]) into q(z₀); z₀ is drawn by
        reparameterisation, a neural ODE in a 2-d latent space carries it over [0, {f3(duration)}], and a linear decoder
        maps z(t) to x(t). Trained on the ELBO of the observed points; after t = {f3(window)} the model extrapolates.
        Left: four trajectories&apos; truth (dashed), observations (points) and the decoded path at z₀ = μ up to the
        time t. Right: their latent paths over the latent field. Below, the training figure: loss, work, and the
        interpolation and extrapolation errors at checkpoints. Play time t; the slider (or the training figure&apos;s
        marker) picks the model. {error ? `Run stopped: ${error}` : ''}
      </>
    ),
    body: (
      <>
        <Dashboard>
          <DashboardRow ratio={1.3}>
            <DashboardCell>
              <Plot
                x={D === 1 ? tAxis : p1}
                y={D === 1 ? xAxis : p2}
                title={pending ?? 'observations, truth and the decoded path'}
              >
                {D === 1 && <Annotation x={window} text="end of observations" dashed />}
                {traj.map((tr, k) => (
                  <Curve
                    key={`t${k}`}
                    name={`trajectory ${k + 1}`}
                    x={tr.truth.x}
                    y={tr.truth.y}
                    slot={k}
                    dashed
                    thin
                  />
                ))}
                {traj.map((tr, k) => (
                  <Points key={`o${k}`} name={`trajectory ${k + 1}`} x={tr.obs.x} y={tr.obs.y} slot={k} size={5} />
                ))}
                {traj.map(
                  (tr, k) =>
                    tr.pred && (
                      <Curve key={`p${k}`} name={`trajectory ${k + 1}`} x={tr.pred.x} y={tr.pred.y} slot={k} />
                    ),
                )}
              </Plot>
            </DashboardCell>
            <DashboardCell>
              <Plot x={z1} y={z2} title="latent paths z(t) and the latent field">
                {quiver.length > 0 && <Vectors vectors={quiver} />}
                {latents.map((l, k) => (
                  <Curve key={k} name={`trajectory ${k + 1}`} x={l.x} y={l.y} slot={k} />
                ))}
                {latents.length > 0 && (
                  <Points
                    name="z(t)"
                    x={latents.map((l) => l.x[l.x.length - 1])}
                    y={latents.map((l) => l.y[l.y.length - 1])}
                    emphasis
                    size={6}
                  />
                )}
              </Plot>
            </DashboardCell>
          </DashboardRow>
        </Dashboard>
        {player}
        <TimePlayer frame={f} setFrame={setFrame} times={run?.times ?? null} />
      </>
    ),
    training,
  }
}
