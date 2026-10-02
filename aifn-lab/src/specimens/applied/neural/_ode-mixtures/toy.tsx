/**
 * "Neural ODEs with stochastic vector field mixtures: the toy tasks": the failure cases of fig. 1 and moons, nested
 * circles and XOR (§4.1.1), trained in the worker with any member of fig. 6's lattice, its losses, K and the component
 * selection; the learned transforms over time (figs. 8–9), each component's field and the component posterior along a
 * pinned path.
 */
import { useMemo, useState } from 'react'
import type { SvfmRun, SvfmRunOptions } from 'aifn-applied/neural/ode-mixtures'
import { Player, Select, Switch } from '@lab/controls'
import { Dashboard, DashboardCell, DashboardRow, Figure } from '@lab/layout'
import { call, choice, float, int, row, setting, useFigureState, type Task } from '@lab/state'
import { TrainControls, useTrainedRun } from '@lab/views'
import {
  Annotation,
  argmaxMargins,
  Contours,
  Curve,
  Plot,
  Points,
  Raster,
  Readout,
  Rug,
  Segments,
  useAxis,
  Vectors,
  type Vector,
} from '@lab/viz'
import {
  architecture,
  arrows,
  fieldRow,
  lossRow,
  piRow,
  solverRow,
  f3,
  isMixture,
  meanOf,
  modelName,
  modelOptions,
  MODELS,
  pathLosses,
  rows,
  SELECTIONS,
  typicalLength,
  useCheckpoint,
  usePick,
  type ModelKind,
  type PathLoss,
} from './shared'
import { TrainingFigure } from './training-figure'

const TASKS = [
  { value: 'crossing', label: 'crossing (fig. 1a)' },
  { value: 'splitting', label: 'splitting (fig. 1b)' },
  { value: 'scaling', label: 'scaling (fig. 1c)' },
  { value: 'moons', label: 'moons' },
  { value: 'circles', label: 'nested circles' },
  { value: 'xor', label: 'XOR' },
] as const
type TaskKind = (typeof TASKS)[number]['value']
const ONE_D = new Set<TaskKind>(['crossing', 'splitting', 'scaling'])

type Settings = {
  task: TaskKind
  model: ModelKind
  components: number
  selection: string
  augment: boolean
  path: PathLoss
  lambdaT: number
  lambdaV: number
  steps: number
  learningRate: number
  seed: number
  arch: Partial<SvfmRunOptions>
}

/** The worker task of a setting: the data (1,000 points for the 2-d sets, as §4.1.1) and the run. */
function taskOf(s: Settings): Task<SvfmRun> {
  const random = call('foundation/random/stream', s.seed + 1)
  const data = ONE_D.has(s.task)
    ? call(
        'applied/neural/ode-mixtures/endpointTask',
        call('applied/data/synthetic/odeFailureCase', random, { kind: s.task, n: 200 }),
      )
    : call(
        'applied/neural/ode-mixtures/classificationTask',
        s.task === 'moons'
          ? call('applied/data/synthetic/moons', random, { n: 1000, noise: 0.1 })
          : s.task === 'circles'
            ? call('applied/data/synthetic/circles', random, { n: 1000, noise: 0.05, factor: 0.5 })
            : call('applied/data/synthetic/xor', random, { n: 1000, kind: 'gaussian', sd: 0.35 }),
      )
  const options: SvfmRunOptions = {
    ...modelOptions(s.model, s.components, s.selection, s.augment),
    shown: ONE_D.has(s.task) ? 120 : 200,
    ...s.arch,
    losses: pathLosses(s.path, s.lambdaT, s.lambdaV),
    steps: s.steps,
    learningRate: s.learningRate,
    seed: s.seed,
  }
  return call<SvfmRun>('applied/neural/ode-mixtures/svfmRun', data, options)
}

export function SvfmToyShowcase() {
  const state = useFigureState({
    setup: row('1 · task and model', {
      task: choice(TASKS, 'splitting', { label: 'task' }),
      model: choice(MODELS, 'svfm', { label: 'model' }),
      components: choice([2, 4, 8], 4, { label: 'components K', when: (v) => isMixture(v.model) }),
      selection: choice(SELECTIONS, 'pick-and-stick', {
        label: 'component selection',
        when: (v) => isMixture(v.model),
      }),
      augment: setting(false, { label: 'augmented (A-, one extra dimension)' }),
    }),
    losses: lossRow('2 · path losses'),
    pi: piRow(),
    fields: fieldRow(),
    solver: solverRow(5, 200),
    train: row('training', {
      steps: int(300, { ge: 1, suggestions: [150, 300, 500, 1000], label: 'iterations' }),
      learningRate: float(0.01, { label: 'Adam rate', gt: 0, le: 0.1, scale: 'log10', suggestions: [0.003, 0.01] }),
      seed: int(0, { label: 'seed', ge: 0, le: 9999 }),
    }),
  })
  const { setup, losses, train } = state
  const settings: Settings = {
    task: setup.task as TaskKind,
    model: setup.model as ModelKind,
    components: Number(setup.components),
    selection: String(setup.selection),
    augment: Boolean(setup.augment),
    path: losses.path as PathLoss,
    lambdaT: Number(losses.lambdaT),
    lambdaV: Number(losses.lambdaV),
    steps: Number(train.steps),
    learningRate: train.learningRate,
    seed: train.seed,
    arch: architecture(state.pi, state.fields, state.solver),
  }
  const trained = useTrainedRun(settings, taskOf)
  const run = trained.trained ? trained.run.value : null
  const runKey = trained.trained
  const done = run?.done ?? 0
  const total = trained.trained?.steps ?? settings.steps
  const { shot, marker, slider } = useCheckpoint(run, runKey)
  const [frame, setFrame] = usePick(runKey)
  const [pinned, setPinned] = usePick(runKey)
  const [component, setComponent] = useState(0)
  const [contours, setContours] = useState(true)

  const P = run?.shown.length ?? 0
  const S = run?.stateDim ?? 2
  const D = run?.dim ?? 2
  const K = run?.components ?? 1
  const T = run ? run.gridTimes.length - 1 : 1
  const F = run ? run.frameTimes.length : 1
  const f = Math.min(frame, F - 1)
  const t = run?.frameTimes[f] ?? 0
  const gi = Math.round(t * T)
  const k = Math.min(component, K - 1)
  const pin = Math.min(pinned, Math.max(0, P - 1))
  const oneD = D === 1
  const classes = run?.task === 'classification' ? run.classes : 2
  const groupOf = useMemo(() => (run ? Int32Array.from(run.shown, (i) => run.groups[i]) : new Int32Array(0)), [run])
  const groupNames =
    run?.task === 'classification'
      ? Array.from({ length: classes }, (_, c) => `class ${c}`)
      : settings.task === 'scaling'
        ? ['points']
        : settings.task === 'crossing'
          ? ['x(0) < 0', 'x(0) > 0']
          : ['to −1', 'to +1']

  const box = run?.box ?? 2
  const extent = useMemo(() => {
    if (!run) return box
    const all: number[] = []
    for (const c of run.checkpoints) for (let i = 0; i < c.paths.length; i += S) all.push(Math.abs(c.paths[i]))
    all.sort((p, q) => p - q)
    const m = all.length ? all[Math.floor(0.98 * (all.length - 1))] : box
    return Math.ceil(Math.min(Math.max(m * 1.1, box), 2 * box) * 2) / 2
  }, [run, box, S])
  const px = useAxis({ label: oneD ? 't' : 'h₁', range: oneD ? [0, 1] : [-extent, extent], key: [runKey, extent] })
  const py = useAxis({
    label: oneD ? 'h(t)' : 'h₂',
    range: [-extent, extent],
    key: [runKey, extent],
    equal: oneD ? undefined : px,
  })
  const qx = useAxis({ label: oneD ? 'h' : 'x₁', range: [-box, box], key: runKey })
  const qy = useAxis({
    label: oneD ? 'density' : 'x₂',
    range: oneD ? [0, undefined] : [-box, box],
    hold: oneD ? 'union' : undefined,
    key: runKey,
    equal: oneD ? undefined : qx,
  })
  const tAxis = useAxis({ label: 't', range: [0, 1], key: runKey })
  const piAxis = useAxis({ label: 'πₖ(t)', range: [0, 1], key: runKey })

  // The pinned point is the shown point nearest a click on the flow (at the frame shown).
  const pinAt = ([cx, cy]: [number, number]) => {
    if (!run || !shot) return
    let best = 0
    let bestD = Infinity
    for (let b = 0; b < P; b++) {
      const at = (f * P + b) * S
      const d = oneD ? Math.abs(shot.paths[at] - cy) : Math.hypot(shot.paths[at] - cx, shot.paths[at + 1] - cy)
      if (d < bestD) {
        bestD = d
        best = b
      }
    }
    setPinned(best)
  }

  const drawn = useMemo(() => {
    if (!run || !shot) return null
    const groups = Math.max(...Array.from(groupOf, (g) => g + 1), 1)
    const segs: { from: [number, number]; to: [number, number] }[][] = Array.from({ length: groups }, () => [])
    const xy = (g: number, b: number): [number, number] =>
      oneD
        ? [run.frameTimes[g], shot.paths[(g * P + b) * S]]
        : [shot.paths[(g * P + b) * S], shot.paths[(g * P + b) * S + 1]]
    for (let g = 0; g < f; g++) for (let b = 0; b < P; b++) segs[groupOf[b]].push({ from: xy(g, b), to: xy(g + 1, b) })
    const now = Array.from({ length: P }, (_, b) => xy(f, b))
    const pinPath = Array.from({ length: f + 1 }, (_, g) => xy(g, pin))
    return { segs, now, pinPath }
  }, [run, shot, f, P, S, oneD, groupOf, pin])

  const quiver = useMemo((): Vector[] => {
    if (!run || !shot) return []
    const field = shot.fields[k]
    if (!field) return []
    if (!oneD) {
      const cell = Math.abs(run.fieldAxis[1] - run.fieldAxis[0])
      return arrows(field[gi], run.fieldAxis, (0.7 * cell) / typicalLength(field))
    }
    // 1-d: the direction field (1, f(t, h)) on the (t, h) grid, scaled per axis.
    const xs = run.fieldAxis
    const dt = 1 / T
    const dx = Math.abs(xs[1] - xs[0])
    let m = 0
    for (const g of field) for (const v of g) m = Math.max(m, Math.abs(v))
    const out: Vector[] = []
    for (let i = 0; i <= T; i++)
      for (let j = 0; j < xs.length; j += 1) {
        const v = (field[i][j] / Math.max(1e-9, m)) * (dx / dt)
        const len = Math.hypot(1, v)
        out.push({
          from: [run.gridTimes[i], xs[j]],
          to: [run.gridTimes[i] + (0.6 * dt) / len, xs[j] + ((0.6 * dt) / len) * v],
          head: 5,
          width: 1,
          muted: true,
        })
      }
    return out
  }, [run, shot, k, gi, oneD, T])

  const decision = useMemo(
    () => (run && shot && shot.decision.length ? rows(shot.decision, run.rasterAxis.length) : null),
    [run, shot],
  )
  const prior = useMemo(() => {
    if (!run || !shot || K === 1 || !shot.prior.length) return null
    const g = run.rasterAxis.length
    if (oneD)
      return { curves: Array.from({ length: K }, (_, c) => Array.from({ length: g }, (_, i) => shot.prior[i * K + c])) }
    // Regions: the most probable component at t₀ (fixed component slots); boundaries: the argmax margins at 0.
    const fields = Array.from({ length: K }, (_, c) =>
      rows(
        Array.from({ length: g * g }, (__, i) => shot.prior[i * K + c]),
        g,
      ),
    )
    const regions = rows(
      Array.from({ length: g * g }, (_, i) => {
        let best = 0
        for (let c = 1; c < K; c++) if (shot.prior[i * K + c] > shot.prior[i * K + best]) best = c
        return best
      }),
      g,
    )
    const margins = argmaxMargins(fields)
    return { regions, margins: K === 2 ? margins.slice(0, 1) : margins }
  }, [run, shot, K, oneD])
  // 1-d: the predictive density at the grid time shown, averaged over the shown points (fig. 1's Gaussian PDFs).
  const density = useMemo(() => {
    if (!run || !shot || !oneD) return null
    const xs = Array.from({ length: 201 }, (_, i) => -box + (2 * box * i) / 200)
    const y = xs.map((x) => {
      let s = 0
      for (let b = 0; b < P; b++)
        for (let c = 0; c < K; c++) {
          const at = ((gi * P + b) * K + c) * 3
          const [w, mu, sd] = [shot.mixture[at], shot.mixture[at + 1], shot.mixture[at + 2]]
          s += (w * Math.exp(-0.5 * ((x - mu) / sd) ** 2)) / (sd * Math.sqrt(2 * Math.PI))
        }
      return s / Math.max(1, P)
    })
    return { x: xs, y }
  }, [run, shot, oneD, gi, P, K, box])
  const posterior = useMemo(() => {
    if (!run || !shot || K === 1) return null
    return Array.from({ length: K }, (_, c) =>
      Array.from({ length: T + 1 }, (_, i) => shot.weights[(i * P + pin) * K + c]),
    )
  }, [run, shot, K, T, P, pin])
  const data2d = useMemo(() => {
    if (!run || oneD) return null
    const n = run.data.length / 2
    return {
      x: Array.from({ length: n }, (_, i) => run.data[2 * i]),
      y: Array.from({ length: n }, (_, i) => run.data[2 * i + 1]),
      g: Array.from(run.groups),
    }
  }, [run, oneD])

  const pending = !trained.trained ? 'press Train to start' : !run ? 'training…' : undefined
  const name = modelName(run, trained.trained?.path)
  const controls = (
    <TrainControls run={trained as never} progress={done / total} progressText={`${done} / ${total} iterations`} />
  )
  return (
    <>
      <Figure
        title="The learned transform"
        id="the-transform"
        purpose="Where each model carries the points over the integration interval, with each component's field and the component posterior along a path."
        state={state}
        defaultSize="XL"
        controls={controls}
        readouts={
          <>
            <Readout label="model" value={name || '—'} />
            <Readout label="iteration" value={shot ? shot.step : '—'} />
            <Readout
              label={run?.task === 'classification' ? 'accuracy' : 'predictive loss'}
              value={run?.task === 'classification' ? f3(shot?.accuracy) : f3(shot?.predictive)}
            />
            <Readout label="TLoss / VLoss" value={shot ? `${f3(shot.transport)} / ${f3(shot.variance)}` : '—'} />
            <Readout
              label="NFE mean per point / all as one"
              value={shot ? `${f3(meanOf(shot.nfe))} / ${shot.batchNfe}` : '—'}
            />
            <Readout label="parameters" value={run ? run.parameters : '—'} />
          </>
        }
        caption={
          <>
            aifn svfmRun (aifn-applied/neural/ode-mixtures): a neural ODE ∇h(t) = f(h(t), t) whose VF is one of K
            components; π chooses which field and the fields carry the dynamics, so π is simple (linear by default, at
            most one layer of 16) and the capacity sits in the component VFs. The components are VF units (fig. 2a) or
            stochastic SVF units (fig. 3: a direction on the sphere and a log-normal length), chosen by pick and stick
            (π fixed at π(t₀), eq. 2) or by forward filtering (π(tᵢ) ∝ Ψ(tᵢ)ᵀ(ψ(tᵢ) ⊙ π(tᵢ₋₁)), eq. 5) on a grid of 5
            intervals; trained in the worker by Adam on minibatches of 50 with the mixture density loss (eq. 10;
            cross-entropy or squared error for a single VF) plus λ times TLoss and VLoss as chosen. Top left: realised
            paths of the shown points (one component draw and one field sample per path, held for the whole solve as in
            §3) up to the time t, coloured by {oneD ? 'their start or target side' : 'class'}, over component k&apos;s
            mean field at the nearest grid time. A 1-d neural ODE cannot cross or split; mixtures take one component per
            branch, and stochastic VFs spread along one direction (scaling). Top right:{' '}
            {oneD
              ? 'the predictive density at the grid time shown, averaged over the points (the targets as ticks)'
              : 'P(class 1) over the inputs (as figs. 8–9)'}
            . Bottom: the component posterior π(t) along the pinned path (click a point on the flow to pin it) and the
            regions where each component is the most probable at t₀ (π(t₀), the pick-and-stick choice), with thin ink
            lines on the boundaries where the most probable component changes (the level-0 contours of πₖ − max over j ≠
            k of πⱼ; for K = 2 the π₁ = 0.5 contour). Results vary with the seed: a mixture can gate a component off
            before it has specialised. Play time t; the slider (or the training figure&apos;s marker) picks the
            checkpoint.
            {run?.error ? ` Run stopped: ${run.error}` : ''}
          </>
        }
      >
        <Dashboard>
          <DashboardRow ratio={1.6}>
            <DashboardCell>
              <Plot x={px} y={py} title={pending ?? `${name}: the flow at time t`} onPlotClick={pinAt}>
                {quiver.length > 0 && <Vectors vectors={quiver} />}
                {drawn?.segs.map((s, g) => s.length > 0 && <Segments key={g} segments={s} slot={g} width={1} />)}
                {drawn && (
                  <Points
                    name="points"
                    x={drawn.now.map((p) => p[0])}
                    y={drawn.now.map((p) => p[1])}
                    group={Array.from(groupOf)}
                    groupNames={groupNames}
                    size={4}
                  />
                )}
                {drawn && (
                  <Curve
                    name="pinned path"
                    x={drawn.pinPath.map((p) => p[0])}
                    y={drawn.pinPath.map((p) => p[1])}
                    emphasis
                  />
                )}
              </Plot>
            </DashboardCell>
            <DashboardCell>
              <Plot
                x={qx}
                y={qy}
                title={oneD ? `predictive density at t = ${f3(run?.gridTimes[gi] ?? 0)}` : 'P(class 1) over the inputs'}
              >
                {decision && run && (
                  <Raster
                    x={run.rasterAxis}
                    y={run.rasterAxis}
                    z={decision}
                    scale="diverging"
                    range={[0, 1]}
                    valueLabel="P(class 1)"
                    fillOpacity={0.7}
                  />
                )}
                {data2d && (
                  <Points name="data" x={data2d.x} y={data2d.y} group={data2d.g} groupNames={groupNames} thin />
                )}
                {density && <Curve name="predictive density" x={density.x} y={density.y} slot={2} />}
                {oneD && run && <Rug name="targets" values={run.targets} />}
              </Plot>
            </DashboardCell>
          </DashboardRow>
          <DashboardRow ratio={1}>
            <DashboardCell>
              <Plot
                x={tAxis}
                y={piAxis}
                title={K > 1 ? 'component posterior π(t) along the pinned path' : 'one VF: π = 1'}
              >
                {posterior?.map((c, i) => (
                  <Curve key={i} name={`component ${i + 1}`} x={Array.from(run!.gridTimes)} y={c} slot={i} showPoints />
                ))}
                <Annotation x={t} dashed />
              </Plot>
            </DashboardCell>
            <DashboardCell>
              <Plot
                x={qx}
                y={oneD ? piAxis : qy}
                title={
                  K > 1
                    ? oneD
                      ? 'prior π(t₀) over the start h(t₀)'
                      : 'most probable component at t₀'
                    : 'one VF: no prior'
                }
              >
                {prior?.regions && run && (
                  <Raster
                    x={run.rasterAxis}
                    y={run.rasterAxis}
                    z={prior.regions}
                    scale="categorical"
                    categoryNames={Array.from({ length: K }, (_, c) => `component ${c + 1}`)}
                    fillOpacity={0.55}
                  />
                )}
                {contours &&
                  prior?.margins?.map((m, c) => (
                    <Contours
                      key={c}
                      x={Array.from(run!.rasterAxis)}
                      y={Array.from(run!.rasterAxis)}
                      z={m}
                      levels={[0]}
                      labels={false}
                    />
                  ))}
                {prior?.curves?.map((c, i) => (
                  <Curve key={i} name={`component ${i + 1}`} x={Array.from(run!.rasterAxis)} y={c} slot={i} />
                ))}
                {data2d && <Points name="data" x={data2d.x} y={data2d.y} muted thin />}
              </Plot>
            </DashboardCell>
          </DashboardRow>
        </Dashboard>
        <div className="flex flex-wrap items-end gap-4">
          <Select
            label="field shown"
            value={String(k)}
            onChange={(v) => setComponent(Number(v))}
            options={Array.from({ length: K }, (_, i) => ({ value: String(i), label: `component ${i + 1}` }))}
          />
          <Switch label="decision contours (argmax boundaries of π(t₀))" checked={contours} onChange={setContours} />
        </div>
        {slider}
        <Player
          label="time t"
          value={f}
          onChange={setFrame}
          count={Math.max(1, F)}
          format={(i) => `t = ${f3(run?.frameTimes[i] ?? 0)}`}
        />
      </Figure>
      <TrainingFigure
        id="training"
        run={run}
        runKey={runKey}
        marker={marker}
        lossLabel={run?.task === 'classification' ? 'objective' : 'objective'}
      />
    </>
  )
}
