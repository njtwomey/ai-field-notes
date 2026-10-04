/**
 * "Neural ODEs with stochastic vector field mixtures: behaviour": forecasting walks in a house (§4.4, fig. 12) on a
 * synthetic analogue of the paper's data, with fig. 6's lattice of models: a VF goes to the mean of the four
 * destinations, components of a mixture take the branches, and stochastic VFs spread the forecasts along them.
 */
import { useMemo, useState } from 'react'
import { FLOORPLAN } from 'aifn-methods/data/synthetic'
import type { SvfmRun, SvfmRunOptions } from 'aifn-methods/neural/ode-mixtures'
import { Player, Select } from '@lab/controls'
import { Dashboard, DashboardCell, DashboardRow, Figure } from '@lab/layout'
import { call, choice, float, int, row, setting, useFigureState, type Task } from '@lab/state'
import { TrainControls, useTrainedRun } from '@lab/views'
import { Annotation, Bars, Curve, Plot, Points, Readout, Segments, useAxis } from '@lab/viz'
import {
  architecture,
  f3,
  fieldRow,
  piRow,
  solverRow,
  isMixture,
  meanOf,
  modelName,
  modelOptions,
  MODELS,
  PATH_LOSSES,
  pathLosses,
  SELECTIONS,
  useCheckpoint,
  usePick,
  type ModelKind,
  type PathLoss,
} from './shared'
import { TrainingFigure } from './training-figure'

const TARGETS = FLOORPLAN.targets.map((t) => t.name)
type Seg = { from: [number, number]; to: [number, number] }

type Settings = {
  timeOfDay: 'day' | 'night' | 'mixed'
  walks: number
  context: boolean
  model: ModelKind
  components: number
  selection: string
  augment: boolean
  loss: 'forecast' | 'end'
  path: PathLoss
  lambdaT: number
  lambdaV: number
  steps: number
  learningRate: number
  seed: number
  arch: Partial<SvfmRunOptions>
}

function taskOf(s: Settings): Task<SvfmRun> {
  const walks = call('applied/data/synthetic/floorplanWalks', call('foundation/random/stream', s.seed + 1), {
    n: s.walks,
    timeOfDay: s.timeOfDay,
  })
  const options: SvfmRunOptions = {
    ...modelOptions(s.model, s.components, s.selection, s.augment),
    task: 'forecast',
    grid: 10,
    stepSize: 0.1,
    shown: 200,
    ...s.arch,
    losses: s.loss === 'forecast' ? pathLosses('none', 0, 0, true) : pathLosses(s.path, s.lambdaT, s.lambdaV),
    steps: s.steps,
    learningRate: s.learningRate,
    seed: s.seed,
    framesPerInterval: 3,
    fieldGrid: 9,
    decisionGrid: 8,
  }
  return call<SvfmRun>(
    'applied/neural/ode-mixtures/svfmRun',
    call('applied/neural/ode-mixtures/walkTask', walks, { timeOfDay: s.context }),
    options,
  )
}

const nearestTarget = (x: number, y: number) => {
  let best = 0
  FLOORPLAN.targets.forEach((t, i) => {
    if (
      Math.hypot(t.at[0] - x, t.at[1] - y) <
      Math.hypot(FLOORPLAN.targets[best].at[0] - x, FLOORPLAN.targets[best].at[1] - y)
    )
      best = i
  })
  return best
}

export function SvfmBehaviourShowcase() {
  const state = useFigureState({
    data: row('1 · walks (synthetic)', {
      timeOfDay: choice(
        [
          { value: 'day', label: 'by day (targets equally likely)' },
          { value: 'night', label: 'by night (landing 0.9, kitchen 0.1)' },
          { value: 'mixed', label: 'half by day, half by night' },
        ],
        'day',
        { label: 'when' },
      ),
      walks: int(160, { ge: 8, le: 1000, suggestions: [80, 160, 320], label: 'walks' }),
      context: setting(false, { label: 'time of day as an input (cyclic, §4.2)' }),
    }),
    setup: row('2 · model (fig. 6)', {
      model: choice(MODELS, 'svfm', { label: 'model' }),
      components: choice([2, 4, 8], 4, { label: 'components K', when: (v) => isMixture(v.model) }),
      selection: choice(SELECTIONS, 'pick-and-stick', {
        label: 'component selection',
        when: (v) => isMixture(v.model),
      }),
      augment: setting(false, { label: 'augmented (A-)' }),
    }),
    losses: row('3 · losses', {
      loss: choice(
        [
          { value: 'forecast', label: 'FLoss: the whole path (eq. 11)' },
          { value: 'end', label: 'the end point only' },
        ],
        'forecast',
        { label: 'predictive loss' },
      ),
      // FLoss penalises any deviation from the path, so TLoss and VLoss are offered only without it (§4.1.3).
      path: choice(PATH_LOSSES, 'none', { label: 'TLoss / VLoss', when: (v) => v.loss === 'end' }),
      lambdaT: float(0.1, {
        gt: 0,
        scale: 'log10',
        suggestions: [0.01, 0.1, 1],
        label: 'λ of TLoss',
        when: (v) => v.loss === 'end' && (v.path === 'T' || v.path === 'TV'),
      }),
      lambdaV: float(0.1, {
        gt: 0,
        scale: 'log10',
        suggestions: [0.01, 0.1, 1],
        label: 'λ of VLoss',
        when: (v) => v.loss === 'end' && (v.path === 'V' || v.path === 'TV'),
      }),
    }),
    pi: piRow(),
    fields: fieldRow(),
    solver: solverRow(10, 200),
    train: row('training', {
      steps: int(300, { ge: 1, suggestions: [150, 300, 500, 1000], label: 'iterations' }),
      learningRate: float(0.01, { label: 'Adam rate', gt: 0, le: 0.1, scale: 'log10', suggestions: [0.003, 0.01] }),
      seed: int(0, { label: 'seed', ge: 0, le: 9999 }),
    }),
  })
  const settings: Settings = {
    timeOfDay: state.data.timeOfDay as Settings['timeOfDay'],
    walks: state.data.walks,
    context: Boolean(state.data.context),
    model: state.setup.model as ModelKind,
    components: Number(state.setup.components),
    selection: String(state.setup.selection),
    augment: Boolean(state.setup.augment),
    loss: state.losses.loss as Settings['loss'],
    path: (state.losses.loss === 'end' ? state.losses.path : 'none') as PathLoss,
    lambdaT: Number(state.losses.lambdaT),
    lambdaV: Number(state.losses.lambdaV),
    steps: state.train.steps,
    learningRate: state.train.learningRate,
    seed: state.train.seed,
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
  const [when, setWhen] = useState<'all' | 'day' | 'night'>('all')

  const P = run?.shown.length ?? 0
  const S = run?.stateDim ?? 2
  const K = run?.components ?? 1
  const T = run ? run.gridTimes.length - 1 : 1
  const F = run?.frameTimes.length ?? 1
  const f = Math.min(frame, F - 1)
  const t = run?.frameTimes[f] ?? 0
  const pin = Math.min(pinned, Math.max(0, P - 1))
  const toM = (z0: number, z1: number): [number, number] =>
    run ? [run.shift[0] + run.scale * z0, run.shift[1] + run.scale * z1] : [z0, z1]

  // Which shown walks are drawn: all, or those of the day or the night (their start's hour, when walks are mixed).
  const night = useMemo(() => {
    if (!run) return new Uint8Array(0)
    // walkTask puts the time of day in the context only when asked; the mixed walks alternate day and night.
    return Uint8Array.from(run.shown, (i) =>
      trained.trained?.timeOfDay === 'night' ? 1 : trained.trained?.timeOfDay === 'mixed' ? i % 2 : 0,
    )
  }, [run, trained.trained])
  const mask = useMemo(
    () => Uint8Array.from(night, (v) => (when === 'all' || (when === 'night') === (v === 1) ? 1 : 0)),
    [night, when],
  )
  const keep = (b: number) => mask[b] === 1

  const geometry = useMemo(() => {
    const seg = (w: readonly number[]): Seg => ({ from: [w[0], w[1]], to: [w[2], w[3]] })
    return { walls: FLOORPLAN.walls.map(seg), furniture: FLOORPLAN.furniture.map(seg) }
  }, [])
  const forecasts = useMemo(() => {
    if (!run || !shot) return null
    const end = (b: number) => toM(shot.paths[((F - 1) * P + b) * S], shot.paths[((F - 1) * P + b) * S + 1])
    const target = Int32Array.from({ length: P }, (_, b) => nearestTarget(...end(b)))
    const segs: Seg[][] = TARGETS.map(() => [])
    for (let g = 0; g < f; g++)
      for (let b = 0; b < P; b++) {
        if (!keep(b)) continue
        const p = (g * P + b) * S
        const q = ((g + 1) * P + b) * S
        segs[target[b]].push({ from: toM(shot.paths[p], shot.paths[p + 1]), to: toM(shot.paths[q], shot.paths[q + 1]) })
      }
    const now = Array.from({ length: P }, (_, b) => toM(shot.paths[(f * P + b) * S], shot.paths[(f * P + b) * S + 1]))
    const idx = Array.from({ length: P }, (_, b) => b).filter(keep)
    const share = TARGETS.map((_, k) => idx.filter((b) => target[b] === k).length / Math.max(1, idx.length))
    const pinPath = Array.from({ length: f + 1 }, (_, g) =>
      toM(shot.paths[(g * P + pin) * S], shot.paths[(g * P + pin) * S + 1]),
    )
    return { segs, now, idx, target, share, pinPath }
    // oxlint-disable-next-line react-hooks/exhaustive-deps -- keep and toM read mask and run, which are listed
  }, [run, shot, f, F, P, S, pin, mask])
  const walks = useMemo(() => {
    if (!run) return null
    const segs: Seg[] = []
    const G = T + 1
    for (let b = 0; b < P; b++) {
      if (!keep(b)) continue
      for (let g = 0; g < T; g++) {
        const p = (b * G + g) * 2
        const q = (b * G + g + 1) * 2
        segs.push({ from: toM(run.targets[p], run.targets[p + 1]), to: toM(run.targets[q], run.targets[q + 1]) })
      }
    }
    const counts = TARGETS.map(() => 0)
    let m = 0
    for (let b = 0; b < P; b++)
      if (keep(b)) {
        counts[run.groups[run.shown[b]]]++
        m++
      }
    return { segs, share: counts.map((c) => c / Math.max(1, m)) }
    // oxlint-disable-next-line react-hooks/exhaustive-deps -- keep and toM read mask and run, which are listed
  }, [run, T, P, mask])
  const posterior = useMemo(() => {
    if (!run || !shot || K === 1) return null
    return Array.from({ length: K }, (_, c) =>
      Array.from({ length: T + 1 }, (_, i) => shot.weights[(i * P + pin) * K + c]),
    )
  }, [run, shot, K, T, P, pin])

  const pinAt = ([x, y]: [number, number]) => {
    if (!forecasts) return
    let best = forecasts.idx[0] ?? 0
    for (const b of forecasts.idx)
      if (
        Math.hypot(forecasts.now[b][0] - x, forecasts.now[b][1] - y) <
        Math.hypot(forecasts.now[best][0] - x, forecasts.now[best][1] - y)
      )
        best = b
    setPinned(best)
  }

  const fx = useAxis({ label: 'x (m)', range: [-0.6, FLOORPLAN.width + 0.6], key: 'house' })
  const fy = useAxis({ label: 'y (m)', range: [-0.6, FLOORPLAN.height + 0.6], key: 'house', equal: fx })
  const tAxis = useAxis({ label: 't (× 10 s)', range: [0, 1], key: runKey })
  const piAxis = useAxis({ label: 'πₖ(t)', range: [0, 1], key: runKey })
  const catX = useAxis({ label: 'target', categories: TARGETS })
  const shareY = useAxis({ label: 'share', range: [0, 1], key: runKey })
  const name = modelName(run, trained.trained?.path)
  const pending = !trained.trained ? 'press Train to start' : !run ? 'training…' : undefined

  return (
    <>
      <Figure
        title="Forecasts in the house"
        id="forecasts"
        purpose="Where each model expects a person leaving the sofa to walk, against where the walks went."
        state={state}
        defaultSize="XL"
        controls={
          <TrainControls
            run={trained as never}
            progress={done / total}
            progressText={`${done} / ${total} iterations`}
          />
        }
        readouts={
          <>
            <Readout label="model" value={name || '—'} />
            <Readout label="iteration" value={shot ? shot.step : '—'} />
            <Readout label="predictive loss" value={f3(shot?.predictive)} />
            <Readout
              label="forecasts ending at door / kitchen / landing / study"
              value={forecasts ? forecasts.share.map((v) => `${Math.round(100 * v)}%`).join(' / ') : '—'}
            />
            <Readout label="NFE mean per path" value={f3(shot ? meanOf(shot.nfe) : undefined)} />
          </>
        }
        caption={
          <>
            The walks are synthetic (aifn-methods/data floorplanWalks), made to stand in for the paper&apos;s LiDAR and
            SLAM recordings, which are not available. aifn svfmRun trains a model to forecast the walk from the sofa
            over 10 s (t ∈ [0, 1], a grid of 10 intervals) with FLoss (eq. 11: the mixture density, or the squared error
            for one VF, at every grid time against the walk interpolated by a cubic spline) or on the end point alone.
            Left: the house, the shown walks (grey) and 200 forecasts from their starts up to the time t, each a
            realised path (§3: one component draw from π and one sample of the stochastic VF per path), coloured by the
            target its end lies nearest. π chooses which field (linear by default) and the fields carry the dynamics. A
            single VF forecasts one path, toward the mean of the four destinations; components of a mixture take the
            branches where the walks split at the doorways (fig. 4a), and stochastic VFs spread the forecasts along each
            branch (figs. 4b–c, 6). Top right: the component posterior π(t) along the pinned forecast (click a forecast
            to pin it): flat under pick and stick, filtered at every grid time under forward filtering (eq. 5). Bottom
            right: where the forecasts end against where the walks went. With walks by day and by night and the time of
            day as an input, the forecasts for night starts favour the landing (fig. 12b); choose which starts to show
            below. Play time t; the slider (or the training figure&apos;s marker) picks the checkpoint.
            {run?.error ? ` Run stopped: ${run.error}` : ''}
          </>
        }
      >
        <Dashboard>
          <DashboardRow ratio={1.7}>
            <DashboardCell>
              <Plot x={fx} y={fy} title={pending ?? `${name}: forecasts at t = ${f3(t)}`} onPlotClick={pinAt}>
                {walks && <Segments segments={walks.segs} width={1} />}
                <Segments segments={geometry.furniture} width={1} />
                <Segments segments={geometry.walls} emphasis width={3} />
                {FLOORPLAN.rooms.map((r) => (
                  <Annotation key={r.name} at={r.at} text={r.name} muted />
                ))}
                {forecasts?.segs.map((s, k) => s.length > 0 && <Segments key={k} segments={s} slot={k} width={1} />)}
                {forecasts && (
                  <Points
                    name="forecasts"
                    x={forecasts.idx.map((b) => forecasts.now[b][0])}
                    y={forecasts.idx.map((b) => forecasts.now[b][1])}
                    group={forecasts.idx.map((b) => forecasts.target[b])}
                    groupNames={TARGETS}
                    size={4}
                  />
                )}
                {forecasts && (
                  <Curve
                    name="pinned forecast"
                    x={forecasts.pinPath.map((p) => p[0])}
                    y={forecasts.pinPath.map((p) => p[1])}
                    emphasis
                  />
                )}
                <Annotation at={FLOORPLAN.origin} text="sofa (start)" />
              </Plot>
            </DashboardCell>
          </DashboardRow>
          <DashboardRow ratio={1}>
            <DashboardCell>
              <Plot x={tAxis} y={piAxis} title={K > 1 ? 'π(t) along the pinned forecast' : 'one VF: π = 1'}>
                {posterior?.map((c, i) => (
                  <Curve key={i} name={`component ${i + 1}`} x={Array.from(run!.gridTimes)} y={c} slot={i} showPoints />
                ))}
                <Annotation x={t} dashed />
              </Plot>
            </DashboardCell>
            <DashboardCell>
              <Plot x={catX} y={shareY} title="where the walks and the forecasts end">
                {walks && <Bars name="walks" x={TARGETS.map((_, i) => i - 0.18)} y={walks.share} width={0.34} muted />}
                {forecasts && (
                  <Bars
                    name="forecasts"
                    x={TARGETS.map((_, i) => i + 0.18)}
                    y={forecasts.share}
                    width={0.34}
                    emphasis
                  />
                )}
              </Plot>
            </DashboardCell>
          </DashboardRow>
        </Dashboard>
        <Select
          label="starts shown"
          value={when}
          onChange={(v) => setWhen(v as typeof when)}
          options={[
            { value: 'all', label: 'all' },
            { value: 'day', label: 'by day' },
            { value: 'night', label: 'by night' },
          ]}
        />
        {slider}
        <Player
          label="time t"
          value={f}
          onChange={setFrame}
          count={Math.max(1, F)}
          format={(i) => `t = ${f3(run?.frameTimes[i] ?? 0)} (${f3(10 * (run?.frameTimes[i] ?? 0))} s)`}
        />
      </Figure>
      <TrainingFigure id="training" run={run} runKey={runKey} marker={marker} lossLabel="objective" />
    </>
  )
}
