/**
 * Transfer, continual and meta-learning runs trained in the worker (`aifn-methods/learning/transfer`): domain
 * adaptation of a small network from labelled source to unlabelled target moons; continual learning of a task
 * sequence with naive fine-tuning, EWC or replay; and MAML on sinusoids against a pretrained baseline.
 */
import { useMemo } from 'react'
import type { AdaptationRun, ContinualRun, MamlRun } from 'aifn-methods/learning/transfer'
import { Player } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import { call, choice, float, int, row, slider, useFigureState, type AnyValues, type Task } from 'aifn-render/state'
import { TrainControls, useTrainedRun } from '@lab/views'
import { fmt, usePicked } from './shared'
import { Curve, Handle, Plot, Plots, Points, Raster, Readout, useAxis } from 'aifn-render/viz'

const rowsOf = (v: ArrayLike<number>, g: number) =>
  Array.from({ length: g }, (_, i) => Array.from({ length: g }, (_, j) => v[i * g + j]))
/** The points of class c of a row-major [n × 2] array. */
const classPoints = (x: ArrayLike<number>, y: ArrayLike<number>, c: number) => {
  const idx = Array.from({ length: y.length }, (_, i) => i).filter((i) => y[i] === c)
  return { x: idx.map((i) => x[2 * i]), y: idx.map((i) => x[2 * i + 1]) }
}

// ── 1 · Domain adaptation ───────────────────────────────────────────────────────────────────────────────────────────

const METHODS = [
  { value: 'source-only', label: 'source only' },
  { value: 'mmd', label: 'MMD' },
  { value: 'coral', label: 'CORAL' },
  { value: 'dann', label: 'DANN (gradient reversal)' },
] as const
type DaSettings = { shift: string; amount: number; seed: number; options: Record<string, unknown> }
const daTask = (s: DaSettings): Task<AdaptationRun> =>
  call<AdaptationRun>(
    'applied/learning/transfer/domainAdaptationRun',
    call('applied/data/synthetic/shiftedMoons', call('foundation/random/stream', s.seed), {
      shift: s.shift,
      amount: s.amount,
    }),
    s.options,
  )

export function DomainAdaptationSpecimen() {
  const state = useFigureState({
    data: row('1 · domains', {
      shift: choice(['rotation', 'translation', 'scale'], 'rotation', { label: 'target shift' }),
      amount: slider(0, 1.5, 0.8, { step: 0.05, label: 'amount (radians, units or scale − 1)' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
    model: row('2 · method', {
      method: choice(METHODS, 'dann', { label: 'alignment' }),
      lambda: float(1, { ge: 0, suggestions: [0.1, 1, 10, 100], label: 'λ (DANN 1, MMD 10, CORAL 100 work)' }),
      steps: int(1500, { ge: 1, suggestions: [500, 1500, 3000], label: 'Adam steps' }),
    }),
  })
  const { shift, amount, seed } = state.data
  const { method, lambda, steps } = state.model
  const settings: DaSettings = { shift, amount, seed, options: { method, lambda, steps, seed } }
  const trained = useTrainedRun(settings, daTask)
  const run = trained.run.value
  const shots = run?.checkpoints ?? []
  const [index, pick] = usePicked(trained.trained, shots.length)
  const shot = shots[index]
  const box = 2.5
  const x1 = useAxis({ label: 'x₁', range: [-box, box] })
  const x2 = useAxis({ label: 'x₂', range: [-box, box], equal: x1 })
  const f1 = useAxis({ label: 'feature 1', hold: 'union', key: JSON.stringify(trained.trained) })
  const f2 = useAxis({ label: 'feature 2', hold: 'union', key: JSON.stringify(trained.trained), equal: f1 })
  const stepAxis = useAxis({ label: 'Adam step', range: [0, run?.steps ?? 1], key: run?.steps })
  const accAxis = useAxis({ label: 'target accuracy', range: [0.3, 1] })
  const g = run?.gridX.length ?? 0
  const field = useMemo(() => (shot && g ? rowsOf(shot.field, g) : null), [shot, g])
  const pickStep = (v: number) => {
    let best = 0
    shots.forEach((c, i) => {
      if (Math.abs(c.step - v) < Math.abs(shots[best].step - v)) best = i
    })
    pick(best)
  }
  return (
    <Figure
      title="Domain adaptation: aligning features across a shift"
      purpose="A classifier trained only on the source fails on a shifted target; adding a penalty that makes the target's features look like the source's (a small MMD or CORAL distance, or a discriminator that cannot tell them apart) moves the decision boundary with the data, without any target labels."
      state={state}
      defaultSize="XL"
      controls={
        <>
          <TrainControls
            run={trained as never}
            progress={run ? run.done / run.steps : 0}
            progressText={run ? `${run.done} / ${run.steps} steps` : ''}
          />
          <ControlRow label="3 · checkpoints">
            <Player
              className="col-span-full"
              value={index}
              onChange={pick}
              count={Math.max(1, shots.length)}
              label="checkpoint"
              format={(i) => `step ${shots[i]?.step ?? 0}`}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="step" value={shot ? shot.step : '—'} />
          <Readout label="source accuracy" value={shot ? fmt(shot.sourceAccuracy) : '—'} />
          <Readout label="target accuracy" value={shot ? fmt(shot.targetAccuracy) : '—'} />
        </>
      }
      caption="aifn shiftedMoons (target = source moons rotated, translated or scaled) and domainAdaptationRun: a 2 → 32 → 32 → 2 tanh feature extractor, a linear head and, for DANN, a 2 → 32 → 1 discriminator behind the gradient-reversal layer; Adam on 64 rows of each domain per step. Target labels only score the run. Left: P(class 1) of the classifier with its ½ boundary, source points filled (slots 0, 1) and target points small (the same slots). Middle: the 2-d features of both domains (target faded). Right: target accuracy over training. Play the checkpoints or drag the step marker."
    >
      <Plots cols={3}>
        <Plot x={x1} y={x2} title={!trained.trained ? 'press Train to start' : 'inputs and the decision boundary'}>
          {field && run && (
            <Raster
              x={run.gridX}
              y={run.gridX}
              z={field}
              scale="diverging"
              range={[0, 1]}
              valueLabel="P(class 1)"
              fillOpacity={0.45}
              boundary={0.5}
            />
          )}
          {run &&
            [0, 1].map((c) => (
              <Points
                key={`s${c}`}
                name={`source ${c}`}
                {...classPoints(run.source.x, run.source.y, c)}
                slot={c}
                size={5}
              />
            ))}
          {run &&
            [0, 1].map((c) => (
              <Points
                key={`t${c}`}
                name={`target ${c}`}
                {...classPoints(run.target.x, run.target.y, c)}
                slot={c}
                thin
                size={3}
                shape={2}
              />
            ))}
        </Plot>
        <Plot x={f1} y={f2} title="features">
          {shot &&
            run &&
            [0, 1].map((c) => (
              <Points
                key={`fs${c}`}
                name={`source ${c}`}
                {...classPoints(shot.sourceFeatures, run.source.y, c)}
                slot={c}
                size={5}
              />
            ))}
          {shot &&
            run &&
            [0, 1].map((c) => (
              <Points
                key={`ft${c}`}
                name={`target ${c}`}
                {...classPoints(shot.targetFeatures, run.target.y, c)}
                slot={c}
                thin
                size={3}
                shape={2}
              />
            ))}
        </Plot>
        <Plot x={stepAxis} y={accAxis} title="target accuracy">
          {run && <Curve name="target accuracy" slot={2} x={run.history.step} y={run.history.targetAccuracy} />}
          {shot && <Handle kind="x" at={shot.step} onDrag={pickStep} label={`step ${shot.step}`} />}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 2 · Continual learning ──────────────────────────────────────────────────────────────────────────────────────────

type ClSettings = { tasks: number; seed: number; options: Record<string, unknown> }
const clTask = (s: ClSettings): Task<ContinualRun> =>
  call<ContinualRun>(
    'applied/learning/transfer/continualRun',
    call('applied/data/synthetic/rotatingTasks', call('foundation/random/stream', s.seed), { tasks: s.tasks }),
    s.options,
  )
const methodOf = (v: AnyValues) => String(v.method)

export function ContinualSpecimen() {
  const state = useFigureState({
    setup: row('1 · tasks and strategy', {
      tasks: int(3, { ge: 2, le: 6, suggestions: [2, 3, 4], label: 'tasks' }),
      method: choice(
        [
          { value: 'naive', label: 'naive fine-tuning' },
          { value: 'ewc', label: 'EWC' },
          { value: 'replay', label: 'experience replay' },
        ],
        'ewc',
        { label: 'strategy' },
      ),
      lambda: float(1000, {
        ge: 0,
        scale: 'log10',
        suggestions: [100, 1000, 10_000],
        label: 'EWC λ',
        when: (v) => methodOf(v) === 'ewc',
      }),
      memory: int(20, {
        ge: 1,
        le: 300,
        suggestions: [5, 20, 50],
        label: 'memory per task',
        when: (v) => methodOf(v) === 'replay',
      }),
      seed: int(2, { ge: 0, le: 9999, label: 'seed' }),
    }),
  })
  const { tasks, method, lambda, memory, seed } = state.setup
  const settings: ClSettings = { tasks, seed, options: { method, lambda, memory, seed } }
  const trained = useTrainedRun(settings, clTask)
  const run = trained.run.value
  const fields = run?.fields ?? []
  const [index, pick] = usePicked(trained.trained, fields.length)
  const stepAxis = useAxis({
    label: 'training step (tasks in turn)',
    range: [0, (trained.trained?.tasks ?? tasks) * 400],
    key: trained.trained?.tasks,
  })
  const accAxis = useAxis({ label: 'accuracy', range: [0.3, 1] })
  const box = 4
  const x1 = useAxis({ label: 'x₁', range: [-box, box] })
  const x2 = useAxis({ label: 'x₂', range: [-box, box], equal: x1 })
  const g = run?.gridX.length ?? 0
  return (
    <Figure
      title="Continual learning: forgetting and its remedies"
      purpose="Trained on one task after another, a network overwrites what made the earlier tasks work (catastrophic forgetting); EWC anchors the weights the old tasks relied on, as measured by their Fisher information, and replay keeps a few old examples in every batch."
      state={state}
      defaultSize="L"
      controls={
        <>
          <TrainControls
            run={trained as never}
            progress={run ? fields.length / run.tasks : 0}
            progressText={run ? `${fields.length} / ${run.tasks} tasks` : ''}
          />
          <ControlRow label="2 · after task">
            <Player
              className="col-span-full"
              value={index}
              onChange={pick}
              count={Math.max(1, fields.length)}
              label="task"
              format={(i) => `after task ${i + 1}`}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          {run?.accuracy.map((a, k) => (
            <Readout key={k} label={`task ${k + 1} at the end`} value={fmt(a.at(-1)!)} />
          ))}
        </>
      }
      caption="aifn rotatingTasks (two Gaussian classes per task, each task in its own region with its class axis turned by 90°) and continualRun (a 2 → 32 → 32 → 2 tanh network, 400 Adam steps per task, batches of 32; EWC's Fisher from 100 examples per task; replay mixes the memory into every batch). Left: the accuracy on every task (one colour per task) as training moves through them. Right: P(class 1) after the chosen task over all the task regions; earlier regions keep their boundary only under EWC or replay."
    >
      <Plots cols={2}>
        <Plot x={stepAxis} y={accAxis} title={!trained.trained ? 'press Train to start' : 'accuracy on each task'}>
          {run?.accuracy.map((a, k) => (
            <Curve key={k} name={`task ${k + 1}`} slot={k} x={run.steps} y={a} />
          ))}
        </Plot>
        <Plot x={x1} y={x2} title={fields.length ? `after task ${index + 1}` : 'decision field'}>
          {fields[index] && run && (
            <Raster
              x={run.gridX}
              y={run.gridX}
              z={rowsOf(fields[index], g)}
              scale="diverging"
              range={[0, 1]}
              valueLabel="P(class 1)"
              fillOpacity={0.5}
              boundary={0.5}
            />
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 3 · MAML ────────────────────────────────────────────────────────────────────────────────────────────────────────

const mamlTask = (options: Record<string, unknown>): Task<MamlRun> =>
  call<MamlRun>('applied/learning/transfer/mamlRun', options)

export function MamlSpecimen() {
  const state = useFigureState({
    setup: row('1 · meta-learning', {
      order: choice(
        [
          { value: 'second', label: 'MAML (second order)' },
          { value: 'first', label: 'first-order MAML' },
        ],
        'second',
        { label: 'variant' },
      ),
      innerRate: float(0.01, { gt: 0, scale: 'log10', suggestions: [0.001, 0.01, 0.05], label: 'inner step size α' }),
      steps: int(2000, { ge: 1, suggestions: [500, 2000, 5000], label: 'meta-updates' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
  })
  const options = { ...state.setup }
  const trained = useTrainedRun(options, mamlTask)
  const run = trained.run.value
  const shots = run?.checkpoints ?? []
  const [index, pick] = usePicked(trained.trained, shots.length)
  const shot = shots[index]
  const xAxis = useAxis({ label: 'x', range: [-5, 5] })
  const yAxis = useAxis({ label: 'y', range: [-5.5, 5.5] })
  const kAxis = useAxis({ label: 'adaptation steps on 10 points', range: [0, 10] })
  const mseAxis = useAxis({ label: 'query MSE (20 new tasks)', range: [0, 6] })
  const ks = Array.from({ length: 11 }, (_, k) => k)
  const clip = (a: ArrayLike<number>) => Array.from(a, (v) => Math.max(-5.5, Math.min(5.5, v)))
  return (
    <Figure
      title="MAML: an initialisation built for fine-tuning"
      purpose="MAML trains the starting weights so that one gradient step on a handful of points of a new task already fits it; a network pretrained on all tasks at once learns their average instead, and a few steps from there barely help."
      state={state}
      defaultSize="L"
      controls={
        <>
          <TrainControls
            run={trained as never}
            progress={run ? run.done / run.steps : 0}
            progressText={run ? `${run.done} / ${run.steps} meta-updates` : ''}
          />
          <ControlRow label="2 · checkpoints">
            <Player
              className="col-span-full"
              value={index}
              onChange={pick}
              count={Math.max(1, shots.length)}
              label="checkpoint"
              format={(i) => `update ${shots[i]?.step ?? 0}`}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="meta-update" value={shot ? shot.step : '—'} />
          <Readout
            label="MAML MSE after 1 / 10 steps"
            value={shot ? `${fmt(shot.mamlCurve[1])} / ${fmt(shot.mamlCurve[10])}` : '—'}
          />
          <Readout
            label="pretrained after 1 / 10"
            value={shot ? `${fmt(shot.pretrainedCurve[1])} / ${fmt(shot.pretrainedCurve[10])}` : '—'}
          />
          <Readout
            label="test task"
            value={run ? `A = ${fmt(run.testTask.amplitude, 2)}, φ = ${fmt(run.testTask.phase, 2)}` : '—'}
          />
        </>
      }
      caption="aifn mamlRun: a 1 → 40 → 40 → 1 ReLU network (inputs scaled to [−1, 1]); each meta-update draws 8 tasks y = A sin(x − φ), A ∈ [0.1, 5], φ ∈ [0, π], with 10 support and 10 query points, adapts by one inner step and updates the initialisation by Adam on the query loss (through the inner gradient for second order). The baseline is pretrained by Adam on the same points as one regression. Left: an unseen task (dashed), its 10 support points, and the fits from MAML's initialisation after 0, 1 and 10 steps of size α (slot 0, darkening), and the pretrained network's after 10 (slot 1). Right: query MSE on 20 unseen tasks against adaptation steps."
    >
      <Plots cols={2}>
        <Plot x={xAxis} y={yAxis} title={!trained.trained ? 'press Train to start' : 'an unseen task'}>
          {run && <Curve name="truth" x={run.grid} y={run.truth} emphasis dashed />}
          {run && (
            <Points name="support" x={Array.from(run.support.x)} y={Array.from(run.support.y)} emphasis size={7} />
          )}
          {shot && <Curve name="MAML, 0 steps" slot={0} x={run!.grid} y={clip(shot.maml[0])} thin dashed />}
          {shot && <Curve name="MAML, 1 step" slot={0} x={run!.grid} y={clip(shot.maml[1])} thin />}
          {shot && <Curve name="MAML, 10 steps" slot={0} x={run!.grid} y={clip(shot.maml[2])} />}
          {shot && <Curve name="pretrained, 10 steps" slot={1} x={run!.grid} y={clip(shot.pretrained[2])} />}
        </Plot>
        <Plot x={kAxis} y={mseAxis} title="few-shot adaptation">
          {shot && (
            <Curve name="MAML" slot={0} x={ks} y={Array.from(shot.mamlCurve, (v) => Math.min(6, v))} showPoints />
          )}
          {shot && (
            <Curve
              name="pretrained"
              slot={1}
              x={ks}
              y={Array.from(shot.pretrainedCurve, (v) => Math.min(6, v))}
              showPoints
            />
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}
