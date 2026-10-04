/**
 * Showcase: mixture of experts. A mixture of linear or MLP experts with a softmax, top-k, noisy top-k, Switch,
 * expert-choice or hierarchical gate, trained in the worker by EM (`moeEm`) or Adam with the load-balancing and router
 * z auxiliary losses (`mixtureOfExpertsRun`, over `aifn/nn/experts`), on regime data from `aifn-methods/data`
 * (`piecewiseLinear`, `interleavedFunctions`, `regressionMixture`, `quadrantPlanes`) whose true regimes score the
 * gate. Every number drawn is read from what the run streams (curves, checkpoints) or from `moePredict` at a
 * checkpoint; the lab computes no model maths.
 */
import { useEffect, useMemo, useState } from 'react'
import { datasetRegistry, generate, type RegimeTruth } from 'aifn-methods/data'
import type { RegimeDataset } from 'aifn-methods/data/synthetic'
import {
  emApplies,
  moeModel,
  moePredict,
  type GateChoice,
  type MoeSnapshot,
} from 'aifn-methods/learning/mixture-of-experts'
import { stream } from 'aifn/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { Player, StatusText } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { call, choice, int, row, slider, toggle, useFigureState, useStreamed, type AnyValues, float } from '@lab/state'
import { Button } from '@lab/ui/button'
import { Annotation, Area, Bars, Curve, Plot, Plots, Points, Raster, Readout, useAxis } from '@lab/viz'

// ── Data: registered regime generators, the same call on the page and in the worker ──────────────────────────────

const SETS = {
  piecewise: {
    label: 'piecewise linear (1-D)',
    generator: 'piecewiseLinear',
    knobs: { n: 200, pieces: 3 },
    dims: 1,
    task: 'regression',
  },
  interleaved: {
    label: 'interleaved functions (1-D)',
    generator: 'interleavedFunctions',
    knobs: { n: 240, bands: 4 },
    dims: 1,
    task: 'regression',
  },
  mixture: {
    label: 'overlapping regimes (1-D)',
    generator: 'regressionMixture',
    knobs: { n: 300, regimes: 2, overlap: 1.5 },
    dims: 1,
    task: 'regression',
  },
  quadrants: {
    label: 'a plane per quadrant (2-D)',
    generator: 'quadrantPlanes',
    knobs: { n: 300 },
    dims: 2,
    task: 'regression',
  },
  checkerboard: {
    label: 'a boundary per quadrant (2-D classes)',
    generator: 'quadrantPlanes',
    knobs: { n: 400, task: 'classification' },
    dims: 2,
    task: 'classification',
  },
} as const
type SetName = keyof typeof SETS
const SET_OPTIONS = (Object.keys(SETS) as SetName[]).map((value) => ({ value, label: SETS[value].label }))
const streamKey = (name: SetName) => `showcase-moe/${name}`

function makeData(name: SetName): RegimeDataset {
  const set = SETS[name]
  return generate(datasetRegistry[set.generator], stream(streamKey(name)), set.knobs) as RegimeDataset
}

const GATES: readonly { value: GateChoice; label: string }[] = [
  { value: 'softmax', label: 'softmax (dense)' },
  { value: 'top-k', label: 'top-k' },
  { value: 'noisy-top-k', label: 'noisy top-k' },
  { value: 'switch', label: 'Switch (top-1)' },
  { value: 'expert-choice', label: 'expert choice' },
  { value: 'hierarchical', label: 'hierarchical (2 groups)' },
]
const CAPACITY = [
  { value: 0, label: '∞ (no dropping)' },
  { value: 2, label: '2' },
  { value: 1.25, label: '1.25' },
  { value: 1, label: '1' },
  { value: 0.75, label: '0.75' },
]
const sparse = (g: string) => g === 'top-k' || g === 'noisy-top-k' || g === 'switch' || g === 'expert-choice'
// A `when` closure inside a row sees that row's values.
const gateOf = (v: AnyValues) => String(v.gate)
const methodOf = (v: AnyValues) => String(v.method)

// ── Presets: one click sets the controls and trains ──────────────────────────────────────────────────────────────

const PRESETS = {
  em: {
    label: 'EM on piecewise linear',
    values: {
      'data.dataset': 'piecewise',
      'experts.expert': 'linear',
      'experts.count': 3,
      'experts.objective': 'mixture',
      'gate.gate': 'softmax',
      'gate.temperature': 1,
      'gate.capacity': 0,
      'train.method': 'em',
      'train.emSteps': 40,
      'train.seed': 0,
    },
  },
  balanced: {
    label: 'top-2 with balancing on the checkerboard',
    values: {
      'data.dataset': 'checkerboard',
      'experts.expert': 'linear',
      'experts.count': 4,
      'experts.objective': 'mixture',
      'gate.gate': 'top-k',
      'gate.k': 2,
      'gate.temperature': 1,
      'gate.capacity': 0,
      'train.method': 'adam',
      'train.balance': 0.1,
      'train.z': 0,
      'train.adamSteps': 600,
      'train.stepSize': 0.03,
      'train.seed': 1,
    },
  },
  collapse: {
    label: 'expert collapse (Switch, α = 0)',
    values: {
      'data.dataset': 'quadrants',
      'experts.expert': 'linear',
      'experts.count': 8,
      'experts.objective': 'blend',
      'gate.gate': 'switch',
      'gate.temperature': 1,
      'gate.capacity': 0,
      'train.method': 'adam',
      'train.balance': 0,
      'train.z': 0,
      'train.adamSteps': 400,
      'train.stepSize': 0.03,
      'train.seed': 2,
    },
  },
} as const

/** What a run was trained with: everything the worker task depends on. */
type Setup = {
  dataset: SetName
  expert: 'linear' | 'mlp'
  experts: number
  objective: 'mixture' | 'blend'
  gate: GateChoice
  k: number
  temperature: number
  capacityFactor: number
  method: 'em' | 'adam' | 'lbfgs'
  memory: number
  steps: number
  balance: number
  z: number
  stepSize: number
  seed: number
}
const keyOf = (s: Setup) => JSON.stringify(s)

const fmt = (v: number | undefined, digits = 3) =>
  v === undefined || !Number.isFinite(v) ? '—' : Number(v.toPrecision(digits)).toString()
const GRID_1D = 200
const GRID_2D = 56

/** The page's evaluation grid over the data's box: x [M, d] and its axes. */
function gridOf(truth: RegimeTruth) {
  const { lower, upper } = truth.model
  if (lower.length === 1) {
    const xs = Array.from({ length: GRID_1D }, (_, i) => lower[0] + ((upper[0] - lower[0]) * i) / (GRID_1D - 1))
    return { dims: 1 as const, xs, ys: [] as number[], points: fromData(Float64Array.from(xs), [GRID_1D, 1]) }
  }
  const xs = Array.from({ length: GRID_2D }, (_, j) => lower[0] + ((j + 0.5) * (upper[0] - lower[0])) / GRID_2D)
  const ys = Array.from({ length: GRID_2D }, (_, i) => lower[1] + ((i + 0.5) * (upper[1] - lower[1])) / GRID_2D)
  const points = fromData(Float64Array.from(ys.flatMap((y) => xs.flatMap((x) => [x, y]))), [GRID_2D * GRID_2D, 2])
  return { dims: 2 as const, xs, ys, points }
}

// ── The figure ───────────────────────────────────────────────────────────────────────────────────────────────────

export function MixtureOfExpertsShowcase() {
  const state = useFigureState({
    data: row('1 · data', {
      dataset: choice(SET_OPTIONS, 'piecewise', { label: 'dataset' }),
    }),
    experts: row('2 · experts', {
      expert: choice(
        [
          { value: 'linear', label: 'linear' },
          { value: 'mlp', label: 'MLP (16 tanh units)' },
        ],
        'linear',
        { label: 'expert' },
      ),
      count: slider(2, 8, 3, { label: 'experts N', step: 1 }),
      objective: choice(
        [
          { value: 'mixture', label: 'mixture likelihood (classic)' },
          { value: 'blend', label: 'blended output (sparse layer)' },
        ],
        'mixture',
        { label: 'objective' },
      ),
    }),
    gate: row('3 · gate', {
      gate: choice(GATES, 'softmax', { label: 'gate' }),
      k: slider(1, 4, 2, {
        label: 'experts per token k',
        step: 1,
        when: (v) => gateOf(v) === 'top-k' || gateOf(v) === 'noisy-top-k' || gateOf(v) === 'expert-choice',
      }),
      temperature: slider(0.1, 3, 1, { label: 'temperature τ (small: hard)', step: 0.05 }),
      capacity: choice(CAPACITY, 0, { label: 'capacity factor', when: (v) => sparse(gateOf(v)) }),
    }),
    train: row('4 · training', {
      method: choice(
        [
          { value: 'em', label: 'EM' },
          { value: 'adam', label: 'Adam' },
          { value: 'lbfgs', label: 'L-BFGS (full batch)' },
        ],
        'em',
        { label: 'training' },
      ),
      emSteps: int(40, {
        ge: 1,
        le: 500,
        suggestions: [20, 40, 80],
        label: 'EM iterations',
        when: (v) => methodOf(v) === 'em',
      }),
      adamSteps: int(600, {
        ge: 1,
        suggestions: [300, 600, 1200],
        label: 'steps',
        when: (v) => methodOf(v) !== 'em',
      }),
      memory: int(10, {
        ge: 1,
        le: 50,
        suggestions: [5, 10, 20],
        label: 'L-BFGS memory m',
        when: (v) => methodOf(v) === 'lbfgs',
      }),
      stepSize: float(0.03, {
        gt: 0,
        scale: 'log10',
        suggestions: [0.003, 0.01, 0.03, 0.1],
        label: 'step size',
        when: (v) => methodOf(v) === 'adam',
      }),
      balance: slider(0, 1, 0, {
        label: 'load-balancing weight α',
        step: 0.01,
        when: (v) => methodOf(v) !== 'em',
      }),
      z: slider(0, 0.05, 0, { label: 'router z-loss weight', step: 0.001, when: (v) => methodOf(v) !== 'em' }),
      seed: slider(0, 20, 0, { label: 'seed', step: 1 }),
    }),
    show: row('5 · show', { boundary: toggle(true, 'decision boundaries') }),
  })

  const current: Setup = {
    dataset: state.data.dataset as SetName,
    expert: state.experts.expert as 'linear' | 'mlp',
    experts: state.experts.count,
    objective: state.experts.objective as 'mixture' | 'blend',
    gate: state.gate.gate as GateChoice,
    k: state.gate.k,
    temperature: state.gate.temperature,
    capacityFactor: (state.gate.capacity as number) || Infinity,
    method: state.train.method as Setup['method'],
    memory: state.train.memory,
    steps: state.train.method === 'em' ? state.train.emSteps : state.train.adamSteps,
    balance: state.train.balance,
    z: state.train.z,
    stepSize: state.train.stepSize as number,
    seed: state.train.seed,
  }
  const problem = useMemo(() => {
    if (current.gate === 'hierarchical' && current.experts % 2 !== 0)
      return 'the hierarchical gate splits the experts into two equal groups: choose an even N'
    if (current.method === 'em')
      return emApplies(
        moeModel({
          inputs: SETS[current.dataset].dims,
          task: SETS[current.dataset].task,
          expert: current.expert,
          gate: current.gate,
          objective: current.objective,
          experts: current.gate === 'hierarchical' ? 2 : current.experts,
        }),
      )
    return null
  }, [current.gate, current.experts, current.method, current.dataset, current.expert, current.objective])

  // Nothing trains until Train is pressed; a preset sets the controls, then trains once they have landed.
  const [trained, setTrained] = useState<Setup | null>(null)
  const [pending, setPending] = useState(false)
  const currentKey = keyOf(current)
  useEffect(() => {
    if (!pending) return
    setPending(false)
    setTrained({ ...current })
    // oxlint-disable-next-line react-hooks/exhaustive-deps -- `current` is a fresh object each render; its key is the dependency
  }, [pending, currentKey])
  const stale = trained !== null && keyOf(trained) !== currentKey
  const shownName = trained?.dataset ?? current.dataset
  const data = useMemo(() => makeData(shownName), [shownName])
  const truth = data.meta.truth as RegimeTruth
  const grid = useMemo(() => gridOf(truth), [truth])

  const task = useMemo(() => {
    if (!trained) return null
    const set = SETS[trained.dataset]
    return call<MoeSnapshot>('applied/learning/mixture-of-experts/mixtureOfExpertsRun', {
      data: call(
        `applied/data/synthetic/${set.generator}`,
        call('foundation/random/stream', streamKey(trained.dataset)),
        set.knobs,
      ),
      task: set.task,
      method: trained.method,
      memory: trained.memory,
      steps: trained.steps,
      every: Math.max(1, Math.round(trained.steps / (trained.method === 'em' ? 40 : 60))),
      experts: trained.experts,
      expert: trained.expert,
      objective: trained.objective,
      gate: trained.gate,
      k: trained.k,
      temperature: trained.temperature,
      capacityFactor: trained.capacityFactor,
      groups: 2,
      balance: trained.balance,
      z: trained.z,
      stepSize: trained.stepSize,
      seed: trained.seed,
    })
  }, [trained])
  const run = useStreamed(task)
  const snap = run.value
  const checkpoints = snap?.checkpoints ?? []

  // The player's position belongs to the run it was set on; a new run opens at checkpoint 0.
  const [picked, setPicked] = useState<{ task: typeof task; at: number } | null>(null)
  const at = Math.max(0, Math.min(picked?.task === task ? picked.at : 0, checkpoints.length - 1))
  const setAt = (p: number) => setPicked({ task, at: p })
  const checkpoint = checkpoints[at]

  const specKey = snap ? JSON.stringify(snap.spec) : ''
  const model = useMemo(
    () => (specKey ? moeModel(JSON.parse(specKey, (_k, v) => (v === null ? Infinity : v))) : null),
    [specKey],
  )
  const fit = useMemo(() => {
    if (!model || !checkpoint) return null
    return {
      grid: moePredict(model, checkpoint.params, grid.points),
      train: moePredict(model, checkpoint.params, data.x as Tensor),
    }
  }, [model, checkpoint, grid, data])

  // Curves at the checkpoint: the history entry recorded at its step.
  const h = snap?.history
  const hi = h && checkpoint ? h.step.indexOf(checkpoint.step) : -1
  const N = snap?.spec.experts ?? current.experts
  const expertNames = Array.from({ length: N }, (_, i) => `expert ${i + 1}`)
  const dims = SETS[shownName].dims
  const classification = SETS[shownName].task === 'classification'

  // ── Axes: the data's held per dataset; training curves on the run's step range ──
  const ax = useAxis({ label: dims === 1 ? 'x' : 'x₁', hold: 'initial', key: shownName })
  const ay = useAxis({
    label: dims === 1 ? 'y' : 'x₂',
    hold: 'initial',
    key: shownName,
    ...(dims === 2 ? { equal: ax } : {}),
  })
  const gateY = useAxis({ label: 'gate weight gᵢ(x)', range: [0, 1] })
  const stepAxis = useAxis({
    label: trained?.method === 'em' ? 'EM iteration' : trained?.method === 'lbfgs' ? 'L-BFGS iteration' : 'Adam step',
    range: [0, trained?.steps ?? 1],
    integer: true,
  })
  const lossAxis = useAxis({ label: 'data loss', hold: 'union', key: task })
  const loadAxis = useAxis({ label: 'share of assignments', range: [0, 1] })
  const scoreAxis = useAxis({ label: 'nats · ARI', range: [Math.min(0, -0.1), Math.max(1, Math.log(N))] })
  const expertAxis = useAxis({
    label: 'expert (dense gate: mean weight)',
    categories: expertNames.map((_, i) => `E${i + 1}`),
    key: N,
  })

  const xs = toFlat(data.x as Tensor)
  const x0 = dims === 1 ? xs : xs.filter((_, i) => i % 2 === 0)
  const x1 = dims === 1 ? toFlat(data.y as Tensor) : xs.filter((_, i) => i % 2 === 1)
  const labels = classification ? toFlat(data.y as Tensor) : null

  // ── Panels ──
  const dataPanel =
    dims === 1 ? (
      <Plot x={ax} y={ay} title="data and each expert's fit">
        {fit ? (
          <Points
            name="rows (colour, shape: assigned expert)"
            x={x0}
            y={x1}
            group={fit.train.assignment}
            shape={fit.train.assignment}
            groupNames={expertNames}
            size={5}
          />
        ) : (
          <Points name="rows" x={x0} y={x1} muted size={5} />
        )}
        {fit &&
          expertNames.map((name, i) => {
            const mean = fit.grid.experts.map((r) => r[i])
            const owned = mean.map((v, j) => (fit.grid.gate[j][i] >= 0.5 ? v : NaN))
            return [
              <Curve key={`${i}-all`} name={name} x={grid.xs} y={mean} slot={i} thin dashed />,
              <Curve key={`${i}-own`} name={name} x={grid.xs} y={owned} slot={i} width={3} />,
            ]
          })}
        {fit && <Curve name="E[y | x]" x={grid.xs} y={fit.grid.prediction} emphasis />}
      </Plot>
    ) : (
      <Plot x={ax} y={ay} title="gate partition (colour: top expert)">
        {fit && (
          <Raster
            x={grid.xs}
            y={grid.ys}
            z={rowsOfGrid(fit.grid.assignment)}
            scale="categorical"
            fillOpacity={0.45}
            categoryNames={expertNames}
            boundary={state.show.boundary}
          />
        )}
        <Points
          name="rows"
          x={x0}
          y={x1}
          muted
          size={4}
          shape={labels ?? 0}
          shapeNames={labels ? ['class 0', 'class 1'] : undefined}
        />
      </Plot>
    )

  const gatePanel =
    dims === 1 ? (
      <Plot x={ax} y={gateY} title="gate weights gᵢ(x) (responsibility shading)">
        {fit &&
          expertNames.map((name, i) => (
            <Area key={i} name={name} x={grid.xs} y={fit.grid.gate.map((g) => g[i])} slot={i} opacity={0.25} line />
          ))}
      </Plot>
    ) : (
      <Plot x={ax} y={ay} title={classification ? 'P(y = 1 | x) of the mixture' : 'the mixture’s prediction E[y | x]'}>
        {fit && (
          <Raster
            x={grid.xs}
            y={grid.ys}
            z={rowsOfGrid(fit.grid.prediction)}
            scale="sequential"
            {...(classification ? { range: [0, 1] as const } : {})}
            fillOpacity={0.8}
            valueLabel={classification ? 'P(y = 1)' : 'E[y | x]'}
            boundary={classification && state.show.boundary ? 0.5 : false}
          />
        )}
        {classification ? (
          <Points name="rows" x={x0} y={x1} group={labels} groupNames={['class 0', 'class 1']} size={4} />
        ) : (
          <Points name="rows" x={x0} y={x1} muted size={4} />
        )}
      </Plot>
    )

  const loadNow = h && hi >= 0 ? h.load[hi] : null
  const barsPanel = (
    <Plot x={expertAxis} y={loadAxis} title="load at this checkpoint" legend={false}>
      {loadNow && loadNow.map((v, i) => <Bars key={i} name={expertNames[i]} x={[i]} y={[v]} slot={i} />)}
      <Annotation y={1 / N} text="even share 1/N" dashed />
    </Plot>
  )
  const marker = checkpoint ? <Annotation x={checkpoint.step} dashed /> : null
  const lossPanel = (
    <Plot x={stepAxis} y={lossAxis} title={lossTitle(snap?.spec.objective ?? current.objective, classification)}>
      {h && <Curve name="data loss" x={h.step} y={h.loss} emphasis />}
      {marker}
    </Plot>
  )
  const loadPanel = (
    <Plot x={stepAxis} y={loadAxis} title="load per expert over training">
      {h && expertNames.map((name, i) => <Curve key={i} name={name} x={h.step} y={h.load.map((l) => l[i])} slot={i} />)}
      {marker}
    </Plot>
  )
  const scorePanel = (
    <Plot x={stepAxis} y={scoreAxis} title="router entropy and regime recovery">
      {h && <Curve name="regime agreement (ARI)" x={h.step} y={h.agreement} emphasis />}
      {h && <Curve name="router entropy (nats)" x={h.step} y={h.entropy} muted dashed />}
      <Annotation y={Math.log(N)} text="log N" dashed />
      {marker}
    </Plot>
  )

  const progress = snap ? snap.step / snap.steps : 0
  const status = !trained
    ? problem
      ? `Cannot train: ${problem}.`
      : 'Not trained yet: choose the settings or a preset, then press Train.'
    : run.error
      ? `failed: ${run.error}`
      : stale
        ? `Settings changed since this run: press Retrain${problem ? ` (but ${problem})` : ''}.`
        : run.stopped
          ? `stopped at step ${snap?.step ?? 0} of ${trained.steps}`
          : `${snap?.step ?? 0} / ${trained.steps} ${trained.method === 'em' ? 'EM iterations' : 'steps'}${run.running ? '…' : ''} (${(run.ms / 1000).toFixed(1)} s)`

  return (
    <Figure
      title="Showcase: mixture of experts"
      purpose="A gate splits the input space among experts; each expert fits its own region. EM and gradient descent fit the classic mixture, and without a balancing loss top-1 routing starves most experts."
      state={state}
      defaultSize="XL"
      controls={
        <>
          <ControlRow label="train">
            <div className="flex flex-wrap items-center gap-3">
              {run.running ? (
                <Button size="sm" variant="destructive" aria-label="Stop" onClick={run.stop}>
                  Stop
                </Button>
              ) : (
                <Button
                  size="sm"
                  variant={!trained || stale ? 'default' : 'outline'}
                  aria-label="Train"
                  disabled={problem !== null}
                  onClick={() => setTrained({ ...current })}
                >
                  {trained ? 'Retrain' : 'Train'}
                </Button>
              )}
              {(Object.keys(PRESETS) as (keyof typeof PRESETS)[]).map((key) => (
                <Button
                  key={key}
                  size="sm"
                  variant="outline"
                  aria-label={`preset: ${PRESETS[key].label}`}
                  disabled={run.running}
                  onClick={() => {
                    for (const [path, value] of Object.entries(PRESETS[key].values)) state.set(path, value)
                    setPending(true)
                  }}
                >
                  {PRESETS[key].label}
                </Button>
              ))}
              <div className="h-1.5 w-40 overflow-hidden rounded bg-muted" aria-busy={run.running}>
                <div className="h-full bg-primary" style={{ width: `${100 * Math.min(1, progress)}%` }} />
              </div>
              <StatusText tone={trained && run.error ? 'error' : !trained || stale || problem ? 'attention' : 'muted'}>
                {status}
              </StatusText>
            </div>
          </ControlRow>
          <ControlRow label="checkpoint">
            <Player
              label="checkpoint"
              value={at}
              onChange={setAt}
              count={Math.max(1, checkpoints.length)}
              format={(p) => String(checkpoints[p]?.step ?? 0)}
            />
          </ControlRow>
        </>
      }
      readouts={{
        [`at step ${checkpoint?.step ?? 0}`]: (
          <>
            <Readout label="data loss" value={fmt(h?.loss[hi])} />
            <Readout label="load-balancing loss" value={fmt(h?.balance[hi])} />
            <Readout label="router z-loss" value={fmt(h?.z[hi])} />
            <Readout label="router entropy (nats)" value={fmt(h?.entropy[hi])} />
            <Readout label="idle experts" value={h && hi >= 0 ? `${h.idle[hi]} of ${N}` : '—'} />
            <Readout label="assignments dropped" value={h && hi >= 0 ? `${(100 * h.dropped[hi]).toFixed(1)}%` : '—'} />
            <Readout label="regime agreement (ARI)" value={fmt(h?.agreement[hi])} />
          </>
        ),
        truth: (
          <>
            <Readout label="true regimes" value={truth.regimes} />
            <Readout
              label={classification ? 'Bayes error' : 'Bayes risk (squared error)'}
              value={fmt(truth.bayesRisk)}
            />
          </>
        ),
      }}
      caption={
        <>
          {data.meta.description} Trained in the worker by aifn <code>mixtureOfExpertsRun</code> (
          {trained?.method === 'em' ? (
            <>
              <code>moeEm</code>: E-step responsibilities, weighted least squares or IRLS per expert, an L-BFGS refit of
              the gate
            </>
          ) : (
            <>
              {trained?.method === 'lbfgs' ? 'full-batch L-BFGS (core methodTraining)' : 'full-batch Adam'} on the data
              loss + α·<code>loadBalancingLoss</code> + the z-loss weight·<code>routerZLoss</code>
            </>
          )}
          ) over <code>MixtureOfExperts</code> and <code>route</code> of <code>aifn/nn/experts</code>. Press Train (or a
          preset), then play the checkpoints: the left panels show the experts and the gate at the chosen checkpoint
          (solid where an expert holds at least half the gate weight), the lower panels the whole run with the
          checkpoint marked. ARI compares each row's largest-weight expert with the regime that generated it. With
          decision boundaries on, ink lines mark where the gate's argmax expert changes and, for classification, the P(y
          = 1) = 0.5 contour. Try the collapse preset, then raise α and retrain: the idle experts come back.
        </>
      }
    >
      <Plots cols={3}>
        {dataPanel}
        {gatePanel}
        {barsPanel}
        {lossPanel}
        {loadPanel}
        {scorePanel}
      </Plots>
    </Figure>
  )
}

/** A grid's values in row-major order (y rows of x columns), as the Raster's rows. */
function rowsOfGrid(values: readonly number[]): number[][] {
  return Array.from({ length: GRID_2D }, (_, i) => values.slice(i * GRID_2D, (i + 1) * GRID_2D))
}

function lossTitle(objective: string, classification: boolean): string {
  if (objective === 'mixture') return 'mixture negative log-likelihood (mean)'
  return classification ? 'log-loss of the blend' : 'squared error of the blend'
}
