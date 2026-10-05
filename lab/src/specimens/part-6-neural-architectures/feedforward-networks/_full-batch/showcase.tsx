/**
 * Showcase: a small MLP trained by full-batch L-BFGS against gradient descent, Adam and SGD from the same initial
 * weights. The worker runs aifn-methods `fullBatchComparison` (core `fullBatchTraining` for the full-batch methods,
 * `trainingLoop` for the minibatch ones); the page draws the fit at any recorded iteration from the checkpointed θ.
 */
import { useMemo, useState } from 'react'
import {
  COMPARISON_OPTIMISERS,
  comparisonModel,
  type ComparisonOptimiser,
  type ComparisonOptions,
  type ComparisonRun,
  type ComparisonSnapshot,
  type ComparisonTask,
} from 'aifn-methods/neural/full-batch'
import { fromData, toFlat, unwrap, type Tensor } from 'aifn-compute/foundation/tensor'
import { sigmoid } from 'aifn-compute/numerics/special'
import { Select, Slider } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import { call, choice, float, int, row, setting, toggle, useFigureState, type Task } from 'aifn-render/state'
import {
  CLASSIFICATION_CASES,
  datasetChoice,
  formatValue,
  REGRESSION_CASES,
  TrainControls,
  useTrainedRun,
  type DatasetValue,
} from '@lab/views'
import { Bars, Curve, Plot, Plots, Points, Raster, Readout, useAxis } from 'aifn-render/viz'

const f3 = (v: number | undefined) =>
  v !== undefined && Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—'

const LABELS: Record<ComparisonOptimiser, string> = {
  lbfgs: 'L-BFGS (full batch)',
  'gradient-descent': 'gradient descent (full batch)',
  adam: 'Adam (minibatch)',
  sgd: 'SGD (minibatch)',
}
/** Fixed colour slots by optimiser, whichever are trained. */
const SLOT: Record<ComparisonOptimiser, number> = { lbfgs: 0, 'gradient-descent': 1, adam: 2, sgd: 3 }

/** Every 2-d classification set and 1-d regression curve, from `aifn-methods/data`'s registered generators. */
const DATA = datasetChoice({ ...CLASSIFICATION_CASES, ...REGRESSION_CASES })

type Settings = {
  data: DatasetValue
  width: number
  depth: number
  activation: 'tanh' | 'gelu' | 'relu'
  optimiser: ComparisonOptimiser
  compareAll: boolean
  memory: number
  tolerance: number
  gdStep: number
  adamStep: number
  sgdStep: number
  batchSize: number
  iterations: number
  l2: number
  seed: number
}

/** The data's stream name: one per dataset choice, so the page and the worker draw the same points. */
const dataSeed = (d: DatasetValue) => `full-batch-data-${d.key}`

function comparisonTask(s: Settings): Task<ComparisonSnapshot> {
  const data = DATA.task(s.data, dataSeed(s.data))
  const options: ComparisonOptions = {
    task: DATA.taskOf(s.data) as ComparisonTask,
    network: { width: s.width, depth: s.depth, activation: s.activation },
    optimisers: s.compareAll ? COMPARISON_OPTIMISERS : [s.optimiser],
    iterations: s.iterations,
    memory: s.memory,
    tolerance: s.tolerance,
    gdStep: s.gdStep,
    adamStep: s.adamStep,
    sgdStep: s.sgdStep,
    batchSize: s.batchSize,
    l2: s.l2,
    seed: s.seed,
  }
  return call<ComparisonSnapshot>('applied/neural/full-batch/fullBatchComparison', data, options)
}

/** Presets set the controls; Train runs them. */
const PRESETS: Record<string, { label: string; values: Record<string, string | number | boolean> }> = {
  smooth: {
    label: 'tanh on moons: L-BFGS shines',
    values: { data: 'moons', 'network.activation': 'tanh' },
  },
  kinks: {
    label: 'ReLU on moons: kinks hurt the line search',
    values: { data: 'moons', 'network.activation': 'relu' },
  },
  spirals: {
    label: 'tanh on two spirals (16 × 2)',
    values: {
      data: 'spirals',
      'network.activation': 'tanh',
      'network.width': 16,
      'network.depth': 2,
    },
  },
  sine: {
    label: 'ReLU regression of sin x',
    values: { data: 'fn:sine', 'network.activation': 'relu' },
  },
}

const GRID = 60

/** Evenly spaced points spanning [lo, hi] padded by 10%. */
function spaced(values: ArrayLike<number>, count: number): number[] {
  let lo = Infinity
  let hi = -Infinity
  for (let i = 0; i < values.length; i++) {
    lo = Math.min(lo, values[i])
    hi = Math.max(hi, values[i])
  }
  const pad = 0.1 * (hi - lo || 1)
  return Array.from({ length: count }, (_, i) => lo - pad + ((hi - lo + 2 * pad) * i) / (count - 1))
}

export function FullBatchShowcase() {
  const state = useFigureState({
    data: DATA.field({ label: '1 · data (classification sets, then regression curves)', initial: 'moons' }),
    network: row('2 · network', {
      width: int(8, { ge: 1, le: 64, suggestions: [4, 8, 16, 32], label: 'hidden units' }),
      depth: int(2, { ge: 1, le: 4, suggestions: [1, 2, 3], label: 'hidden layers' }),
      activation: choice(['tanh', 'gelu', 'relu'], 'tanh', { label: 'activation' }),
    }),
    optimiser: row('3 · optimiser', {
      optimiser: choice(
        COMPARISON_OPTIMISERS.map((value) => ({ value, label: LABELS[value] })),
        'lbfgs',
        { label: 'optimiser' },
      ),
      compareAll: setting(true, { label: 'compare all (same initial weights)' }),
      memory: int(10, {
        ge: 1,
        le: 50,
        suggestions: [3, 5, 10, 20],
        label: 'L-BFGS memory m',
        when: (v) => v.compareAll === true || v.optimiser === 'lbfgs',
      }),
      tolerance: float(1e-6, { gt: 0, scale: 'log10', suggestions: [1e-4, 1e-6, 1e-8], label: 'tolerance ‖∇‖' }),
      gdStep: float(0.3, {
        gt: 0,
        scale: 'log10',
        suggestions: [0.1, 0.3, 1],
        label: 'gradient-descent η',
        when: (v) => v.compareAll === true || v.optimiser === 'gradient-descent',
      }),
      adamStep: float(0.01, {
        gt: 0,
        scale: 'log10',
        suggestions: [0.003, 0.01, 0.03],
        label: 'Adam η',
        when: (v) => v.compareAll === true || v.optimiser === 'adam',
      }),
      sgdStep: float(0.1, {
        gt: 0,
        scale: 'log10',
        suggestions: [0.03, 0.1, 0.3],
        label: 'SGD η',
        when: (v) => v.compareAll === true || v.optimiser === 'sgd',
      }),
      batchSize: int(32, {
        ge: 1,
        suggestions: [8, 16, 32, 64],
        label: 'minibatch size',
        when: (v) => v.compareAll === true || v.optimiser === 'adam' || v.optimiser === 'sgd',
      }),
    }),
    run: row('4 · objective and run', {
      l2: float(1e-3, { ge: 0, step: 1e-4, suggestions: [0, 1e-4, 1e-3, 1e-2], label: 'L2 strength λ' }),
      iterations: int(300, { ge: 1, suggestions: [100, 200, 300, 500], label: 'iterations (each)' }),
      seed: int(0, { label: 'initialisation seed', ge: 0, le: 9999 }),
    }),
    show: row('5 · show', { boundary: toggle(true, 'decision boundary (P = 0.5)') }),
  })
  const settings: Settings = {
    data: { key: state.data.key, values: { ...state.data.values } },
    width: Number(state.network.width),
    depth: Number(state.network.depth),
    activation: state.network.activation as Settings['activation'],
    optimiser: state.optimiser.optimiser as ComparisonOptimiser,
    compareAll: Boolean(state.optimiser.compareAll),
    memory: Number(state.optimiser.memory),
    tolerance: Number(state.optimiser.tolerance),
    gdStep: Number(state.optimiser.gdStep),
    adamStep: Number(state.optimiser.adamStep),
    sgdStep: Number(state.optimiser.sgdStep),
    batchSize: Number(state.optimiser.batchSize),
    iterations: Number(state.run.iterations),
    l2: Number(state.run.l2),
    seed: state.run.seed,
  }
  const trained = useTrainedRun(settings, comparisonTask)
  const result = trained.run.value
  const shown = trained.trained ?? settings
  const shownKey = DATA.key(shown.data)
  // oxlint-disable-next-line react-hooks/exhaustive-deps -- the key changes exactly when the dataset does
  const data = useMemo(() => DATA.make(shown.data, dataSeed(shown.data)), [shownKey])
  const task = DATA.taskOf(shown.data)
  const runs = useMemo(() => result?.runs ?? [], [result])

  // The fit is shown for one optimiser: the chosen one, or a pick among those trained.
  const [pickedOptimiser, setPickedOptimiser] = useState<{ run: unknown; value: ComparisonOptimiser } | null>(null)
  const fitOptimiser =
    pickedOptimiser && pickedOptimiser.run === trained.trained ? pickedOptimiser.value : shown.optimiser
  const fitRun: ComparisonRun | undefined = runs.find((r) => r.optimiser === fitOptimiser) ?? runs[0]
  const shots = fitRun?.checkpoints ?? []
  // The iteration slider opens at the last checkpoint (the trained fit) and follows the run while it trains.
  const [picked, setPicked] = useState<{ run: unknown; optimiser: string; index: number } | null>(null)
  const pickedHere = picked && picked.run === trained.trained && picked.optimiser === fitRun?.optimiser
  const index = Math.min(pickedHere ? picked.index : shots.length - 1, Math.max(0, shots.length - 1))
  const shot = shots[index]
  const at = shot ? fitRun!.iteration.indexOf(shot.iteration) : -1

  const network = result?.network
  const model = useMemo(() => (network ? comparisonModel(network) : null), [network])
  const xs = useMemo(() => toFlat(data.x), [data])
  const labels = useMemo(() => toFlat(data.y!), [data])
  const isClass = (result?.task ?? task) === 'classification'
  const axes = useMemo(() => {
    if (!isClass) return { gx: spaced(xs, 200), gy: [] as number[] }
    return {
      gx: spaced(
        xs.filter((_, i) => i % 2 === 0),
        GRID,
      ),
      gy: spaced(
        xs.filter((_, i) => i % 2 === 1),
        GRID,
      ),
    }
  }, [xs, isClass])
  const gridPoints = useMemo(() => {
    const { gx, gy } = axes
    if (!isClass) return fromData(Float64Array.from(gx), [gx.length, 1])
    const flat = new Float64Array(2 * GRID * GRID)
    gy.forEach((y, i) =>
      gx.forEach((x, j) => {
        flat[2 * (i * GRID + j)] = x
        flat[2 * (i * GRID + j) + 1] = y
      }),
    )
    return fromData(flat, [GRID * GRID, 2])
  }, [axes, isClass])
  const fit = useMemo(() => {
    if (!model || !shot) return null
    const params = model.unravel(shot.theta)
    const out = unwrap(model.model.apply(params, gridPoints)) as Tensor
    if (!isClass) return { curve: toFlat(out), field: null }
    const p = toFlat(unwrap(sigmoid(out)) as Tensor)
    return { curve: null, field: axes.gy.map((_, i) => p.slice(i * GRID, (i + 1) * GRID)) }
  }, [model, shot, gridPoints, isClass, axes])
  const score = useMemo(() => {
    if (!model || !shot) return NaN
    const out = toFlat(unwrap(model.model.apply(model.unravel(shot.theta), data.x)) as Tensor)
    if (isClass) return out.reduce((a, z, i) => a + ((z > 0 ? 1 : 0) === labels[i] ? 1 : 0), 0) / out.length
    return Math.sqrt(out.reduce((a, v, i) => a + (v - labels[i]) ** 2, 0) / out.length)
  }, [model, shot, data, labels, isClass])
  const points = useMemo(
    () =>
      isClass
        ? {
            x: xs.filter((_, i) => i % 2 === 0),
            y: xs.filter((_, i) => i % 2 === 1),
            group: Array.from(labels),
          }
        : { x: Array.from(xs), y: Array.from(labels), group: null },
    [xs, labels, isClass],
  )

  const runKey = trained.trained
  const x1 = useAxis({ label: isClass ? 'x₁' : 'x', key: runKey })
  const x2 = useAxis({ label: isClass ? 'x₂' : 'y', equal: isClass ? x1 : undefined, key: runKey })
  const iterAxis = useAxis({ label: 'iteration', key: runKey, hold: 'union', integer: true })
  const evalAxis = useAxis({ label: 'full-data gradient evaluations', key: runKey, hold: 'union' })
  const lossAxis = useAxis({ label: 'training objective', log: true, hold: 'union', key: runKey })
  const kAxis = useAxis({ label: 'iteration', key: runKey, hold: 'union', integer: true })
  const alphaAxis = useAxis({ label: 'step length α', log: true, hold: 'union', key: runKey })
  const evalsAxis = useAxis({ label: 'evaluations', range: [0, undefined], hold: 'union', key: runKey, integer: true })
  const gradAxis = useAxis({ label: '‖∇f‖', log: true, hold: 'union', key: runKey })
  const syAxis = useAxis({ label: 'sᵀy', log: true, hold: 'union', key: runKey })

  const [preset, setPreset] = useState('')
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
  const done = result?.done ?? 0
  const total = result?.total ?? settings.iterations * (settings.compareAll ? COMPARISON_OPTIMISERS.length : 1)
  const viewControls =
    runs.length > 0 ? (
      <ControlRow label="6 · view">
        <div className="flex flex-wrap items-end gap-3">
          {runs.length > 1 && (
            <Select
              label="fit shown for"
              value={fitRun?.optimiser ?? ''}
              onChange={(v) => setPickedOptimiser({ run: trained.trained, value: v as ComparisonOptimiser })}
              options={runs.map((r) => ({ value: r.optimiser, label: LABELS[r.optimiser] }))}
            />
          )}
          {shots.length > 1 && (
            <Slider
              label="iteration"
              value={index}
              onChange={(i) => setPicked({ run: trained.trained, optimiser: fitRun?.optimiser ?? '', index: i })}
              min={0}
              max={shots.length - 1}
              step={1}
              steppable
              withArrows
              format={(k) => `${shots[k]?.iteration ?? 0}`}
            />
          )}
        </div>
      </ControlRow>
    ) : null
  const lbfgs = runs.find((r) => r.optimiser === 'lbfgs')
  const lbfgsIters = useMemo(
    () => (lbfgs ? Array.from({ length: lbfgs.stepSize.length }, (_, k) => k + 1) : []),
    [lbfgs],
  )
  const kept = useMemo(() => {
    if (!lbfgs) return { x: [] as number[], y: [] as number[], skipX: [] as number[], skipY: [] as number[] }
    const out = { x: [] as number[], y: [] as number[], skipX: [] as number[], skipY: [] as number[] }
    lbfgs.curvature.forEach((sy, k) => {
      if (lbfgs.skipped[k] || !(sy > 0)) {
        out.skipX.push(k + 1)
        out.skipY.push(Math.max(Math.abs(sy), 1e-12))
      } else {
        out.x.push(k + 1)
        out.y.push(sy)
      }
    })
    return out
  }, [lbfgs])
  const meanEvals = lbfgs?.lineEvaluations.length
    ? lbfgs.lineEvaluations.reduce((a, b) => a + b, 0) / lbfgs.lineEvaluations.length
    : NaN
  const waiting = (
    <div className="py-6 text-center text-sm text-muted-foreground">
      {trained.trained ? 'Training…' : 'Press Train: this figure fills in as the run streams.'}
    </div>
  )
  const lossAt = fitRun && at >= 0 ? fitRun.loss[at] : undefined
  const gradAt = fitRun && at >= 0 ? fitRun.gradNorm[at] : undefined

  return (
    <>
      <Figure
        title="The fit"
        id="the-fit"
        purpose="A small MLP fitted by full-batch L-BFGS or a first-order method: the decision regions (or the regression curve) at any recorded iteration of the chosen optimiser."
        state={state}
        defaultSize="L"
        controls={
          <>
            {presets}
            <TrainControls
              run={trained as never}
              progress={total ? done / total : 0}
              progressText={`${done} / ${total} iterations`}
            />
            {viewControls}
          </>
        }
        readouts={
          <>
            <Readout label="optimiser" value={fitRun ? LABELS[fitRun.optimiser] : '—'} />
            <Readout label="iteration" value={shot ? shot.iteration : '—'} />
            <Readout label="training objective" value={f3(lossAt)} />
            <Readout
              label={isClass ? 'training accuracy' : 'training RMSE'}
              value={isClass ? (Number.isFinite(score) ? `${Math.round(100 * score)}%` : '—') : f3(score)}
            />
            <Readout label="gradient norm" value={f3(gradAt)} />
            <Readout label="wall time (steps)" value={fitRun ? `${Math.round(fitRun.ms)} ms` : '—'} />
            <Readout label="parameters" value={result ? result.parameterCount : '—'} />
            <Readout label="stopped" value={fitRun?.stop ?? '—'} />
          </>
        }
        caption={
          <>
            aifn-methods <code>fullBatchComparison</code> trains an MLP ({shown.depth} hidden{' '}
            {shown.depth === 1 ? 'layer' : 'layers'} of {shown.width} {shown.activation} units, one output) in the
            worker on {data.x.shape[0]} points, minimising the mean{' '}
            {isClass ? 'binary cross-entropy of one logit' : 'squared error'} plus (λ/2)‖W‖² over the weight matrices.
            L-BFGS and gradient descent go through core <code>fullBatchTraining</code>: the parameter tree is raveled to
            one vector θ and every evaluation uses the whole set. Adam and SGD use minibatches through{' '}
            <code>trainingLoop</code>. With compare all, every optimiser starts from the same initial weights (seed{' '}
            {shown.seed}).{' '}
            {isClass
              ? 'The field is P(class 1) on a diverging scale, pale at 0.5; with the decision boundary on, the ink line is its 0.5 contour; points keep their class colours.'
              : 'The curve is the network’s prediction over the inputs; points are the training data.'}{' '}
            Step through the recorded iterations with the slider (it opens at the last); pick another trained optimiser
            to compare fits.
          </>
        }
      >
        <Plot
          x={x1}
          y={x2}
          title={fitRun ? `${LABELS[fitRun.optimiser]}, iteration ${shot?.iteration ?? 0}` : 'the fit'}
        >
          {isClass && fit?.field && (
            <Raster
              x={axes.gx}
              y={axes.gy}
              z={fit.field}
              scale="diverging"
              range={[0, 1]}
              valueLabel="P(class 1)"
              fillOpacity={0.7}
              boundary={state.show.boundary ? 0.5 : false}
            />
          )}
          {isClass ? (
            <Points
              name="data"
              x={points.x}
              y={points.y}
              group={points.group}
              groupNames={['class 0', 'class 1']}
              thin
            />
          ) : (
            <Points name="data" x={points.x} y={points.y} muted />
          )}
          {!isClass && fit?.curve && <Curve name="fit" x={axes.gx} y={fit.curve} emphasis />}
        </Plot>
      </Figure>
      <Figure
        title="Convergence"
        id="convergence"
        purpose="The training objective against iterations and against full-data gradient evaluations, one line per optimiser from the same start: an iteration of L-BFGS costs a line search, a minibatch step only its share of the data."
        defaultSize="XL"
        readouts={
          <>
            {runs.map((r) => (
              <Readout
                key={r.optimiser}
                label={LABELS[r.optimiser]}
                value={`${f3(r.loss.at(-1))} after ${f3(r.evaluations.at(-1))} evals`}
              />
            ))}
          </>
        }
        caption={
          <>
            Log scale. The work axis counts full-data gradient evaluations: one per gradient-descent step, the number of
            loss-and-gradient evaluations of each L-BFGS line search (often 1, more where the search must backtrack or
            zoom), and b/n per minibatch step of size b on n points. The second plot is the honest comparison: an L-BFGS
            iteration is worth several cheap steps, and minibatch methods finish their iterations having touched the
            data only a few dozen times. The minibatch methods&apos; objective is evaluated on the full set at each
            record, outside their timed steps. Colours are fixed by optimiser.
          </>
        }
      >
        {runs.length ? (
          <Plots cols={2} scale={0.6}>
            <Plot x={iterAxis} y={lossAxis} title="objective by iteration">
              {runs.map((r) => (
                <Curve
                  key={r.optimiser}
                  name={LABELS[r.optimiser]}
                  x={r.iteration}
                  y={r.loss}
                  slot={SLOT[r.optimiser]}
                />
              ))}
            </Plot>
            <Plot x={evalAxis} y={lossAxis} title="objective by gradient evaluations">
              {runs.map((r) => (
                <Curve
                  key={r.optimiser}
                  name={LABELS[r.optimiser]}
                  x={r.evaluations}
                  y={r.loss}
                  slot={SLOT[r.optimiser]}
                />
              ))}
            </Plot>
          </Plots>
        ) : (
          waiting
        )}
      </Figure>
      <Figure
        title="Inside L-BFGS"
        id="lbfgs-internals"
        purpose="What each L-BFGS iteration did: the step length its strong Wolfe line search accepted, how many loss-and-gradient evaluations that search took, the gradient norm, and the curvature sᵀy of the pair it stored."
        defaultSize="XL"
        readouts={
          <>
            <Readout label="iterations" value={lbfgs ? lbfgs.stepSize.length : '—'} />
            <Readout label="evaluations per iteration" value={f3(meanEvals)} />
            <Readout
              label="α = 1 accepted"
              value={
                lbfgs?.stepSize.length
                  ? `${Math.round((100 * lbfgs.stepSize.filter((a) => a === 1).length) / lbfgs.stepSize.length)}%`
                  : '—'
              }
            />
            <Readout label="skipped pairs" value={lbfgs ? lbfgs.skipped.filter(Boolean).length : '—'} />
            <Readout label="stopped" value={lbfgs?.stop ?? '—'} />
          </>
        }
        caption={
          <>
            L-BFGS (memory m = {shown.memory}) builds its direction from the last m pairs s = θₖ₊₁ − θₖ, y = ∇fₖ₊₁ − ∇fₖ
            and stores a pair only when sᵀy &gt; 0 (skipped pairs, drawn in red at |sᵀy|, are left out). That test and
            the line search both compare values and gradients of one deterministic function, which is why L-BFGS needs
            full batches: with minibatches, y would mix a change of position with a change of sample, its curvature
            pairs would be noise, and a step judged on one batch would be tested on another. On a small, smooth problem
            (tanh or GELU) the unit step is accepted almost every time and each iteration costs about one evaluation. A
            ReLU network is piecewise linear: its gradient jumps where units switch on or off, so the line search needs
            more trials, sᵀy can come out zero or negative, and the run may stall. Try the ReLU preset.
          </>
        }
      >
        {lbfgs ? (
          <Plots cols={2} scale={0.55}>
            <Plot x={kAxis} y={alphaAxis} title="step length α" legend={false}>
              <Points name="α" x={lbfgsIters} y={lbfgs.stepSize} slot={0} size={4} />
            </Plot>
            <Plot x={kAxis} y={evalsAxis} title="evaluations per iteration" legend={false}>
              <Bars name="evaluations" x={lbfgsIters} y={lbfgs.lineEvaluations} slot={0} />
            </Plot>
            <Plot x={kAxis} y={gradAxis} title="gradient norm" legend={false}>
              <Curve name="‖∇f‖" x={lbfgs.iteration} y={lbfgs.gradNorm} slot={0} />
            </Plot>
            <Plot x={kAxis} y={syAxis} title="curvature sᵀy">
              <Points name="stored pair" x={kept.x} y={kept.y} slot={0} size={4} />
              {kept.skipX.length > 0 && (
                <Points name="skipped (|sᵀy|)" x={kept.skipX} y={kept.skipY} tone="destructive" size={6} />
              )}
            </Plot>
          </Plots>
        ) : (
          <div className="py-6 text-center text-sm text-muted-foreground">
            {trained.trained
              ? 'Train with L-BFGS (or compare all) to see its internals.'
              : 'Press Train: this figure shows the L-BFGS run.'}
          </div>
        )}
      </Figure>
    </>
  )
}
