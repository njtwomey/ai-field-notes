/**
 * "Quantisation-aware training": aifn-methods `quantisationAwareTraining` in the worker. A small MLP is trained in
 * floating point, then at each bit width its weights are either rounded once (post-training quantisation) or
 * fine-tuned with fake-quantised weights and the straight-through estimator (core `fakeQuantise`) before rounding. The
 * player walks one bit width's fine-tuning: its quantised loss, and the weights in units of their layer's step against
 * the integer levels.
 */
import { useMemo } from 'react'
import type { QatOptions, QatSnapshot } from 'aifn-methods/neural/quantisation'
import { integerRange } from 'aifn/nn/quantise'
import { Player, usePlayhead } from '@lab/controls'
import { Figure } from '@lab/layout'
import { call, choice, float, int, row, useFigureState, type Task } from '@lab/state'
import { CLASSIFICATION_CASES, TrainControls, datasetChoice, useTrainedRun, type DatasetValue } from '@lab/views'
import { Annotation, Bars, Curve, Plot, Plots, Points, Readout, formatNumber, useAxis } from '@lab/viz'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(v) : '—')
const pct = (v: number | undefined) => (v !== undefined && Number.isFinite(v) ? `${Math.round(100 * v)}%` : '—')
const BITS = [2, 3, 4, 5, 6]
const DATA = datasetChoice(CLASSIFICATION_CASES)
type Settings = {
  data: DatasetValue
  width: number
  depth: number
  steps: number
  stepSize: number
  qatSteps: number
  qatStepSize: number
  seed: number
}

function qatTask(s: Settings): Task<QatSnapshot> {
  const options: QatOptions = {
    width: s.width,
    depth: s.depth,
    steps: s.steps,
    stepSize: s.stepSize,
    qatSteps: s.qatSteps,
    qatStepSize: s.qatStepSize,
    seed: s.seed,
    bits: BITS,
  }
  return call<QatSnapshot>(
    'applied/neural/quantisation/quantisationAwareTraining',
    DATA.task(s.data, `quantise-qat-data-${s.data.key}`),
    options,
  )
}

/** The weights of one checkpoint in units of their own layer's quantiser step, binned. */
function stepHistogram(weights: Float64Array, layers: [number, number][], scales: number[], binsPerStep = 8) {
  const units: number[] = []
  let at = 0
  layers.forEach(([a, b], k) => {
    for (let i = 0; i < a * b; i++) units.push(weights[at + i] / scales[k])
    at += a * b
  })
  const lo = Math.floor(Math.min(...units)) - 0.5
  const hi = Math.ceil(Math.max(...units)) + 0.5
  const width = 1 / binsPerStep
  const count = Math.max(1, Math.round((hi - lo) / width))
  const y = new Array<number>(count).fill(0)
  for (const u of units) y[Math.min(count - 1, Math.max(0, Math.floor((u - lo) / width)))] += 1
  return { x: y.map((_, i) => lo + (i + 0.5) * width), y, width }
}

export function QatSpecimen() {
  const state = useFigureState({
    data: DATA.field({ label: '1 · data', initial: 'spirals' }),
    net: row('2 · network and training', {
      width: int(24, { ge: 2, le: 64, suggestions: [8, 16, 24, 32], label: 'hidden units' }),
      depth: int(2, { ge: 1, le: 4, suggestions: [1, 2, 3], label: 'hidden layers' }),
      steps: int(1500, { ge: 1, suggestions: [500, 1500, 3000], label: 'float Adam steps' }),
      stepSize: float(0.01, { gt: 0, scale: 'log10', suggestions: [0.003, 0.01, 0.03], label: 'float Adam η' }),
      qatSteps: int(300, { ge: 1, suggestions: [100, 300, 1000], label: 'QAT steps per bit width' }),
      qatStepSize: float(0.003, { gt: 0, scale: 'log10', suggestions: [0.001, 0.003, 0.01], label: 'QAT Adam η' }),
      seed: int(0, { ge: 0, le: 9999, label: 'seed' }),
    }),
    bits: choice(BITS, 2, { label: '3 · bit width shown' }),
  })
  const settings: Settings = {
    data: { key: state.data.key, values: { ...state.data.values } },
    width: Number(state.net.width),
    depth: Number(state.net.depth),
    steps: Number(state.net.steps),
    stepSize: Number(state.net.stepSize),
    qatSteps: Number(state.net.qatSteps),
    qatStepSize: Number(state.net.qatStepSize),
    seed: Number(state.net.seed),
  }
  const trained = useTrainedRun(settings, qatTask)
  const r = trained.run.value
  const runKey = trained.trained
  const results = r?.results ?? []
  const shown = results.find((b) => b.bits === state.bits) ?? null
  const [i, setI] = usePlayhead(shown?.step.length ?? 1)
  const at = shown ? Math.min(i, shown.step.length - 1) : 0
  const hist = useMemo(
    () => (shown && r ? stepHistogram(shown.weights[at], r.layers, shown.scales[at]) : null),
    [shown, r, at],
  )
  const { qmin, qmax } = integerRange(state.bits, true)
  const done = r?.done ?? 0
  const total = r?.total ?? settings.steps + BITS.length * settings.qatSteps
  const finished = results.filter((b) => Number.isFinite(b.qat))

  const stepAxis = useAxis({ label: 'float Adam step', key: runKey, hold: 'union' })
  const lossAxis = useAxis({ label: 'training loss', log: true, key: runKey, hold: 'union' })
  const bitAxis = useAxis({ label: 'bits b', integer: true, range: [1.5, 6.5] })
  const accAxis = useAxis({ label: 'training accuracy', range: [0.4, 1.02] })
  const qatStepAxis = useAxis({ label: 'QAT step', key: runKey, hold: 'union' })
  const qatLossAxis = useAxis({ label: 'loss of the quantised network', log: true, key: runKey, hold: 'union' })
  const unitAxis = useAxis({ label: 'weight / layer step s', key: [runKey, state.bits], hold: 'union' })
  const countAxis = useAxis({ label: 'weights', key: [runKey, state.bits], hold: 'union' })
  return (
    <Figure
      title="Quantisation-aware training"
      purpose="Rounding a trained network's weights to a few bits (post-training quantisation) moves them off the minimum the network was trained to. Quantisation-aware training fine-tunes through the rounding: the forward pass uses quantised weights, and the straight-through estimator passes the gradient to the floating-point weights underneath as if rounding were the identity."
      defaultSize="XL"
      state={state}
      controls={
        <>
          <TrainControls
            run={trained as never}
            progress={total ? done / total : 0}
            progressText={`${done} / ${total} steps`}
          />
          <Player
            value={at}
            onChange={setI}
            count={shown?.step.length ?? 1}
            format={(k) => `QAT step ${shown?.step[k] ?? 0}`}
            label="4 · fine-tuning step"
          />
        </>
      }
      readouts={{
        network: (
          <>
            <Readout label="float accuracy" value={pct(r?.accuracy)} />
            <Readout label="phase" value={r?.phase ?? '—'} />
          </>
        ),
        [`at ${state.bits} bits`]: (
          <>
            <Readout label="PTQ accuracy" value={pct(shown?.ptq)} />
            <Readout label="QAT accuracy" value={pct(shown?.qat)} />
            <Readout label="integer levels" value={`${qmin} … ${qmax}`} />
            <Readout label="quantised loss at this step" value={shown ? fmt(shown.loss[at]) : '—'} />
          </>
        ),
      }}
      caption="Press Train: the worker trains a ReLU MLP in floating point by Adam (top left), then, for each bit width b from 2 to 6, rounds every weight matrix once to b-bit signed integers with one symmetric scale per matrix (PTQ), and separately fine-tunes the trained network with fake-quantised weights (QAT) before rounding it the same way. Top right: training accuracy against bits; the dashed line is floating point. Bottom left: the loss of the quantised network during QAT fine-tuning at every bit width (the chosen one in colour); step 0 is the PTQ network. Bottom right: every weight divided by its layer's step s at the player's step; the dashed lines are the integer levels the weights round to. Choose the bit width shown, then play or step through its fine-tuning. Biases stay in floating point, and activations are not quantised."
    >
      {r ? (
        <Plots cols={2}>
          <Plot x={stepAxis} y={lossAxis} title="floating-point training" legend={false}>
            <Curve name="loss" x={r.step} y={r.loss} slot={0} />
          </Plot>
          <Plot x={bitAxis} y={accAxis} title="accuracy after quantisation">
            {Number.isFinite(r.accuracy) && <Annotation y={r.accuracy} dashed />}
            <Curve
              name="PTQ (round once)"
              x={finished.map((b) => b.bits)}
              y={finished.map((b) => b.ptq)}
              slot={0}
              showPoints
            />
            <Curve
              name="QAT (fine-tune, then round)"
              x={finished.map((b) => b.bits)}
              y={finished.map((b) => b.qat)}
              slot={1}
              showPoints
            />
          </Plot>
          <Plot x={qatStepAxis} y={qatLossAxis} title="QAT fine-tuning">
            {results
              .filter((b) => b.bits !== state.bits)
              .map((b) => (
                <Curve key={b.bits} name="other bit widths" x={b.step} y={b.loss} muted />
              ))}
            {shown && <Curve name={`${state.bits} bits`} x={shown.step} y={shown.loss} slot={1} />}
            {shown && <Points name="now" x={[shown.step[at]]} y={[shown.loss[at]]} slot={1} live />}
          </Plot>
          <Plot x={unitAxis} y={countAxis} title={`weights against the ${state.bits}-bit levels`} legend={false}>
            {hist && <Bars name="weights" x={hist.x} y={hist.y} width={hist.width} slot={1} opacity={0.7} />}
            {hist &&
              Array.from({ length: qmax - qmin + 1 }, (_, k) => qmin + k).map((q) => (
                <Annotation key={q} x={q} dashed />
              ))}
          </Plot>
        </Plots>
      ) : (
        <div className="py-6 text-center text-sm text-muted-foreground">
          Press Train: the floating-point loss streams in, then PTQ and QAT at each bit width.
        </div>
      )}
    </Figure>
  )
}
