/**
 * Showcase: inverse kinematics of a two-link arm. A squared-error MLP and a mixture density network trained side by
 * side in the worker (`mixtureDensityRun`) to predict the joint angles from the hand position (`twoLinkArm`). Dragging
 * the target draws the two most probable configurations of the MDN against the squared-error network's single blended
 * one. Every number drawn comes from the run, from `mdnPredict` at a checkpoint, from `twoLinkJoints` or from the
 * dataset's truth.
 */
import { useMemo, useState } from 'react'
import { stream } from 'aifn/foundation/random'
import { fromData, toFlat } from 'aifn/foundation/tensor'
import type { InverseTruth } from 'aifn-methods/data'
import { twoLinkArm, twoLinkJoints } from 'aifn-methods/data/synthetic'
import { mdnPredict, type MdnSnapshot } from 'aifn-methods/learning/mixture-density'
import { Player } from '@lab/controls'
import { Figure } from '@lab/layout'
import { choice, int, row, useFigureState } from '@lab/state'
import { formatValue, optimiserField, TrainControls, trainingMethodOf, useTrainedRun } from '@lab/views'
import { Curve, Handle, Plot, Plots, Points, Raster, Readout, Segments, useAxis, type Vec2 } from '@lab/viz'
import { TrainingCurves } from './curves'
import { paramsOf, runTask, SLOT, useCheckpoint, useNetworks, type RunSettings } from './shared'

const f3 = (v: number) => (Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—')
const DATA_SEED = 1
const N = 600
const LENGTHS: [number, number] = [0.8, 0.5]
/** The prior box of the angles in `twoLinkArm` (shoulder θ₁, elbow θ₂). */
const BOX = { t1: [0, 1.8], t2: [-2.4, 2.4] } as const
const grid = (lo: number, hi: number, n: number) => Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
const T1 = grid(-0.4, 2.2, 66)
const T2 = grid(-2.8, 2.8, 71)
const LOG_SPAN = 12

/** The arm drawn as a polyline: shoulder, elbow, hand. */
function armLine(angles: readonly number[]) {
  const { elbow, hand } = twoLinkJoints(angles, LENGTHS)
  return { x: [0, elbow[0], hand[0]], y: [0, elbow[1], hand[1]], hand }
}

export function ArmShowcase() {
  const state = useFigureState({
    data: row('1 · data', {
      noise: choice([0.005, 0.01, 0.03], 0.01, { label: 'noise sd on the hand position' }),
    }),
    net: row('2 · networks (same body)', {
      components: int(3, { ge: 1, le: 8, suggestions: [1, 2, 3, 4, 6], label: 'MDN components K' }),
      width: int(24, { ge: 4, le: 48, suggestions: [16, 24, 32], label: 'hidden width (two tanh layers)' }),
    }),
    optimiser: optimiserField({ label: '3 · optimiser (full batch)', stepSize: 0.01 }),
    run: row('4 · run', {
      steps: int(1500, { ge: 1, suggestions: [1000, 1500, 2500], label: 'steps' }),
      seed: int(1, { label: 'seed', ge: 0, le: 9999 }),
    }),
  })
  const settings: RunSettings = {
    generator: 'twoLinkArm',
    knobs: { n: N, l1: LENGTHS[0], l2: LENGTHS[1], noise: state.data.noise },
    dataSeed: DATA_SEED,
    components: state.net.components,
    hidden: [state.net.width, state.net.width],
    method: trainingMethodOf(state.optimiser, { clipNorm: 10 }),
    steps: state.run.steps,
    seed: state.run.seed,
  }
  const trained = useTrainedRun(settings, runTask)
  const snap: MdnSnapshot | undefined = trained.run.value ?? undefined
  const shown = trained.trained ?? settings
  const nets = useNetworks(snap)
  const { shots, index, shot, pick, pickStep } = useCheckpoint(snap, trained.trained)

  const noise = shown.knobs.noise
  const data = useMemo(() => twoLinkArm(stream(DATA_SEED), { n: N, l1: LENGTHS[0], l2: LENGTHS[1], noise }), [noise])
  const truth = data.meta.truth as InverseTruth
  const hands = useMemo(() => {
    const v = toFlat(data.x)
    return { x: Array.from({ length: N }, (_, i) => v[2 * i]), y: Array.from({ length: N }, (_, i) => v[2 * i + 1]) }
  }, [data])

  // The target: a draggable hand position.
  const [target, setTarget] = useState<Vec2>([0.55, 0.75])
  const solutions = useMemo(() => truth.solutions(target).map((s) => s.value), [truth, target])

  const answer = useMemo(() => {
    if (!nets || !shot) return null
    const xt = fromData(Float64Array.from(target), [1, 2])
    const mix = mdnPredict(nets.mdn, paramsOf(shot.mixture), xt).mixture!
    const modes = mix.modes(0).slice(0, 2)
    const blended = Array.from(mdnPredict(nets.mean, paramsOf(shot.mean), xt).mean)
    // log p(θ | target) on the grid of angles, relative to its maximum and cut at −LOG_SPAN: the peaks are too narrow
    // for the density itself to show.
    const z = T2.map((t2) => T1.map((t1) => mix.logDensity(0, [t1, t2])))
    const top = Math.max(...z.flat())
    return { modes, blended, z: z.map((r) => r.map((v) => Math.max(-LOG_SPAN, v - top))) }
  }, [nets, shot, target])

  const miss = (angles: readonly number[] | undefined) => {
    if (!angles) return NaN
    const { hand } = armLine(angles)
    return Math.hypot(hand[0] - target[0], hand[1] - target[1])
  }

  const px = useAxis({ label: 'p₁', range: [-0.9, 1.4] })
  const py = useAxis({ label: 'p₂', range: [-0.7, 1.4], equal: px })
  const a1 = useAxis({ label: 'θ₁ shoulder', range: [T1[0], T1[T1.length - 1]] })
  const a2 = useAxis({ label: 'θ₂ elbow', range: [T2[0], T2[T2.length - 1]] })
  const box = [
    { from: [BOX.t1[0], BOX.t2[0]], to: [BOX.t1[1], BOX.t2[0]] },
    { from: [BOX.t1[1], BOX.t2[0]], to: [BOX.t1[1], BOX.t2[1]] },
    { from: [BOX.t1[1], BOX.t2[1]], to: [BOX.t1[0], BOX.t2[1]] },
    { from: [BOX.t1[0], BOX.t2[1]], to: [BOX.t1[0], BOX.t2[0]] },
  ] as const

  const empty = !trained.trained ? 'press Train to start' : !shot ? 'training…' : undefined
  const done = snap?.step ?? 0
  const total = shown.steps
  const h = snap?.history
  const hi = h && shot ? h.step.indexOf(shot.step) : -1
  const targetHandle = <Handle kind="point" at={target} onDrag={setTarget} label="target" />

  return (
    <Figure
      title="Two-link arm: the elbow can bend either way"
      purpose="Predict the joint angles that put the hand at a target. Most targets are reached with the elbow bent either way; the squared-error network averages the two answers into an arm that reaches neither, while the MDN proposes both."
      state={state}
      defaultSize="XL"
      controls={
        <TrainControls run={trained as never} progress={done / total} progressText={`${done} / ${total} steps`} />
      }
      readouts={
        <>
          <Readout label="step" value={shot ? shot.step : '—'} />
          <Readout label="MDN NLL" value={hi >= 0 ? f3(h!.nll[hi]) : '—'} />
          <Readout label="squared-error NLL" value={hi >= 0 ? f3(h!.meanNll[hi]) : '—'} />
          <Readout label="true solutions" value={String(solutions.length)} />
          <Readout label="miss: MDN mode 1" value={f3(miss(answer?.modes[0]?.value))} />
          <Readout label="miss: MDN mode 2" value={f3(miss(answer?.modes[1]?.value))} />
          <Readout label="miss: squared error" value={f3(miss(answer?.blended))} />
        </>
      }
      caption={
        <>
          aifn <code>mixtureDensityRun</code> on <code>twoLinkArm</code> ({N} hand positions of an arm with links{' '}
          {LENGTHS[0]} and {LENGTHS[1]}, angles uniform on θ₁ ∈ [0, 1.8], θ₂ ∈ [−2.4, 2.4], position noise sd {noise}):
          two MLPs with two tanh layers of {shown.hidden[0]} units, one on the squared error of the angles and one an
          MDN with K = {shown.components} diagonal Gaussians over (θ₁, θ₂). Left: the hand positions (grey), the true
          solutions at the target (ink, dashed), the MDN&apos;s two most probable modes drawn as arms (slots{' '}
          {SLOT.mdn + 1} and {SLOT.second + 1}), and the squared-error network&apos;s single answer, the average of the
          two bends, whose hand misses the target (slot {SLOT.mean + 1}). Right: the MDN&apos;s log p(θ | target)
          relative to its maximum (cut {LOG_SPAN} nats below), with the prior box of the angles, the true solutions
          (ink), the MDN modes and the squared-error answer between them. Drag the target on the left chart; play the
          checkpoints from step 0, or drag the step marker on either curve.
        </>
      }
    >
      <Plots cols={2} scale={0.7}>
        <Plot x={px} y={py} title={empty ?? 'arms reaching for the target'}>
          <Points name="hand positions" x={hands.x} y={hands.y} muted thin />
          {solutions.map((s, i) => {
            const l = armLine(s)
            return <Curve key={`truth-${i}`} name="true solutions" x={l.x} y={l.y} emphasis dashed width={1} />
          })}
          {answer?.modes.map((m, i) => {
            const l = armLine(m.value)
            return (
              <Curve
                key={`mode-${i}`}
                name={`MDN mode ${i + 1}`}
                x={l.x}
                y={l.y}
                slot={i === 0 ? SLOT.mdn : SLOT.second}
                width={3}
                showPoints
              />
            )
          })}
          {answer && (
            <Curve
              name="squared-error network"
              {...(({ x, y }) => ({ x, y }))(armLine(answer.blended))}
              slot={SLOT.mean}
              width={3}
              showPoints
            />
          )}
          {targetHandle}
        </Plot>
        <Plot x={a1} y={a2} title={empty ?? 'MDN log p(θ | target) over the angles'}>
          {answer && (
            <Raster
              x={T1}
              y={T2}
              z={answer.z}
              scale="sequential"
              range={[-LOG_SPAN, 0]}
              valueLabel="log p(θ | target) − max"
              fillOpacity={0.7}
            />
          )}
          <Segments segments={box} />
          {solutions.length > 0 && (
            <Points
              name="true solutions"
              x={solutions.map((s) => s[0])}
              y={solutions.map((s) => s[1])}
              emphasis
              size={8}
            />
          )}
          {answer && answer.modes.length > 0 && (
            <Points
              name="MDN modes"
              x={answer.modes.map((m) => m.value[0])}
              y={answer.modes.map((m) => m.value[1])}
              slot={SLOT.mdn}
              size={7}
            />
          )}
          {answer && (
            <Points
              name="squared-error network"
              x={[answer.blended[0]]}
              y={[answer.blended[1]]}
              slot={SLOT.mean}
              size={7}
            />
          )}
        </Plot>
      </Plots>
      <TrainingCurves snap={snap} step={shot?.step} onStep={pickStep} runKey={trained.trained} />
      <Player
        label="checkpoint"
        value={index}
        onChange={pick}
        count={Math.max(1, shots.length)}
        format={(i) => `step ${shots[i]?.step ?? 0}`}
      />
    </Figure>
  )
}
