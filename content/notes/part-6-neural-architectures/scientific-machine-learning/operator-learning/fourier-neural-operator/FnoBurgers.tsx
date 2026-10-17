import { useMemo } from 'react'
import { Curve, Figure, float, int, Player, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import type { Params } from 'aifn-compute/foundation/pytree'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, mean, readonlyData, square, sub, type Tensor } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { methodTraining, type MethodTrainingState } from 'aifn-compute/nn/training'
import { relativeL2Error } from 'aifn-compute/learning/metrics'
import { FourierNeuralOperator1d, type FnoParams } from 'aifn-methods/neural/operator'
import { closePeriodic, FixedHeight, usePlayerPosition, useSlicedRun } from '../_shared/figure-helpers'
import { BURGERS_GRID, burgersInputs, burgersOutputs } from './burgersPairs'

/** Viscosity and output time of u_t + (u²/2)_x = ν u_xx on the periodic interval [0, 1). */
const NU = 0.05
const T = 0.5
/** Training grid, and the solver's finer grid (64 points) on which the trained model is also evaluated. */
const COARSE = 32
const FINE = BURGERS_GRID
const TRAIN = 16
const TEST = 8
const EVERY = 10
const PANEL = 240
const data = (t: unknown) => readonlyData(t as Tensor) as Float64Array

/** The precomputed pairs (burgersPairs.ts): 64-point rows, and every second point for the 32-point training grid. */
const fineX = Array.from({ length: FINE }, (_, j) => j / FINE)
const coarseX = Array.from({ length: COARSE }, (_, j) => j / COARSE)
const fineA = Float64Array.from(burgersInputs.flat())
const fineU = Float64Array.from(burgersOutputs.flat())
const stride = BURGERS_GRID / COARSE
const coarsen = (v: Float64Array) => v.filter((_, k) => k % stride === 0)
const take = (v: Float64Array, n: number, from: number, count: number) =>
  fromData(v.slice(from * n, (from + count) * n), [count, n, 1])
const pairs = {
  train: { a: take(coarsen(fineA), COARSE, 0, TRAIN), u: take(coarsen(fineU), COARSE, 0, TRAIN) },
  test: { a: take(coarsen(fineA), COARSE, TRAIN, TEST), u: take(coarsen(fineU), COARSE, TRAIN, TEST) },
  testFine: { a: take(fineA, FINE, TRAIN, TEST), u: take(fineU, FINE, TRAIN, TEST) },
}

type Setup = { modes: number; width: number; layers: number; adamSteps: number; rate: number; seed: number }
type Checkpoint = {
  step: number
  params: FnoParams
  trainError: number
  testCoarse: number
  testFine: number
}

const model = (s: Pick<Setup, 'modes' | 'width' | 'layers'>) =>
  FourierNeuralOperator1d({ width: s.width, modes: s.modes, layers: s.layers, projection: 16 })

/**
 * Adam on the mean squared error at the 32-point grid, with the test error at 32 and at 64 points. The first checkpoint
 * (the untrained model) comes before any optimiser step; every loop is bounded by `adamSteps`.
 */
function* training(s: Setup): Generator<Checkpoint> {
  const p = pairs
  const net = model(s)
  const loss = (q: FnoParams, b: { a: Tensor; u: Tensor }) => mean(square(sub(net.apply(q, b.a), b.u)))
  let params = net.init(stream(`fno/init/${s.seed}`))
  const checkpoint = (step: number): Checkpoint => ({
    step,
    params,
    trainError: relativeL2Error(p.train.u, net.apply(params, p.train.a) as Tensor),
    testCoarse: relativeL2Error(p.test.u, net.apply(params, p.test.a) as Tensor),
    testFine: relativeL2Error(p.testFine.u, net.apply(params, p.testFine.a) as Tensor),
  })
  yield checkpoint(0)
  if (s.adamSteps === 0) return
  const it = live(methodTraining(loss as never, p.train as never, { method: 'adam', stepSize: s.rate }), {
    params: params as unknown as Params,
  })
  it.next()
  for (let k = 1; k <= s.adamSteps; k++) {
    const next = it.next()
    const state = next.done ? null : (next.value.state as MethodTrainingState<Params>)
    if (state && !state.diverged) params = state.params as unknown as FnoParams
    const stopped = !state || state.diverged
    if (k % EVERY === 0 || k === s.adamSteps || stopped) yield checkpoint(k)
    if (stopped) break
  }
}

/** A Fourier neural operator trained on 32-point Burgers data, evaluated at 32 and at 64 points. */
export function FnoBurgers() {
  const state = useFigureState({
    modes: int(4, { ge: 1, le: 16, suggestions: [1, 2, 4, 8, 16], label: 'Fourier modes k_max' }),
    width: int(4, { ge: 2, le: 32, suggestions: [4, 8, 16], label: 'channels' }),
    layers: int(2, { ge: 1, le: 4, suggestions: [1, 2, 4], label: 'Fourier layers' }),
    adamSteps: int(200, { ge: 0, le: 3000, suggestions: [100, 200, 1000], label: 'Adam steps' }),
    rate: float(0.02, { ge: 1e-4, le: 0.1, suggestions: [0.005, 0.02, 0.05], label: 'Adam learning rate' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
    instance: int(1, { ge: 1, le: TEST, label: 'test function' }),
  })
  const { modes, width, layers, adamSteps, rate, seed, instance } = state
  const setup = useMemo<Setup>(
    () => ({ modes, width, layers, adamSteps, rate, seed }),
    [modes, width, layers, adamSteps, rate, seed],
  )
  const run = useSlicedRun(setup, training)
  const count = run.checkpoints.length
  const [pos, setPos] = usePlayerPosition(setup)
  const at = Math.min(pos, Math.max(0, count - 1))
  const cp = run.checkpoints[at]

  const view = useMemo(() => {
    const i = TRAIN + instance - 1
    const a = Array.from(fineA.subarray(i * FINE, (i + 1) * FINE))
    const u = Array.from(fineU.subarray(i * FINE, (i + 1) * FINE))
    let coarse: number[] | null = null
    let fine: number[] | null = null
    if (cp) {
      const net = model(setup)
      const aCoarse = fromData(
        Float64Array.from(coarseX, (_, j) => a[j * stride]),
        [1, COARSE, 1],
      )
      coarse = Array.from(data(net.apply(cp.params, aCoarse)))
      fine = Array.from(data(net.apply(cp.params, fromData(Float64Array.from(a), [1, FINE, 1]))))
    }
    return { a: closePeriodic(fineX, a), u: closePeriodic(fineX, u), coarse, fine }
  }, [cp, instance, setup])

  const history = useMemo(() => {
    const kept = run.checkpoints.slice(0, at + 1)
    const clamp = (e: number) => Math.min(Math.max(e, 1e-3), 2)
    return {
      steps: kept.map((c) => c.step),
      train: kept.map((c) => clamp(c.trainError)),
      coarse: kept.map((c) => clamp(c.testCoarse)),
      fine: kept.map((c) => clamp(c.testFine)),
    }
  }, [run.checkpoints, at])

  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const aAxis = useAxis({ label: 'u(x, 0)', range: [-2, 2] })
  const x2Axis = useAxis({ label: 'x', range: [0, 1] })
  const uAxis = useAxis({ label: `u(x, ${T})`, range: [-1.2, 1.2] })
  const sAxis = useAxis({ label: 'step', range: [0, Math.max(10, adamSteps)] })
  const eAxis = useAxis({ label: 'relative error', log: true, range: [1e-3, 2] })
  const percent = (e?: number) => (e === undefined ? '…' : `${(100 * e).toFixed(1)}%`)

  return (
    <Figure
      title="One operator, two resolutions"
      state={state}
      defaultSize="XL"
      hoverReadout={false}
      caption={`A Fourier neural operator learns the solution operator of Burgers' equation u_t + (u²/2)_x = ${NU} u_xx on the periodic interval [0, 1), from the initial profile to the solution at t = ${T}. The ${TRAIN} training pairs were solved in advance on ${FINE} points by a pseudo-spectral method and are subsampled to ${COARSE} points; the model sees only the ${COARSE}-point data. Each Fourier layer keeps the lowest k_max frequencies. Training is Adam on the mean squared error; the run fills in as it trains, and the player scrubs its checkpoints. Top left: one test initial profile. Top right: its solution (dashed), the model's output at the ${COARSE} training points (dots), and the same model, with the same weights, evaluated on the ${FINE}-point grid (line). Bottom: relative L2 errors during training on the training pairs, and on ${TEST} test pairs at ${COARSE} and at ${FINE} points. With the defaults the whole run takes about a second; more channels, modes, layers or steps take proportionally longer.`}
      controls={
        <Player
          value={at}
          onChange={setPos}
          count={Math.max(1, count)}
          label="checkpoint"
          format={(a) => `step ${run.checkpoints[a]?.step ?? 0}`}
        />
      }
      readouts={
        <>
          <Readout
            label="step"
            value={run.error ? `stopped: ${run.error}` : cp ? `${cp.step}${run.done ? '' : ', training…'}` : '…'}
          />
          <Readout label={`test error at ${COARSE} points`} value={percent(cp?.testCoarse)} />
          <Readout label={`test error at ${FINE} points`} value={percent(cp?.testFine)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={xAxis} y={aAxis} title="input: initial profile">
            <Curve name="u(x, 0)" x={view.a.x} y={view.a.y} muted />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={x2Axis} y={uAxis} title={`output: solution at t = ${T}`}>
            <Curve name="true solution" x={view.u.x} y={view.u.y} emphasis dashed />
            {view.fine && (
              <Curve
                name={`FNO on ${FINE} points`}
                x={closePeriodic(fineX, view.fine).x}
                y={closePeriodic(fineX, view.fine).y}
                slot={0}
              />
            )}
            {view.coarse && <Points name={`FNO on ${COARSE} points`} x={coarseX} y={view.coarse} slot={1} size={6} />}
          </Plot>
        </FixedHeight>
        <div className="md:col-span-2">
          <FixedHeight height={PANEL}>
            <Plot x={sAxis} y={eAxis} title="error during training">
              <Curve name="training pairs" x={history.steps} y={history.train} slot={2} />
              <Curve name={`test, ${COARSE} points`} x={history.steps} y={history.coarse} slot={1} />
              <Curve name={`test, ${FINE} points`} x={history.steps} y={history.fine} slot={0} />
            </Plot>
          </FixedHeight>
        </div>
      </div>
    </Figure>
  )
}
