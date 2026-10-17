import { useMemo } from 'react'
import { Curve, Figure, float, int, Player, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import type { Params } from 'aifn-compute/foundation/pytree'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, readonlyData, type Tensor } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { methodTraining, type MethodTrainingState } from 'aifn-compute/nn/training'
import { relativeL2Error } from 'aifn-compute/learning/metrics'
import { antiderivativeData } from 'aifn-methods/data/synthetic'
import { deepOnet, deepOnetLoss, type DeepOnetParams } from 'aifn-methods/neural/operator'
import { FixedHeight, usePlayerPosition, useSlicedRun } from '../_shared/figure-helpers'

/** Sensors m on [0, 1]; the training queries are the sensor points. */
const M = 25
const TRAIN = 100
/** Hidden width of both networks. */
const HIDDEN = 20
const TEST = 50
/** Optimiser steps between kept checkpoints. */
const EVERY = 20
const PANEL = 230
const SHOWN_BASIS = 8
const data = (t: unknown) => readonlyData(t as Tensor) as Float64Array

/** The functions, drawn once from a Gaussian process with length scale 0.2, and their antiderivatives. */
const pairs = (() => {
  const d = antiderivativeData(stream('deeponet/antiderivative'), { count: TRAIN + TEST, sensors: M, refine: 4 })
  const F = d.x.length
  const u = data(d.u)
  const G = data(d.G)
  const sv = data(d.sensorValues)
  const at = (from: number, count: number) => ({
    u: fromData(sv.slice(from * M, (from + count) * M), [count, M]),
    G: fromData(
      Float64Array.from({ length: count * M }, (_, k) => G[(from + Math.floor(k / M)) * F + d.sensorIndex[k % M]]),
      [count, M],
    ),
  })
  return {
    x: Array.from(d.x),
    F,
    sensorX: d.sensorIndex.map((j) => d.x[j]),
    u,
    G,
    train: at(0, TRAIN),
    test: at(TRAIN, TEST),
  }
})()
/** Query points scaled to [−1, 1] for the trunk: the sensors (training) and the fine grid (plots). */
const scaled = (xs: readonly number[]) =>
  fromData(
    Float64Array.from(xs, (x) => 2 * x - 1),
    [xs.length, 1],
  )
const Y = scaled(pairs.sensorX)
const FINE = scaled(pairs.x)

/** The network for a latent width p. */
const model = (latent: number) =>
  deepOnet({ sensors: M, latent, branchHidden: [HIDDEN], trunkHidden: [HIDDEN, HIDDEN] })

type Setup = { latent: number; adamSteps: number; rate: number; lbfgsSteps: number; seed: number }
type Checkpoint = {
  step: number
  phase: 'Adam' | 'L-BFGS'
  params: DeepOnetParams
  trainError: number
  testError: number
}

/** Adam, then full-batch L-BFGS, on the mean squared error over the training functions and the sensor queries. */
function* training(s: Setup): Generator<Checkpoint> {
  const net = model(s.latent)
  const loss = deepOnetLoss(net, Y)
  let params = net.init(stream(`deeponet/init/${s.seed}`))
  const checkpoint = (step: number, phase: Checkpoint['phase']): Checkpoint => ({
    step,
    phase,
    params,
    trainError: relativeL2Error(pairs.train.G, net.apply(params, pairs.train.u, Y) as Tensor),
    testError: relativeL2Error(pairs.test.G, net.apply(params, pairs.test.u, Y) as Tensor),
  })
  yield checkpoint(0, 'Adam')
  const phases = [
    { phase: 'Adam' as const, steps: s.adamSteps, method: { method: 'adam' as const, stepSize: s.rate } },
    { phase: 'L-BFGS' as const, steps: s.lbfgsSteps, method: { method: 'lbfgs' as const } },
  ]
  let step = 0
  for (const { phase, steps, method } of phases) {
    if (steps === 0) continue
    const it = live(methodTraining(loss as never, pairs.train as never, method), {
      params: params as unknown as Params,
    })
    it.next()
    for (let k = 1; k <= steps; k++) {
      const next = it.next()
      const state = next.done ? null : (next.value.state as MethodTrainingState<Params>)
      if (state && !state.diverged) params = state.params as unknown as DeepOnetParams
      step++
      const stopped = !state || state.stopped || state.diverged
      if (k % EVERY === 0 || k === steps || stopped) yield checkpoint(step, phase)
      if (stopped) break
    }
  }
}

/**
 * A DeepONet learning the antiderivative operator G(u)(y) = ∫₀ʸ u(s) ds from functions observed at 50 sensors, with
 * its trunk basis functions.
 */
export function DeepOnetAntiderivative() {
  const state = useFigureState({
    latent: int(10, { ge: 1, le: 60, suggestions: [1, 2, 5, 10, 40], label: 'latent width p' }),
    adamSteps: int(100, { ge: 0, le: 5000, suggestions: [0, 100, 1000], label: 'Adam steps' }),
    rate: float(0.01, { ge: 1e-4, le: 0.1, suggestions: [0.001, 0.003, 0.01], label: 'Adam learning rate' }),
    lbfgsSteps: int(400, { ge: 0, le: 3000, suggestions: [0, 400, 1000], label: 'L-BFGS steps' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
    instance: int(1, { ge: 1, le: TEST, label: 'test function' }),
  })
  const { latent, adamSteps, rate, lbfgsSteps, seed, instance } = state
  const setup = useMemo<Setup>(
    () => ({ latent, adamSteps, rate, lbfgsSteps, seed }),
    [latent, adamSteps, rate, lbfgsSteps, seed],
  )
  const run = useSlicedRun(setup, training)
  const count = run.checkpoints.length
  const [pos, setPos] = usePlayerPosition(setup)
  const at = Math.min(pos, Math.max(0, count - 1))
  const cp = run.checkpoints[at]

  const view = useMemo(() => {
    const i = TRAIN + instance - 1
    const u = Array.from(pairs.u.subarray(i * pairs.F, (i + 1) * pairs.F))
    const G = Array.from(pairs.G.subarray(i * pairs.F, (i + 1) * pairs.F))
    const sensors = Array.from(data(pairs.test.u).subarray((instance - 1) * M, instance * M))
    if (!cp) return { u, G, sensors, prediction: null, basis: [] as number[][] }
    const net = model(latent)
    const branch = fromData(Float64Array.from(sensors), [1, M])
    const prediction = Array.from(data(net.apply(cp.params, branch, FINE)))
    const B = data(net.basis(cp.params, FINE))
    const basis = Array.from({ length: Math.min(latent, SHOWN_BASIS) }, (_, k) =>
      Array.from({ length: pairs.F }, (_, j) => B[j * latent + k]),
    )
    return { u, G, sensors, prediction, basis }
  }, [cp, instance, latent])

  const history = useMemo(() => {
    const kept = run.checkpoints.slice(0, at + 1)
    const clamp = (e: number) => Math.min(Math.max(e, 1e-3), 2)
    return {
      steps: kept.map((c) => c.step),
      train: kept.map((c) => clamp(c.trainError)),
      test: kept.map((c) => clamp(c.testError)),
    }
  }, [run.checkpoints, at])

  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const uAxis = useAxis({ label: 'u(x)', range: [-3, 3] })
  const yAxis = useAxis({ label: 'y', range: [0, 1] })
  const gAxis = useAxis({ label: 'G(u)(y)', range: [-1.5, 1.5] })
  const bxAxis = useAxis({ label: 'y', range: [0, 1] })
  const bAxis = useAxis({ label: 'trunk output t_k(y)', range: [-1.1, 1.1] })
  const sAxis = useAxis({ label: 'step', range: [0, Math.max(10, adamSteps + lbfgsSteps)] })
  const eAxis = useAxis({ label: 'relative error', log: true, range: [1e-3, 2] })
  const percent = (e?: number) => (e === undefined ? '…' : `${(100 * e).toFixed(1)}%`)

  return (
    <Figure
      title="A DeepONet learns to integrate"
      state={state}
      defaultSize="XL"
      hoverReadout={false}
      caption={`A DeepONet trained on ${TRAIN} functions drawn from a Gaussian process with an RBF kernel of length scale 0.2 on [0, 1]. The branch network (one hidden layer of ${HIDDEN} tanh units) reads each function at ${M} evenly spaced sensors and returns p coefficients; the trunk network (two hidden layers of ${HIDDEN} tanh units) reads a query point y and returns p basis values; the prediction is their dot product plus a bias. The targets are the antiderivatives G(u)(y) = ∫₀ʸ u(s) ds at the sensor points. Training is Adam, then full-batch L-BFGS; the run fills in as it trains, and the player scrubs its checkpoints. Top left: one test function, never seen in training, with its sensor values (dots). Top right: its true antiderivative (dashed) and the DeepONet's prediction on a grid four times finer than the sensors. Bottom left: up to ${SHOWN_BASIS} of the p trunk basis functions. Bottom right: the relative L2 error on the training and test functions. With p = 1 every prediction is a multiple of one function of y.`}
      controls={
        <Player
          value={at}
          onChange={setPos}
          count={Math.max(1, count)}
          label="checkpoint"
          format={(a) => {
            const c = run.checkpoints[a]
            return c ? `step ${c.step} (${c.phase})` : 'step 0'
          }}
        />
      }
      readouts={
        <>
          <Readout label="step" value={cp ? `${cp.step} (${cp.phase})${run.done ? '' : ', training…'}` : '…'} />
          <Readout label="training error" value={percent(cp?.trainError)} />
          <Readout label="test error" value={percent(cp?.testError)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={xAxis} y={uAxis} title="input function u">
            <Curve name="u" x={pairs.x} y={view.u} muted />
            <Points name="sensor values" x={pairs.sensorX} y={view.sensors} slot={2} size={5} />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={yAxis} y={gAxis} title="output function G(u)">
            <Curve name="true antiderivative" x={pairs.x} y={view.G} emphasis dashed />
            {view.prediction && <Curve name="DeepONet" x={pairs.x} y={view.prediction} slot={0} />}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={bxAxis} y={bAxis} title="trunk basis functions" legend={false}>
            {view.basis.map((b, k) => (
              <Curve key={k} name={`t_${k + 1}`} x={pairs.x} y={b} slot={0} thin silent />
            ))}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={sAxis} y={eAxis} title="error during training">
            <Curve name="training functions" x={history.steps} y={history.train} slot={1} />
            <Curve name="test functions" x={history.steps} y={history.test} slot={0} />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
