import { useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  FrameContext,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { partials } from 'aifn-compute/dynamics/fields'
import type { Params } from 'aifn-compute/foundation/pytree'
import { stream } from 'aifn-compute/foundation/random'
import {
  add,
  fromData,
  mean,
  mul,
  sin,
  square,
  sub,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { xavierUniform } from 'aifn-compute/nn/init'
import { Mlp } from 'aifn-compute/nn/layers'
import { methodTraining, type MethodTrainingState } from 'aifn-compute/nn/training'

/** The natural frequency ω₀: about two and a half swings on [0, 1]. */
const OMEGA = 15
/** Observations, evenly spaced from t = 0 to the end of the data window. */
const OBSERVED = 10
/** Optimiser steps between kept checkpoints. */
const EVERY = 10
/** Milliseconds of training per slice, so the page stays responsive while the run fills in. */
const SLICE = 25
const PANEL = 260
const ts = Array.from({ length: 151 }, (_, i) => i / 150)
const column = (v: readonly number[]) => fromData(Float64Array.from(v), [v.length, 1])
const T = column(ts)
/** A rank-0 value as a number. */
const scalar = (v: Value) => {
  const raw = unwrap(v)
  return typeof raw === 'number' ? raw : toFlat(raw as Tensor)[0]
}

const ACTIVATIONS = [
  { value: 'tanh', label: 'tanh' },
  { value: 'sine', label: 'sine' },
  { value: 'softplus', label: 'softplus' },
  { value: 'relu', label: 'ReLU' },
] as const
type ActivationKey = (typeof ACTIVATIONS)[number]['value']

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** The underdamped oscillator released at rest from x = 1: x(t) = e^{−ζω₀t}(cos ω_d t + (ζω₀/ω_d) sin ω_d t). */
const exact = (zeta: number) => {
  const wd = OMEGA * Math.sqrt(1 - zeta * zeta)
  return (t: number) => Math.exp(-zeta * OMEGA * t) * (Math.cos(wd * t) + ((zeta * OMEGA) / wd) * Math.sin(wd * t))
}

type Setup = {
  zeta: number
  until: number
  collocation: number
  weight: number
  activation: ActivationKey
  adamSteps: number
  rate: number
  lbfgsSteps: number
  seed: number
}

type Checkpoint = {
  step: number
  phase: 'Adam' | 'L-BFGS'
  /** Each network on the grid: data only, then with the equation. */
  curves: [number[], number[]]
  /** Each network's residual on the grid, in the same order. */
  residuals: [number[], number[]]
  /** Each network's relative L2 error against the exact motion. */
  errors: [number, number]
  /** The physics-informed network's mean squared residual at the collocation points. */
  residualLoss: number
}

/**
 * Trains the two networks side by side and yields a checkpoint every few steps. Both start from the same weights and see
 * the same observations; the second is also penalised for breaking the equation at the collocation points. The
 * optimisers are the engine's (`methodTraining`): Adam, then full-batch L-BFGS.
 */
function* training(s: Setup): Generator<Checkpoint> {
  const activation = s.activation === 'sine' ? (x: Value) => sin(x) : s.activation
  const net = Mlp([1, 32, 32, 1], { activation, init: xavierUniform() })
  // The network on [0, 1], its input scaled to [−1, 1].
  const u = (p: Params, t: Value) => net.apply(p as never, sub(mul(t, 2), 1))
  const x = exact(s.zeta)
  const td = Array.from({ length: OBSERVED }, (_, i) => (s.until * i) / (OBSERVED - 1))
  const data = { t: column(td), y: column(td.map(x)) }
  const C = column(Array.from({ length: s.collocation }, (_, i) => i / (s.collocation - 1)))
  const truth = ts.map(x)
  const norm = Math.hypot(...truth)
  // The residual divided by ω₀², so that its three terms are of the size of x.
  const residual = (p: Params, at: Value) => {
    const [v, vt, vtt] = partials((t) => u(p, t), at, [[], [0], [0, 0]])
    return add(add(mul(1 / OMEGA ** 2, vtt), mul((2 * s.zeta) / OMEGA, vt)), v)
  }
  const misfit = (p: Params, b: typeof data) => mean(square(sub(u(p, b.t), b.y)))
  const losses = [misfit, (p: Params, b: typeof data) => add(misfit(p, b), mul(s.weight, mean(square(residual(p, C)))))]

  let params: Params[] = Array(2).fill(net.init(stream(`physics-informed-oscillator/${s.seed}`)) as Params)
  const checkpoint = (step: number, phase: Checkpoint['phase']): Checkpoint => {
    const curves = params.map((p) => Array.from(toFlat(u(p, T) as Tensor))) as [number[], number[]]
    return {
      step,
      phase,
      curves,
      residuals: params.map((p) => Array.from(toFlat(residual(p, T) as Tensor))) as [number[], number[]],
      errors: curves.map((c) => Math.hypot(...c.map((v, i) => v - truth[i])) / norm) as [number, number],
      residualLoss: scalar(mean(square(residual(params[1], C)))),
    }
  }
  yield checkpoint(0, 'Adam')

  const phases = [
    { phase: 'Adam' as const, steps: s.adamSteps, method: { method: 'adam' as const, stepSize: s.rate } },
    { phase: 'L-BFGS' as const, steps: s.lbfgsSteps, method: { method: 'lbfgs' as const } },
  ]
  let step = 0
  for (const { phase, steps, method } of phases) {
    if (steps === 0) continue
    const runs = params.map((p, i) => {
      const it = live(methodTraining(losses[i], data, method), { params: p })
      it.next()
      return it
    })
    const stopped = [false, false]
    for (let k = 1; k <= steps; k++) {
      params = params.map((p, i) => {
        if (stopped[i]) return p
        const next = runs[i].next()
        const state = next.done ? null : (next.value.state as MethodTrainingState<Params>)
        if (!state || state.stopped || state.diverged) stopped[i] = true
        return state && !state.diverged ? state.params : p
      })
      step++
      if (k % EVERY === 0 || k === steps) yield checkpoint(step, phase)
      if (stopped.every(Boolean)) {
        if (k % EVERY !== 0) yield checkpoint(step, phase)
        break
      }
    }
  }
}

/** Runs `training` in short slices between frames and returns the checkpoints so far, keyed to the setup. */
function useTraining(setup: Setup) {
  const [run, setRun] = useState<{ setup: Setup; checkpoints: Checkpoint[]; done: boolean }>({
    setup,
    checkpoints: [],
    done: false,
  })
  useEffect(() => {
    const it = training(setup)
    const checkpoints: Checkpoint[] = []
    let timer = 0
    const slice = () => {
      const start = performance.now()
      let done = false
      while (performance.now() - start < SLICE) {
        const next = it.next()
        if (next.done) {
          done = true
          break
        }
        checkpoints.push(next.value)
      }
      setRun({ setup, checkpoints: [...checkpoints], done })
      if (!done) timer = window.setTimeout(slice, 0)
    }
    timer = window.setTimeout(slice, 0)
    return () => window.clearTimeout(timer)
  }, [setup])
  return run.setup === setup ? run : { setup, checkpoints: [], done: false }
}

/**
 * A network trained on a few early observations of a damped oscillator, against the same network also trained to obey
 * the oscillator's equation x″ + 2ζω₀x′ + ω₀²x = 0 at collocation points across the whole interval.
 */
export function OscillatorPinn() {
  const state = useFigureState({
    zeta: slider(0.02, 0.4, 0.2, { step: 0.02, label: 'damping ratio ζ' }),
    until: slider(0.1, 0.6, 0.24, { step: 0.02, label: 'data up to t' }),
    collocation: int(40, { ge: 2, le: 200, suggestions: [5, 10, 20, 40, 80], label: 'collocation points' }),
    weight: float(1, { ge: 0, le: 100, suggestions: [0.01, 0.1, 1, 10], label: 'equation weight λ' }),
    activation: choice(ACTIVATIONS, 'tanh', { label: 'activation' }),
    adamSteps: int(200, { ge: 0, le: 5000, suggestions: [100, 200, 1000, 3000], label: 'Adam steps' }),
    rate: float(0.03, { ge: 1e-4, le: 0.3, suggestions: [0.003, 0.01, 0.03], label: 'Adam learning rate' }),
    lbfgsSteps: int(1000, { ge: 0, le: 3000, suggestions: [0, 500, 1000, 2000], label: 'L-BFGS steps' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { zeta, until, collocation, weight, activation, adamSteps, rate, lbfgsSteps, seed } = state
  const setup = useMemo<Setup>(
    () => ({ zeta, until, collocation, weight, activation, adamSteps, rate, lbfgsSteps, seed }),
    [zeta, until, collocation, weight, activation, adamSteps, rate, lbfgsSteps, seed],
  )
  const run = useTraining(setup)
  // The end of the data window follows the pointer while dragged and restarts training when let go.
  const [draft, setDraft] = useState<number | null>(null)
  const end = draft ?? until

  const [pos, setPos] = useState({ setup, at: 0 })
  const count = run.checkpoints.length
  const at = pos.setup === setup ? Math.min(pos.at, Math.max(0, count - 1)) : 0
  const cp = run.checkpoints[at]

  const view = useMemo(() => {
    const x = exact(zeta)
    const td = Array.from({ length: OBSERVED }, (_, i) => (end * i) / (OBSERVED - 1))
    const tc = Array.from({ length: collocation }, (_, i) => i / (collocation - 1))
    return { truth: ts.map(x), td, yd: td.map(x), tc }
  }, [zeta, end, collocation])
  const history = useMemo(() => {
    const kept = run.checkpoints.slice(0, at + 1)
    return {
      steps: kept.map((c) => c.step),
      data: kept.map((c) => Math.min(Math.max(c.errors[0], 1e-4), 10)),
      pinn: kept.map((c) => Math.min(Math.max(c.errors[1], 1e-4), 10)),
    }
  }, [run.checkpoints, at])

  const tAxis = useAxis({ label: 't', range: [0, 1] })
  const xAxis = useAxis({ label: 'x', range: [-1.4, 1.4] })
  const tAxis2 = useAxis({ label: 't', range: [0, 1] })
  const rAxis = useAxis({ label: 'residual / ω₀²', range: [-1.5, 1.5] })
  const sAxis = useAxis({ label: 'step', range: [0, Math.max(10, adamSteps + lbfgsSteps)] })
  const eAxis = useAxis({ label: 'relative error', log: true, range: [1e-4, 10] })
  const percent = (e?: number) => (e === undefined ? '…' : `${(100 * e).toFixed(1)}%`)

  return (
    <Figure
      title="Two networks, one equation between them"
      state={state}
      defaultSize="XL"
      hoverReadout={false}
      caption={`Two multilayer perceptrons with two hidden layers of 32 units start from the same weights and see the same ${OBSERVED} observations of a damped oscillator released at rest from x = 1 (orange dots). The data-only network fits the observations alone. The physics-informed network also pays λ times the mean squared residual of x″ + 2ζω₀x′ + ω₀²x = 0 (divided by ω₀², with ω₀ = ${OMEGA}) at the collocation points (grey ticks). Training is Adam, then full-batch L-BFGS; the run fills in as it trains, and the player scrubs its checkpoints. Top: the true motion (dashed) and the two networks. Bottom left: each network's residual on a fine grid. Bottom right: the relative L2 error of each network on [0, 1], on a log scale. Drag the vertical line to end the data earlier or later. With ReLU the network's second derivative is zero almost everywhere, so the residual sees only the first-order part of the equation.`}
      controls={
        <Player
          value={at}
          onChange={(a) => setPos({ setup, at: a })}
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
          <Readout label="error, data only" value={percent(cp?.errors[0])} />
          <Readout label="error, with the equation" value={percent(cp?.errors[1])} />
          <Readout
            label="mean squared residual at the collocation points"
            value={cp ? formatNumber(cp.residualLoss) : '…'}
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="md:col-span-2">
          <FixedHeight height={PANEL + 40}>
            <Plot x={tAxis} y={xAxis}>
              <Curve name="true motion" x={ts} y={view.truth} muted dashed />
              {cp && <Curve name="data only" x={ts} y={cp.curves[0]} slot={1} />}
              {cp && <Curve name="with the equation" x={ts} y={cp.curves[1]} slot={0} emphasis />}
              <Points name="observations" x={view.td} y={view.yd} slot={2} size={8} />
              <Points name="collocation points" x={view.tc} y={view.tc.map(() => -1.3)} size={3} muted />
              <Handle
                kind="x"
                at={end}
                label="data ends"
                onDrag={(t) => setDraft(Math.min(0.6, Math.max(0.1, Math.round(t * 50) / 50)))}
                onRelease={() => {
                  if (draft !== null) state.set('until', draft)
                  setDraft(null)
                }}
              />
            </Plot>
          </FixedHeight>
        </div>
        <FixedHeight height={PANEL}>
          <Plot x={tAxis2} y={rAxis} title="residual">
            {cp && <Curve name="data only" x={ts} y={cp.residuals[0]} slot={1} />}
            {cp && <Curve name="with the equation" x={ts} y={cp.residuals[1]} slot={0} emphasis />}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={sAxis} y={eAxis} title="error against the true motion">
            <Curve name="data only" x={history.steps} y={history.data} slot={1} />
            <Curve name="with the equation" x={history.steps} y={history.pinn} slot={0} emphasis />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
