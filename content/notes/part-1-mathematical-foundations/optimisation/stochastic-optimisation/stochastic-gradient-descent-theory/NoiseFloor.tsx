import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { normal as drawNormal, stream } from 'aifn/foundation/random'

/** f(x) = ½(λ₁x₁² + λ₂x₂²) with gradient noise of standard deviation σ in each coordinate; x* = 0. */
const LAMBDA = [1, 0.2] as const
const MU = LAMBDA[1]
const START: [number, number] = [2, 2]
const RUNS = 20

type Curves = { constant: number[]; decaying: number[]; averaged: number[] }

/** Mean squared distance to x* over RUNS seeded runs, for a constant step, a decaying step and iterate averaging. */
function simulate(eta: number, sigma: number, steps: number, seed: number): Curves {
  const out: Curves = {
    constant: new Array(steps + 1).fill(0),
    decaying: new Array(steps + 1).fill(0),
    averaged: new Array(steps + 1).fill(0),
  }
  const sq = (v: number[]) => v[0] * v[0] + v[1] * v[1]
  for (let r = 0; r < RUNS; r++) {
    const g = stream(seed * 1000 + r)
    const normal = () => drawNormal(g)
    const a = [...START]
    const b = [...START]
    const avg = [...START]
    out.constant[0] += sq(a) / RUNS
    out.decaying[0] += sq(b) / RUNS
    out.averaged[0] += sq(avg) / RUNS
    for (let k = 0; k < steps; k++) {
      // Robbins–Monro schedule: starts at η and decays like 1/(μk).
      const etaK = eta / (1 + MU * eta * k)
      for (let i = 0; i < 2; i++) {
        const noise = sigma * normal()
        a[i] -= eta * (LAMBDA[i] * a[i] + noise)
        b[i] -= etaK * (LAMBDA[i] * b[i] + sigma * normal())
        // Polyak–Ruppert: running mean of the constant-step iterates x₁ … x_{k+1}.
        avg[i] = k === 0 ? a[i] : avg[i] + (a[i] - avg[i]) / (k + 1)
      }
      out.constant[k + 1] += sq(a) / RUNS
      out.decaying[k + 1] += sq(b) / RUNS
      out.averaged[k + 1] += sq(avg) / RUNS
    }
  }
  return out
}

export function NoiseFloor() {
  const state = useFigureState({
    logEta: float(-1, {
      min: -2.5,
      max: 0,
      step: 0.05,
      label: 'step size η',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
    sigma: float(1, { min: 0.1, max: 3, step: 0.1, label: 'noise σ' }),
    steps: int(1000, { min: 100, max: 3000, step: 100, label: 'iterations' }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })

  const eta = 10 ** state.logEta
  const curves = useMemo(
    () => simulate(eta, state.sigma, state.steps, state.seed),
    [eta, state.sigma, state.steps, state.seed],
  )
  // Stationary E‖x − x*‖² for constant steps: Σᵢ ησ² / (λᵢ(2 − ηλᵢ)).
  const floor = LAMBDA.reduce((s, l) => s + (eta * state.sigma * state.sigma) / (l * (2 - eta * l)), 0)

  const series = useMemo(() => {
    const ks = curves.constant.map((_, k) => k)
    return [
      { name: 'constant step η', x: ks, y: curves.constant, slot: 0 },
      { name: 'decaying step η/(1 + μηk)', x: ks, y: curves.decaying, slot: 1 },
      { name: 'average of constant-step iterates', x: ks, y: curves.averaged, slot: 2 },
      { name: 'predicted noise floor', x: [0, state.steps], y: [floor, floor], slot: 0, dashed: true },
    ] as const
  }, [curves, floor, state.steps])

  const tail = curves.constant.slice(Math.floor(state.steps * 0.8))
  const measured = tail.reduce((s, v) => s + v, 0) / tail.length

  const xAxis = useAxis({ label: 'iteration k', hold: 'union' })
  const yAxis = useAxis({ label: 'mean ‖x_k − x*‖²', hold: 'union', log: true })
  return (
    <Figure
      title="Constant steps stall at a noise floor"
      state={state}
      caption={`Mean squared distance to the minimiser over ${RUNS} runs of SGD on f = ½(x₁² + 0.2x₂²), where each gradient carries Gaussian noise of standard deviation σ per coordinate. The constant step converges fast and then fluctuates around the dashed floor ησ²Σ 1/(λᵢ(2 − ηλᵢ)). The decaying step and the running average both keep improving, roughly like 1/k. Halve η and watch the floor halve.`}

      readouts={
        <>
          <Readout label="predicted floor" value={formatNumber(floor)} />
          <Readout label="measured (last 20%)" value={formatNumber(measured)} />
          <Readout label="decaying step, final" value={formatNumber(curves.decaying[state.steps])} />
          <Readout label="average, final" value={formatNumber(curves.averaged[state.steps])} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={360}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Curve {...series[3]} />
      </Plot>
    </Figure>
  )
}
