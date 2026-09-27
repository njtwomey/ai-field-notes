import { useMemo, useState } from 'react'
import { Interactive, ParamButton, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'
import { rng } from '@/lib/math'

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
    const { normal } = rng(seed * 1000 + r)
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
  const [logEta, setLogEta] = useState(-1)
  const [sigma, setSigma] = useState(1)
  const [steps, setSteps] = useState(1000)
  const [seed, setSeed] = useState(1)

  const eta = 10 ** logEta
  const curves = useMemo(() => simulate(eta, sigma, steps, seed), [eta, sigma, steps, seed])
  // Stationary E‖x − x*‖² for constant steps: Σᵢ ησ² / (λᵢ(2 − ηλᵢ)).
  const floor = LAMBDA.reduce((s, l) => s + (eta * sigma * sigma) / (l * (2 - eta * l)), 0)

  const series = useMemo((): XYSeries[] => {
    const ks = curves.constant.map((_, k) => k)
    return [
      { name: 'constant step η', type: 'line', x: ks, y: curves.constant, slot: 0 },
      { name: 'decaying step η/(1 + μηk)', type: 'line', x: ks, y: curves.decaying, slot: 1 },
      { name: 'average of constant-step iterates', type: 'line', x: ks, y: curves.averaged, slot: 2 },
      { name: 'predicted noise floor', type: 'line', x: [0, steps], y: [floor, floor], slot: 0, dashed: true },
    ]
  }, [curves, floor, steps])

  const tail = curves.constant.slice(Math.floor(steps * 0.8))
  const measured = tail.reduce((s, v) => s + v, 0) / tail.length

  return (
    <Interactive
      title="Constant steps stall at a noise floor"
      caption={`Mean squared distance to the minimiser over ${RUNS} runs of SGD on f = ½(x₁² + 0.2x₂²), where each gradient carries Gaussian noise of standard deviation σ per coordinate. The constant step converges fast and then fluctuates around the dashed floor ησ²Σ 1/(λᵢ(2 − ηλᵢ)). The decaying step and the running average both keep improving, roughly like 1/k. Halve η and watch the floor halve.`}
      controls={
        <>
          <ParamSlider
            label="step size η"
            value={logEta}
            onChange={setLogEta}
            min={-2.5}
            max={0}
            step={0.05}
            format={(v) => formatNumber(10 ** v)}
          />
          <ParamSlider label="noise σ" value={sigma} onChange={setSigma} min={0.1} max={3} step={0.1} />
          <ParamSlider label="iterations" value={steps} onChange={setSteps} min={100} max={3000} step={100} />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New noise</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="predicted floor" value={formatNumber(floor)} />
          <Readout label="measured (last 20%)" value={formatNumber(measured)} />
          <Readout label="decaying step, final" value={formatNumber(curves.decaying[steps])} />
          <Readout label="average, final" value={formatNumber(curves.averaged[steps])} />
        </>
      }
    >
      <XYChart height={360} series={series} yLog xLabel="iteration k" yLabel="mean ‖x_k − x*‖²" />
    </Interactive>
  )
}
