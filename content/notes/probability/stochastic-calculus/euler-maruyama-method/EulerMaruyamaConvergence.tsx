import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { histogram, joinPaths } from '../_shared/sde'

/** Geometric Brownian motion dX = μX dt + σX dW on [0, 1] from X₀ = 1; exact solution exp((μ − σ²/2)t + σW_t). */
const MU = 1
const SIGMA = 0.8
const PATHS = 2000
const FINE = 256
const XS = linspace(0.02, 8, 200)

const lognormalPdf = (x: number) => {
  const m = MU - SIGMA ** 2 / 2
  return Math.exp(-((Math.log(x) - m) ** 2) / (2 * SIGMA ** 2)) / (x * SIGMA * Math.sqrt(2 * Math.PI))
}

export function EulerMaruyamaConvergence() {
  const k = useParam(2, { min: 0, max: 8, step: 1 })
  const shown = useParam(4, { min: 1, max: 50, step: 1 })
  const n = 2 ** k.value

  // One set of fine Brownian increments per path; coarser grids sum them, so every step size sees the same noise.
  const dW = useMemo(() => {
    const { normal } = rng(17)
    const sd = Math.sqrt(1 / FINE)
    return Array.from({ length: PATHS }, () => Float64Array.from({ length: FINE }, () => sd * normal()))
  }, [])

  const exactPaths = useMemo(
    () =>
      dW.slice(0, shown.value).map((inc) => {
        const x = [0]
        const y = [1]
        let w = 0
        for (let j = 0; j < FINE; j++) {
          w += inc[j]
          const t = (j + 1) / FINE
          x.push(t)
          y.push(Math.exp((MU - SIGMA ** 2 / 2) * t + SIGMA * w))
        }
        return { x, y }
      }),
    [dW, shown.value],
  )

  // The error statistics always use all 2,000 paths; only the drawn subset depends on the paths slider.
  const result = useMemo(() => {
    const h = 1 / n
    const stride = FINE / n
    const finals = new Float64Array(PATHS)
    let strong = 0
    for (let i = 0; i < PATHS; i++) {
      let x = 1
      let w = 0
      for (let s = 0; s < n; s++) {
        let inc = 0
        for (let j = s * stride; j < (s + 1) * stride; j++) inc += dW[i][j]
        x = x * (1 + MU * h + SIGMA * inc)
        w += inc
      }
      finals[i] = x
      strong += Math.abs(x - Math.exp(MU - SIGMA ** 2 / 2 + SIGMA * w))
    }
    return { finals, strong: strong / PATHS, weak: Math.abs((1 + MU * h) ** n - Math.exp(MU)) }
  }, [dW, n])

  const eulerPaths = useMemo(() => {
    const h = 1 / n
    const stride = FINE / n
    return dW.slice(0, shown.value).map((inc) => {
      let x = 1
      const px = [0]
      const py = [1]
      for (let s = 0; s < n; s++) {
        let dw = 0
        for (let j = s * stride; j < (s + 1) * stride; j++) dw += inc[j]
        x = x * (1 + MU * h + SIGMA * dw)
        px.push((s + 1) * h)
        py.push(x)
      }
      return { x: px, y: py }
    })
  }, [dW, n, shown.value])

  const pathSeries = useMemo<XYSeries[]>(
    () => [
      { name: 'exact paths', type: 'line', ...joinPaths(exactPaths), muted: true, thin: exactPaths.length > 1 },
      {
        name: `Euler–Maruyama, ${n} step${n > 1 ? 's' : ''}`,
        type: 'line',
        ...joinPaths(eulerPaths),
        slot: 0,
        thin: eulerPaths.length > 1,
      },
    ],
    [exactPaths, eulerPaths, n],
  )

  const histSeries = useMemo<XYSeries[]>(() => {
    const hist = histogram(result.finals, 0, 8, 40)
    return [
      { name: 'Euler–Maruyama X₁', type: 'bar', x: hist.x, y: hist.y, slot: 0 },
      { name: 'exact log-normal law', type: 'line', x: XS, y: XS.map(lognormalPdf), emphasis: true },
    ]
  }, [result])

  return (
    <Interactive
      title="Euler–Maruyama converges as the step shrinks"
      caption="Geometric Brownian motion dX = X dt + 0.8 X dW from X₀ = 1, simulated with 2ᵏ Euler–Maruyama steps on [0, 1]. Every step size uses the same Brownian paths, so the simulated and exact paths can be compared one by one (top; the paths slider sets how many pairs are drawn, as light lines when there are several). The histogram and the strong error always use all 2,000 simulated paths; the histogram of X₁ approaches the exact log-normal law (bottom). The path error halves only every two halvings of the step (strong order ½); the error in the mean halves with every halving (weak order 1)."
      controls={
        <>
          <ParamSlider label="k (2ᵏ steps)" param={k} withArrows />
          <ParamSlider label="paths" param={shown} withArrows format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="step h" value={formatNumber(1 / n)} />
          <Readout label="strong error E|X̂₁ − X₁|" value={formatNumber(result.strong)} />
          <Readout label="weak error |E X̂₁ − E X₁|" value={formatNumber(result.weak)} />
        </>
      }
    >
      <XYChart height={240} xLabel="t" yLabel="X_t" series={pathSeries} xRange={[0, 1]} yRange={[0, 8]} />
      <XYChart height={220} xLabel="X₁" yLabel="density" series={histSeries} xRange={[0, 8]} yRange={[0, 0.6]} />
    </Interactive>
  )
}
