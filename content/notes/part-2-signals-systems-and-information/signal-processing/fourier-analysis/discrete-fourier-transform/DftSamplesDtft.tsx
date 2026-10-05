import { useMemo } from 'react'
import { Curve, Figure, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { dft } from 'aifn-compute/foundation/fourier'
import { complexAbs, linspace, toFlat } from 'aifn-compute/foundation/tensor'

// A short sequence: a decaying oscillation, 8 samples.
const X = Array.from({ length: 8 }, (_, n) => 0.8 ** n * Math.cos(0.9 * n))
const OMEGA = toFlat(linspace(0, 2 * Math.PI, 721))
const dtftMagnitude = (w: number) => {
  let re = 0
  let im = 0
  X.forEach((v, n) => {
    re += v * Math.cos(-w * n)
    im += v * Math.sin(-w * n)
  })
  return Math.hypot(re, im)
}
const CURVE = OMEGA.map(dtftMagnitude)

/** The N-point DFT of an 8-sample sequence lands exactly on its DTFT at the frequencies 2πk/N. */
export function DftSamplesDtft() {
  const state = useFigureState({
    n: int(8, { min: 8, max: 64, step: 1, label: 'DFT length N', suggestions: [8, 16, 32, 64] }),
  })

  const r = useMemo(() => {
    const padded = [...X, ...Array(state.n - X.length).fill(0)]
    return toFlat(complexAbs(dft(padded))).map((m, k) => ({ w: (2 * Math.PI * k) / state.n, m }))
  }, [state.n])

  const series = [
    { name: 'DTFT |X(e^{iω})|', x: OMEGA.map((w) => w / Math.PI), y: CURVE, slot: 0 },
    {
      name: `${state.n}-point DFT |X[k]|`,
      x: r.map((p) => p.w / Math.PI),
      y: r.map((p) => p.m),
      emphasis: true,
    },
  ] as const

  const xAxis = useAxis({ label: 'ω / π', range: [0, 2] })
  const yAxis = useAxis({ label: 'magnitude', hold: 'union' })
  return (
    <Figure
      title="The DFT samples the DTFT"
      state={state}
      caption="An 8-sample sequence has a continuous, 2π-periodic DTFT (line). Its N-point DFT, computed after padding to N samples with zeros, gives exactly N equally spaced samples of that curve, at ω = 2πk/N (points). A larger N samples the same curve more densely; it does not change the curve."

      readouts={
        <>
          <Readout label="sequence length" value={X.length} />
          <Readout label="bin spacing 2π/N" value={`${(2 / state.n).toFixed(3)}π`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={260}>
        <Curve {...series[0]} />
        <Points {...series[1]} />
      </Plot>
    </Figure>
  )
}
