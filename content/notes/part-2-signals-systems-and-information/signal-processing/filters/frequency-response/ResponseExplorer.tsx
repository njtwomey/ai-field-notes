import { useMemo } from 'react'
import { choice, Curve, Figure, float, formatNumber, Handle, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { toFlat } from 'aifn/foundation/tensor'
import { unwrapPhase } from 'aifn/signal'
import { applyFilter, db, iirLowpass, response as freqz } from '../_shared/design'

type Preset = 'smoother' | 'average' | 'recursive' | 'butter'

const PRESETS: Record<Preset, { label: string; b: number[]; a: number[] }> = {
  smoother: { label: '[¼, ½, ¼]', b: [0.25, 0.5, 0.25], a: [1] },
  average: { label: '5-point average', b: [0.2, 0.2, 0.2, 0.2, 0.2], a: [1] },
  recursive: { label: 'y = 0.8y[n−1] + 0.2x', b: [0.2], a: [1, -0.8] },
  butter: (() => {
    const f = iirLowpass('butter', 4, 0.3 * Math.PI)
    return { label: 'Butterworth, order 4', b: f.b, a: f.a }
  })(),
}
const T0 = 200
const SHOW = 60

/**
 * The frequency response of a filter, and what it does to one sinusoid: drag the test frequency on the magnitude plot
 * to see the output scaled by |H| and shifted by the phase ∠H.
 */
export function ResponseExplorer() {
  const state = useFigureState({
    preset: choice<Preset>(
      (Object.keys(PRESETS) as Preset[]).map((k) => ({ value: k, label: PRESETS[k].label })),
      'butter',
      { label: 'filter' },
    ),
    w0: float(0.2, { min: 0.01, max: 0.99, step: 0.01, label: 'test frequency ω₀ (×π)' }),
  })
  const { b, a } = PRESETS[state.preset]

  const response = useMemo(() => {
    const f = freqz(b, a, 512)
    return {
      x: f.omega.map((w) => w / Math.PI),
      mag: f.magnitude.map((m) => db(m, -80)),
      phase: toFlat(unwrapPhase(f.phase)),
    }
  }, [b, a])

  const at = useMemo(() => {
    const omega = state.w0 * Math.PI
    // H(e^{iω₀}) = B(e^{iω₀}) / A(e^{iω₀}), evaluated directly.
    const H = (coef: number[]) =>
      coef.reduce((acc, c, k) => [acc[0] + c * Math.cos(omega * k), acc[1] - c * Math.sin(omega * k)], [0, 0])
    const [nr, ni] = H(b)
    const [dr, di] = H(a)
    const d = dr * dr + di * di
    const re = (nr * dr + ni * di) / d
    const im = (ni * dr - nr * di) / d
    const input = Array.from({ length: T0 + SHOW }, (_, n) => Math.cos(omega * n))
    const output = applyFilter(b, a, input)
    const n = Array.from({ length: SHOW }, (_, i) => T0 + i)
    return {
      gain: Math.hypot(re, im),
      phase: Math.atan2(im, re),
      n,
      input: n.map((k) => input[k]),
      output: n.map((k) => output[k]),
    }
  }, [b, a, state.w0])

  const magSeries = [{ name: '|H| (dB)', x: response.x, y: response.mag, slot: 0 }] as const
  const phaseSeries = [{ name: '∠H (unwrapped)', x: response.x, y: response.phase, slot: 0 }] as const
  const timeSeries = [
    { name: 'input cos(ω₀n)', x: at.n, y: at.input, slot: 1, dashed: true },
    { name: 'output (steady state)', x: at.n, y: at.output, slot: 0 },
  ] as const

  const xAxis = useAxis({ label: 'ω / π', range: [0, 1] })
  const yAxis = useAxis({ label: 'magnitude (dB)', range: [-80, 5] })
  const xAxis2 = useAxis({ label: 'ω / π', range: [0, 1] })
  const yAxis2 = useAxis({ label: 'phase (rad)', hold: 'union' })
  const xAxis3 = useAxis({ label: 'n', range: [T0, T0 + SHOW - 1] })
  const yAxis3 = useAxis({ label: 'amplitude', range: [-1.2, 1.2] })
  return (
    <Figure
      title="What a frequency response means"
      state={state}
      caption="A sinusoid passes through a linear time-invariant filter as a sinusoid of the same frequency, scaled by |H(e^{iω₀})| and shifted by ∠H(e^{iω₀}). Drag ω₀ along the magnitude response and compare the input (dashed) with the output after the start-up transient has died away. The phase delay −∠H/ω₀ is the shift in samples."

      readouts={
        <>
          <Readout label="|H(e^{iω₀})|" value={formatNumber(at.gain)} />
          <Readout label="gain (dB)" value={formatNumber(20 * Math.log10(at.gain))} />
          <Readout label="∠H (rad)" value={formatNumber(at.phase)} />
          <Readout label="phase delay (samples)" value={formatNumber(-at.phase / (state.w0 * Math.PI))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={240}>
          <Curve {...magSeries[0]} />
          <Handle {...state.handle('w0', { label: 'ω₀' })} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={240}>
          <Curve {...phaseSeries[0]} />
        </Plot>
      </div>
      <Plot x={xAxis3} y={yAxis3} height={220}>
        <Curve {...timeSeries[0]} />
        <Curve {...timeSeries[1]} />
      </Plot>
    </Figure>
  )
}
