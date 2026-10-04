import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const FS = 8
const T_END = 2
const DENSE = toFlat(linspace(0, T_END, 801))
// Samples well beyond the plotted window, so the truncated sinc sum barely differs from the ideal one inside it.
const PAD = 40
const sinc = (u: number) => (u === 0 ? 1 : Math.sin(Math.PI * u) / (Math.PI * u))

/**
 * A cosine sampled at f_s = 8 Hz and rebuilt by ideal sinc interpolation. Below f_s/2 the rebuilt curve is the
 * original; above it, the samples are those of a lower-frequency alias, and interpolation returns the alias.
 */
export function SampleAndReconstruct() {
  const state = useFigureState({
    f: float(1.5, { min: 0.25, max: 12, step: 0.25, label: 'signal frequency f (Hz)' }),
  })
  const phase = 0.4

  const r = useMemo(() => {
    const signal = (t: number) => Math.cos(2 * Math.PI * state.f * t + phase)
    const n0 = -PAD
    const n1 = Math.ceil(T_END * FS) + PAD
    const samples = Array.from({ length: n1 - n0 + 1 }, (_, i) => {
      const n = n0 + i
      return { n, t: n / FS, x: signal(n / FS) }
    })
    const rebuilt = DENSE.map((t) => samples.reduce((acc, s) => acc + s.x * sinc(FS * t - s.n), 0))
    const shown = samples.filter((s) => s.t >= 0 && s.t <= T_END)
    // The apparent frequency: f folded into [0, f_s/2].
    const folded = Math.abs(state.f - FS * Math.round(state.f / FS))
    return { rebuilt, shown, folded, original: DENSE.map(signal) }
  }, [state.f])

  const series = [
    { name: 'x(t)', x: DENSE, y: r.original, slot: 0 },
    { name: 'sinc reconstruction', x: DENSE, y: r.rebuilt, slot: 1, dashed: true },
    { name: 'samples x[n]', x: r.shown.map((s) => s.t), y: r.shown.map((s) => s.x), emphasis: true },
  ] as const
  const aliased = state.f > FS / 2

  const xAxis = useAxis({ label: 't (s)', range: [0, T_END] })
  const yAxis = useAxis({ label: 'x(t)', range: [-1.6, 1.6] })
  return (
    <Figure
      title="Sampling and ideal reconstruction"
      state={state}
      caption="A cosine of frequency f is sampled at f_s = 8 Hz (dots) and rebuilt from its samples by sinc interpolation (dashed). Below the Nyquist frequency f_s/2 = 4 Hz the reconstruction lies on the original. Above it, the same samples belong to a slower cosine, and interpolation returns that alias instead. Exactly at 4 Hz the result depends on the phase."

      readouts={
        <>
          <Readout label="f" value={`${formatNumber(state.f)} Hz`} />
          <Readout label="Nyquist f_s/2" value={`${FS / 2} Hz`} />
          <Readout label="reconstructed frequency" value={`${formatNumber(r.folded)} Hz`} />
          <Readout label="status" value={aliased ? 'aliased' : state.f === FS / 2 ? 'at Nyquist' : 'recovered'} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Points {...series[2]} />
      </Plot>
    </Figure>
  )
}
