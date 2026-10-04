import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const FS = 10
const F_MAX = 3 * FS
const FREQS = toFlat(linspace(0, F_MAX, 601))
const TIMES = toFlat(linspace(0, 1, 801))
/** f folded into [0, f_s/2]: the frequency a real sinusoid appears to have after sampling at f_s. */
const fold = (f: number) => Math.abs(f - FS * Math.round(f / FS))

/**
 * Left: the folding map from true to apparent frequency, a triangle wave with period f_s. Right: a sinusoid above
 * f_s/2 and its alias pass through the same samples.
 */
export function FoldingDiagram() {
  const state = useFigureState({
    f: float(13, { min: 0, max: F_MAX, step: 0.25, label: 'true frequency f (Hz)' }),
  })
  const apparent = fold(state.f)

  const left = useMemo(() => [{ name: 'apparent frequency', x: FREQS, y: FREQS.map(fold), slot: 0 }] as const, [])

  const right = useMemo(() => {
    const n = Array.from({ length: FS + 1 }, (_, i) => i / FS)
    // The alias has the same samples; for a cosine with zero phase, cos(2π f t) and cos(2π f_a t) agree at t = n/f_s.
    return [
      {
        name: `true: ${formatNumber(state.f)} Hz`,
        x: TIMES,
        y: TIMES.map((t) => Math.cos(2 * Math.PI * state.f * t)),
        slot: 0,
      },
      {
        name: `alias: ${formatNumber(apparent)} Hz`,
        x: TIMES,
        y: TIMES.map((t) => Math.cos(2 * Math.PI * apparent * t)),
        slot: 1,
        dashed: true,
      },
      { name: 'samples', x: n, y: n.map((t) => Math.cos(2 * Math.PI * state.f * t)), emphasis: true },
    ] as const
  }, [state.f, apparent])

  const xAxis = useAxis({ label: 'true frequency (Hz)', range: [0, F_MAX] })
  const yAxis = useAxis({ label: 'apparent frequency (Hz)', range: [0, FS / 2 + 0.5] })
  const xAxis2 = useAxis({ label: 't (s)', range: [0, 1] })
  const yAxis2 = useAxis({ label: 'x(t)', range: [-1.4, 1.4] })
  return (
    <Figure
      title="Frequencies fold back"
      state={state}
      caption="Sampling at f_s = 10 Hz maps every frequency onto [0, f_s/2]: the map folds back and forth like a triangle wave. Drag the line on the left or use the slider. On the right, the true cosine and its alias at the folded frequency pass through exactly the same samples, so after sampling they cannot be told apart."

      readouts={
        <>
          <Readout label="true f" value={`${formatNumber(state.f)} Hz`} />
          <Readout label="apparent" value={`${formatNumber(apparent)} Hz`} />
          <Readout label="nearest multiple of f_s" value={`${FS * Math.round(state.f / FS)} Hz`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={280}>
          <Curve {...left[0]} />
          <Handle {...state.handle('f', { label: 'f' })} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={280}>
          <Curve {...right[0]} />
          <Curve {...right[1]} />
          <Points {...right[2]} />
        </Plot>
      </div>
    </Figure>
  )
}
