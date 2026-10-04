import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { db, groupDelay, iirLowpass, response as freqz } from '../_shared/design'
import { linspace, toFlat } from 'aifn/foundation/tensor'

type Family = 'butter' | 'cheby1' | 'cheby2'
const CIRCLE = toFlat(linspace(0, 2 * Math.PI, 181))

/**
 * Low-pass IIR designs from analog prototypes by the bilinear transform: magnitude, group delay and the pole–zero
 * plot, for a draggable cutoff.
 */
export function IirDesignExplorer() {
  const state = useFigureState({
    family: choice<Family>(
      [
        { value: 'butter', label: 'Butterworth' },
        { value: 'cheby1', label: 'Chebyshev I' },
        { value: 'cheby2', label: 'Chebyshev II' },
      ],
      'butter',
      { label: 'family' },
    ),
    order: int(4, { min: 1, max: 10, step: 1, label: 'order N', format: (v) => String(v) }),
    cutoff: slider(0.05, 0.9, 0.3, { step: 0.005, label: 'cutoff ω_c (×π)' }),
    ripple: float(1, {
      min: 0.1,
      max: 3,
      step: 0.1,
      label: 'passband ripple (dB)',
      when: (v) => v.family === 'cheby1',
    }),
    stop: int(40, {
      min: 20,
      max: 80,
      step: 5,
      label: 'stopband attenuation (dB)',
      format: (v) => `${v} dB`,
      when: (v) => v.family === 'cheby2',
    }),
  })

  const r = useMemo(() => {
    const f = iirLowpass(state.family, state.order, state.cutoff * Math.PI, state.ripple, state.stop)
    const resp = freqz(f.b, f.a, 1024)
    const gd = groupDelay(f.b, f.a, resp.omega)
    const maxRadius = Math.max(...f.poles.map((p) => Math.hypot(p.re, p.im)))
    return {
      x: resp.omega.map((w) => w / Math.PI),
      mag: resp.magnitude.map((m) => db(m, -120)),
      gd,
      poles: f.poles,
      zeros: f.zeros,
      maxRadius,
      gd0: gd[0],
    }
  }, [state.family, state.order, state.cutoff, state.ripple, state.stop])

  const mag = [{ name: '|H| (dB)', x: r.x, y: r.mag, slot: 0 }] as const
  const gd: SeriesSpec[] = [{ name: 'group delay', type: 'line', x: r.x, y: r.gd, slot: 0 }]
  const pz = [
    { name: 'unit circle', x: CIRCLE.map(Math.cos), y: CIRCLE.map(Math.sin), muted: true },
    { name: 'poles', x: r.poles.map((p) => p.re), y: r.poles.map((p) => p.im), slot: 0 },
    { name: 'zeros', x: r.zeros.map((z) => z.re), y: r.zeros.map((z) => z.im), slot: 1 },
  ] as const

  const xAxis = useAxis({ label: 'ω / π', range: [0, 1] })
  const yAxis = useAxis({ label: 'magnitude (dB)', range: [-100, 5] })
  const xAxis2 = useAxis({ label: 'ω / π', range: [0, 1] })
  const yAxis2 = useAxis({ label: 'group delay (samples)', range: [0, 40] })
  const xAxis3 = useAxis({ label: 'Re z', range: [-1.2, 1.2] })
  const yAxis3 = useAxis({ label: 'Im z', range: [-1.2, 1.2], equal: xAxis3 })
  return (
    <Figure
      title="IIR low-pass designs"
      state={state}
      caption="Butterworth, Chebyshev type I and type II low-pass filters, designed as analog prototypes and mapped to discrete time by the bilinear transform with pre-warping. Drag the cutoff. Butterworth is maximally flat; Chebyshev I trades passband ripple for a steeper edge; Chebyshev II keeps the passband flat and puts zeros on the unit circle in the stopband. Raising the order steepens the edge and pushes poles towards the unit circle, and the group delay peaks sharply near the band edge: IIR filters do not have linear phase."

      readouts={
        <>
          <Readout label="largest pole radius" value={formatNumber(r.maxRadius)} />
          <Readout label="group delay at DC (samples)" value={formatNumber(r.gd0)} />
          <Readout label="multiplies per sample (direct form)" value={2 * state.order + 1} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[2fr_1fr]">
        <div className="min-w-0 space-y-4">
          <Plot x={xAxis} y={yAxis} height={240}>
            <Curve {...mag[0]} />
            <Handle {...state.handle('cutoff', { label: 'ω_c' })} />
          </Plot>
          <Plot x={xAxis2} y={yAxis2} height={200}>
            {seriesLayers(gd)}
          </Plot>
        </div>
        <div className="min-w-0">
          <Plot x={xAxis3} y={yAxis3}>
            <Curve {...pz[0]} />
            <Points {...pz[1]} />
            <Points {...pz[2]} />
          </Plot>
        </div>
      </div>
    </Figure>
  )
}
