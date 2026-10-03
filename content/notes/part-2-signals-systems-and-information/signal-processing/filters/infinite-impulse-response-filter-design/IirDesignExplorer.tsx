import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { db, freqz } from '@/lib/dsp'
import { linspace } from '@/lib/math'
import { groupDelay, iirLowpass } from '../_shared/design'

type Family = 'butter' | 'cheby1' | 'cheby2'
const CIRCLE = linspace(0, 2 * Math.PI, 181)

/**
 * Low-pass IIR designs from analog prototypes by the bilinear transform: magnitude, group delay and the pole–zero
 * plot, for a draggable cutoff.
 */
export function IirDesignExplorer() {
  const [family, setFamily] = useState<Family>('butter')
  const order = useParam(4, { min: 1, max: 10, step: 1 })
  const cutoff = useParam(0.3, { min: 0.05, max: 0.9, step: 0.005 })
  const ripple = useParam(1, { min: 0.1, max: 3, step: 0.1 })
  const stop = useParam(40, { min: 20, max: 80, step: 5 })

  const r = useMemo(() => {
    const f = iirLowpass(family, order.value, cutoff.value * Math.PI, ripple.value, stop.value)
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
  }, [family, order.value, cutoff.value, ripple.value, stop.value])

  const handles: Handle[] = [{ kind: 'x', at: cutoff.value, label: 'ω_c', onDrag: (x) => cutoff.set(x) }]
  const mag: XYSeries[] = [{ name: '|H| (dB)', type: 'line', x: r.x, y: r.mag, slot: 0 }]
  const gd: XYSeries[] = [{ name: 'group delay', type: 'line', x: r.x, y: r.gd, slot: 0 }]
  const pz: XYSeries[] = [
    { name: 'unit circle', type: 'line', x: CIRCLE.map(Math.cos), y: CIRCLE.map(Math.sin), muted: true },
    { name: 'poles', type: 'scatter', x: r.poles.map((p) => p.re), y: r.poles.map((p) => p.im), slot: 0 },
    { name: 'zeros', type: 'scatter', x: r.zeros.map((z) => z.re), y: r.zeros.map((z) => z.im), slot: 1 },
  ]

  return (
    <Interactive
      title="IIR low-pass designs"
      caption="Butterworth, Chebyshev type I and type II low-pass filters, designed as analog prototypes and mapped to discrete time by the bilinear transform with pre-warping. Drag the cutoff. Butterworth is maximally flat; Chebyshev I trades passband ripple for a steeper edge; Chebyshev II keeps the passband flat and puts zeros on the unit circle in the stopband. Raising the order steepens the edge and pushes poles towards the unit circle, and the group delay peaks sharply near the band edge: IIR filters do not have linear phase."
      controls={
        <>
          <ParamChoice
            label="family"
            value={family}
            onChange={setFamily}
            options={[
              { value: 'butter', label: 'Butterworth' },
              { value: 'cheby1', label: 'Chebyshev I' },
              { value: 'cheby2', label: 'Chebyshev II' },
            ]}
          />
          <ParamSlider label="order N" param={order} format={(v) => String(v)} withArrows />
          <ParamSlider label="cutoff ω_c (×π)" param={cutoff} />
          {family === 'cheby1' && <ParamSlider label="passband ripple (dB)" param={ripple} />}
          {family === 'cheby2' && (
            <ParamSlider label="stopband attenuation (dB)" param={stop} format={(v) => `${v} dB`} />
          )}
        </>
      }
      readout={
        <>
          <Readout label="largest pole radius" value={formatNumber(r.maxRadius)} />
          <Readout label="group delay at DC (samples)" value={formatNumber(r.gd0)} />
          <Readout label="multiplies per sample (direct form)" value={2 * order.value + 1} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[2fr_1fr]">
        <div className="min-w-0 space-y-4">
          <XYChart
            series={mag}
            xLabel="ω / π"
            yLabel="magnitude (dB)"
            xRange={[0, 1]}
            yRange={[-100, 5]}
            handles={handles}
            height={240}
          />
          <XYChart
            series={gd}
            xLabel="ω / π"
            yLabel="group delay (samples)"
            xRange={[0, 1]}
            yRange={[0, 40]}
            height={200}
          />
        </div>
        <div className="min-w-0">
          <XYChart series={pz} xLabel="Re z" yLabel="Im z" xRange={[-1.2, 1.2]} yRange={[-1.2, 1.2]} equalAspect />
        </div>
      </div>
    </Interactive>
  )
}
