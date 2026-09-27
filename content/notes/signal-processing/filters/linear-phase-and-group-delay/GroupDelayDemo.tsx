import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { freqz, lfilter, unwrap } from '@/lib/dsp'
import { firLowpass, groupDelay, iirLowpass } from '../_shared/design'

const LENGTH = 400
const CENTRE = 120
const WIDTH = 18

const FILTERS = {
  fir: { label: 'linear-phase FIR (41 taps)', b: firLowpass(41, 0.5 * Math.PI, 'hamming'), a: [1] },
  butter: (() => {
    const f = iirLowpass('butter', 6, 0.35 * Math.PI)
    return { label: 'Butterworth, order 6', b: f.b, a: f.a }
  })(),
}
type Key = keyof typeof FILTERS

/**
 * A Gaussian tone burst through a filter. Its envelope is delayed by the group delay at the carrier frequency; the
 * carrier's phase is shifted by the phase delay.
 */
export function GroupDelayDemo() {
  const [key, setKey] = useState<Key>('butter')
  const carrier = useParam(0.2, { min: 0.05, max: 0.45, step: 0.01 })
  const { b, a } = FILTERS[key]

  const r = useMemo(() => {
    const w0 = carrier.value * Math.PI
    const envelope = Array.from({ length: LENGTH }, (_, n) => Math.exp(-0.5 * ((n - CENTRE) / WIDTH) ** 2))
    const x = envelope.map((e, n) => e * Math.cos(w0 * n))
    const y = Array.from(lfilter(b, a, x))
    // Output envelope: the local maximum of |y| over one carrier period, whose peak locates the delayed envelope.
    const period = Math.max(2, Math.round((2 * Math.PI) / w0))
    const env = y.map((_, n) => {
      let m = 0
      for (let k = Math.max(0, n - (period >> 1)); k <= Math.min(LENGTH - 1, n + (period >> 1)); k++)
        m = Math.max(m, Math.abs(y[k]))
      return m
    })
    const peak = env.indexOf(Math.max(...env))
    const tg = groupDelay(b, a, [w0])[0]
    // Phase delay needs the unwrapped phase at ω₀: evaluate the response on a grid from 0 to ω₀ and unwrap.
    const grid = freqz(b, a, 256)
    const upTo = grid.omega.map((w) => (w * w0) / Math.PI)
    const phases = upTo.map((w) => {
      const H = (coef: number[]) =>
        coef.reduce((acc, c, k) => [acc[0] + c * Math.cos(w * k), acc[1] - c * Math.sin(w * k)], [0, 0])
      const [nr, ni] = H(b)
      const [dr, di] = H(a)
      return Math.atan2(ni * dr - nr * di, nr * dr + ni * di)
    })
    const phase = unwrap(phases)[phases.length - 1]
    return { x, y, env, tg, tp: -phase / w0, measured: peak - CENTRE }
  }, [b, a, carrier.value])

  const n = Array.from({ length: LENGTH }, (_, i) => i)
  const series: XYSeries[] = [
    { name: 'input burst', type: 'line', x: n, y: r.x, muted: true },
    { name: 'output', type: 'line', x: n, y: r.y, slot: 0 },
    { name: 'output envelope', type: 'line', x: n, y: r.env, slot: 1, dashed: true },
  ]

  return (
    <Interactive
      title="Group delay moves the envelope"
      caption="A Gaussian tone burst at carrier frequency ω₀ passes through a filter. The envelope of the output is delayed by the group delay −dφ/dω at ω₀; the carrier inside it is shifted by the phase delay −φ/ω₀. For the linear-phase FIR filter the two are equal, 20 samples at every frequency. For the Butterworth filter the group delay depends on ω₀ and grows sharply near its cutoff at 0.35π, so different frequencies arrive at different times."
      controls={
        <>
          <ParamChoice
            label="filter"
            value={key}
            onChange={setKey}
            options={(Object.keys(FILTERS) as Key[]).map((k) => ({ value: k, label: FILTERS[k].label }))}
          />
          <ParamSlider label="carrier ω₀ (×π)" param={carrier} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="group delay at ω₀" value={`${formatNumber(r.tg)} samples`} />
          <Readout label="phase delay at ω₀" value={`${formatNumber(r.tp)} samples`} />
          <Readout label="measured envelope delay" value={`${r.measured} samples`} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="n"
        yLabel="amplitude"
        xRange={[0, LENGTH - 1]}
        yRange={[-1.1, 1.1]}
        height={300}
      />
    </Interactive>
  )
}
