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
import { db, freqz, lfilter, unwrap } from '@/lib/dsp'
import { iirLowpass } from '../_shared/design'

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
  const [preset, setPreset] = useState<Preset>('butter')
  const w0 = useParam(0.2, { min: 0.01, max: 0.99, step: 0.01 })
  const { b, a } = PRESETS[preset]

  const response = useMemo(() => {
    const f = freqz(b, a, 512)
    return {
      x: f.omega.map((w) => w / Math.PI),
      mag: f.magnitude.map((m) => db(m, -80)),
      phase: unwrap(f.phase),
    }
  }, [b, a])

  const at = useMemo(() => {
    const omega = w0.value * Math.PI
    // H(e^{iω₀}) = B(e^{iω₀}) / A(e^{iω₀}), evaluated directly.
    const H = (coef: number[]) =>
      coef.reduce((acc, c, k) => [acc[0] + c * Math.cos(omega * k), acc[1] - c * Math.sin(omega * k)], [0, 0])
    const [nr, ni] = H(b)
    const [dr, di] = H(a)
    const d = dr * dr + di * di
    const re = (nr * dr + ni * di) / d
    const im = (ni * dr - nr * di) / d
    const input = Array.from({ length: T0 + SHOW }, (_, n) => Math.cos(omega * n))
    const output = lfilter(b, a, input)
    const n = Array.from({ length: SHOW }, (_, i) => T0 + i)
    return {
      gain: Math.hypot(re, im),
      phase: Math.atan2(im, re),
      n,
      input: n.map((k) => input[k]),
      output: n.map((k) => output[k]),
    }
  }, [b, a, w0.value])

  const handles: Handle[] = [{ kind: 'x', at: w0.value, label: 'ω₀', onDrag: (x) => w0.set(x) }]
  const magSeries: XYSeries[] = [{ name: '|H| (dB)', type: 'line', x: response.x, y: response.mag, slot: 0 }]
  const phaseSeries: XYSeries[] = [{ name: '∠H (unwrapped)', type: 'line', x: response.x, y: response.phase, slot: 0 }]
  const timeSeries: XYSeries[] = [
    { name: 'input cos(ω₀n)', type: 'line', x: at.n, y: at.input, slot: 1, dashed: true },
    { name: 'output (steady state)', type: 'line', x: at.n, y: at.output, slot: 0 },
  ]

  return (
    <Interactive
      title="What a frequency response means"
      caption="A sinusoid passes through a linear time-invariant filter as a sinusoid of the same frequency, scaled by |H(e^{iω₀})| and shifted by ∠H(e^{iω₀}). Drag ω₀ along the magnitude response and compare the input (dashed) with the output after the start-up transient has died away. The phase delay −∠H/ω₀ is the shift in samples."
      controls={
        <>
          <ParamChoice
            label="filter"
            value={preset}
            onChange={setPreset}
            options={(Object.keys(PRESETS) as Preset[]).map((k) => ({ value: k, label: PRESETS[k].label }))}
          />
          <ParamSlider label="test frequency ω₀ (×π)" param={w0} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="|H(e^{iω₀})|" value={formatNumber(at.gain)} />
          <Readout label="gain (dB)" value={formatNumber(20 * Math.log10(at.gain))} />
          <Readout label="∠H (rad)" value={formatNumber(at.phase)} />
          <Readout label="phase delay (samples)" value={formatNumber(-at.phase / (w0.value * Math.PI))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={magSeries}
          xLabel="ω / π"
          yLabel="magnitude (dB)"
          xRange={[0, 1]}
          yRange={[-80, 5]}
          handles={handles}
          height={240}
        />
        <XYChart series={phaseSeries} xLabel="ω / π" yLabel="phase (rad)" xRange={[0, 1]} height={240} />
      </div>
      <XYChart
        series={timeSeries}
        xLabel="n"
        yLabel="amplitude"
        xRange={[T0, T0 + SHOW - 1]}
        yRange={[-1.2, 1.2]}
        height={220}
      />
    </Interactive>
  )
}
