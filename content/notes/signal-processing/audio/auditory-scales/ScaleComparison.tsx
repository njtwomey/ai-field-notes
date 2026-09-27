import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  ParamSlider,
  type XYSeries,
} from '@/components/viz'
import { hzToMel } from '@/lib/dsp'

const bark = (f: number) => 13 * Math.atan(0.00076 * f) + 3.5 * Math.atan((f / 7500) ** 2)
const erbNumber = (f: number) => 21.4 * Math.log10(1 + (4.37 * f) / 1000)
const erbWidth = (f: number) => 24.7 * (1 + (4.37 * f) / 1000)

type View = 'scales' | 'bandwidth'

/** Mel, Bark and ERB-number scales against frequency, each normalised to its value at the top frequency. */
export function ScaleComparison() {
  const [view, setView] = useState<View>('scales')
  const fMax = useParam(8000, { min: 1000, max: 20000, step: 500 })
  const probe = useParam(1000, { min: 20, max: 20000, step: 10 })

  const r = useMemo(() => {
    const f = Array.from({ length: 400 }, (_, i) => 20 + ((fMax.value - 20) * i) / 399)
    const norm = (g: (x: number) => number) => f.map((x) => g(x) / g(fMax.value))
    return { f, mel: norm(hzToMel), bark: norm(bark), erb: norm(erbNumber), width: f.map(erbWidth) }
  }, [fMax.value])

  const series: XYSeries[] =
    view === 'scales'
      ? [
          { name: 'mel', type: 'line', x: r.f, y: r.mel, slot: 0 },
          { name: 'Bark', type: 'line', x: r.f, y: r.bark, slot: 1 },
          { name: 'ERB number', type: 'line', x: r.f, y: r.erb, slot: 2 },
          { name: 'linear', type: 'line', x: [0, fMax.value], y: [0, 1], muted: true, dashed: true },
        ]
      : [
          { name: 'ERB bandwidth (Hz)', type: 'line', x: r.f, y: r.width, slot: 2 },
          {
            name: 'critical bandwidth, Zwicker (Hz)',
            type: 'line',
            x: r.f,
            y: r.f.map((x) => 25 + 75 * (1 + 1.4 * (x / 1000) ** 2) ** 0.69),
            slot: 1,
          },
        ]
  const x = probe.value

  return (
    <Interactive
      title="Auditory frequency scales"
      caption="Scales: mel, Bark and ERB number against frequency, each divided by its value at the top of the range so their shapes can be compared. All three are roughly linear at low frequencies and logarithmic above about 1 kHz: equal steps on them are closer to equal perceptual steps than equal steps in Hz. Bandwidths: the equivalent rectangular bandwidth and Zwicker's critical bandwidth, the width of the ear's frequency channels, which grow with frequency."
      controls={
        <>
          <ParamChoice
            label="show"
            value={view}
            onChange={setView}
            options={[
              { value: 'scales', label: 'scales' },
              { value: 'bandwidth', label: 'bandwidths' },
            ]}
          />
          <ParamSlider label="top frequency (Hz)" param={fMax} />
          <ParamSlider label="frequency to evaluate (Hz)" param={probe} />
        </>
      }
      readout={
        <>
          <Readout label="mel" value={formatNumber(hzToMel(x))} />
          <Readout label="Bark" value={formatNumber(bark(x))} />
          <Readout label="ERB number" value={formatNumber(erbNumber(x))} />
          <Readout label="ERB (Hz)" value={formatNumber(erbWidth(x))} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="frequency (Hz)"
        yLabel={view === 'scales' ? 'normalised scale value' : 'bandwidth (Hz)'}
        xRange={[0, fMax.value]}
        height={320}
      />
    </Interactive>
  )
}
