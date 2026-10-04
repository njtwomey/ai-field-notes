import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { hzToMel } from 'aifn/signal/audio'

const bark = (f: number) => 13 * Math.atan(0.00076 * f) + 3.5 * Math.atan((f / 7500) ** 2)
const erbNumber = (f: number) => 21.4 * Math.log10(1 + (4.37 * f) / 1000)
const erbWidth = (f: number) => 24.7 * (1 + (4.37 * f) / 1000)

type View = 'scales' | 'bandwidth'

/** Mel, Bark and ERB-number scales against frequency, each normalised to its value at the top frequency. */
export function ScaleComparison() {
  const state = useFigureState({
    view: choice<View>(
      [
        { value: 'scales', label: 'scales' },
        { value: 'bandwidth', label: 'bandwidths' },
      ],
      'scales',
      { label: 'show' },
    ),
    fMax: int(8000, { min: 1000, max: 20000, step: 500, label: 'top frequency (Hz)' }),
    probe: int(1000, { min: 20, max: 20000, step: 10, label: 'frequency to evaluate (Hz)' }),
  })

  const r = useMemo(() => {
    const f = Array.from({ length: 400 }, (_, i) => 20 + ((state.fMax - 20) * i) / 399)
    const norm = (g: (x: number) => number) => f.map((x) => g(x) / g(state.fMax))
    return { f, mel: norm((v: number) => hzToMel(v)), bark: norm(bark), erb: norm(erbNumber), width: f.map(erbWidth) }
  }, [state.fMax])

  const series: SeriesSpec[] =
    state.view === 'scales'
      ? [
          { name: 'mel', type: 'line', x: r.f, y: r.mel, slot: 0 },
          { name: 'Bark', type: 'line', x: r.f, y: r.bark, slot: 1 },
          { name: 'ERB number', type: 'line', x: r.f, y: r.erb, slot: 2 },
          { name: 'linear', type: 'line', x: [0, state.fMax], y: [0, 1], muted: true, dashed: true },
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
  const x = state.probe

  const xAxis = useAxis({ label: 'frequency (Hz)', range: [0, state.fMax] })
  const yAxis = useAxis({ label: state.view === 'scales' ? 'normalised scale value' : 'bandwidth (Hz)', hold: 'union' })
  return (
    <Figure
      title="Auditory frequency scales"
      state={state}
      caption="Scales: mel, Bark and ERB number against frequency, each divided by its value at the top of the range so their shapes can be compared. All three are roughly linear at low frequencies and logarithmic above about 1 kHz: equal steps on them are closer to equal perceptual steps than equal steps in Hz. Bandwidths: the equivalent rectangular bandwidth and Zwicker's critical bandwidth, the width of the ear's frequency channels, which grow with frequency."

      readouts={
        <>
          <Readout label="mel" value={formatNumber(hzToMel(x))} />
          <Readout label="Bark" value={formatNumber(bark(x))} />
          <Readout label="ERB number" value={formatNumber(erbNumber(x))} />
          <Readout label="ERB (Hz)" value={formatNumber(erbWidth(x))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
