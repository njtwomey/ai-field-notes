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
import { melFilterBank, type MelNorm } from '../_shared/audio'

const FS = 16000
const NFFT = 512

/** Triangular filters with edges equally spaced in mel, in the HTK (peak 1) or Slaney (unit area) normalisation. */
export function MelBank() {
  const state = useFigureState({
    count: int(20, { min: 4, max: 80, step: 1, label: 'number of filters' }),
    fMax: int(8000, { min: 2000, max: 8000, step: 250, label: 'top frequency (Hz)' }),
    norm: choice<MelNorm>(
      [
        { value: 'htk', label: 'HTK (peak 1)' },
        { value: 'slaney', label: 'Slaney (unit area)' },
      ],
      'htk',
      { label: 'normalisation' },
    ),
  })

  const r = useMemo(
    () => melFilterBank(state.count, NFFT, FS, 0, state.fMax, state.norm),
    [state.count, state.fMax, state.norm],
  )

  // Draw every filter muted, and a few highlighted ones in colour so their shapes read clearly.
  const highlight = [0, Math.floor(state.count / 2), state.count - 1]
  const series: SeriesSpec[] = r.filters.map((w, m) => {
    const slot = highlight.indexOf(m)
    return {
      name: slot >= 0 ? `filter ${m + 1}` : 'other filters',
      type: 'line',
      x: r.bins,
      y: w,
      ...(slot >= 0 ? { slot } : { muted: true }),
    }
  })
  const centre = (m: number) => r.edges[m + 1]
  const width = (m: number) => r.edges[m + 2] - r.edges[m]

  const xAxis = useAxis({ label: 'frequency (Hz)', range: [0, FS / 2] })
  const yAxis = useAxis({ label: 'weight', hold: 'union' })
  return (
    <Figure
      title="A mel filter bank"
      state={state}
      caption="Triangular filters at 16 kHz with a 512-point FFT, edges equally spaced in mel between 0 Hz and the top frequency. Each filter rises from the previous filter's centre to its own and falls to the next one's, so neighbours overlap by half. Low filters are narrow and closely spaced; high ones wide. HTK normalisation gives every filter a peak of 1, so wide filters collect more energy; Slaney normalisation scales each to unit area, so every filter has equal weight for white noise. With many filters, the lowest ones may cover fewer than one FFT bin."

      readouts={
        <>
          <Readout label="first centre" value={`${formatNumber(centre(0))} Hz`} />
          <Readout label="first width" value={`${formatNumber(width(0))} Hz`} />
          <Readout label="last centre" value={`${formatNumber(centre(state.count - 1))} Hz`} />
          <Readout label="last width" value={`${formatNumber(width(state.count - 1))} Hz`} />
          <Readout label="FFT bin spacing" value={`${formatNumber(FS / NFFT)} Hz`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
