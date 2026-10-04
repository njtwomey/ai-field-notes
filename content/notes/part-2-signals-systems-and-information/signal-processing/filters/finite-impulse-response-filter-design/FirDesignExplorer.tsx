import { useMemo } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { db, firLowpass, response as freqz } from '../_shared/design'

type Win = 'rectangular' | 'hann' | 'hamming' | 'blackman' | 'kaiser'

/**
 * Window-method low-pass design: the ideal sinc response, truncated to M + 1 taps and multiplied by a window. Drag the
 * cutoff; the window sets the stopband level, the length sets the transition width.
 */
export function FirDesignExplorer() {
  const state = useFigureState({
    win: choice<Win>(
      [
        { value: 'rectangular', label: 'rect' },
        { value: 'hann', label: 'Hann' },
        { value: 'hamming', label: 'Hamming' },
        { value: 'blackman', label: 'Blackman' },
        { value: 'kaiser', label: 'Kaiser' },
      ],
      'hamming',
      { label: 'window' },
    ),
    taps: int(41, { min: 5, max: 201, step: 2, label: 'taps M + 1', format: (v) => String(v) }),
    cutoff: slider(0.05, 0.9, 0.3, { step: 0.005, label: 'cutoff ω_c (×π)' }),
    beta: float(5.65, { min: 0, max: 12, step: 0.05, label: 'Kaiser β', when: (v) => v.win === 'kaiser' }),
  })

  const r = useMemo(() => {
    const h = firLowpass(
      state.taps,
      state.cutoff * Math.PI,
      state.win === 'kaiser' ? { kaiser: state.beta } : state.win,
    )
    const f = freqz(h, [1], 2048)
    const mag = f.magnitude
    // Peak stopband level: the largest gain beyond the first null above the cutoff.
    const i0 = f.omega.findIndex((w) => w >= state.cutoff * Math.PI)
    let j = i0
    while (j + 1 < mag.length && mag[j + 1] <= mag[j]) j++
    const stop = j + 1 < mag.length ? Math.max(...mag.slice(j)) : 0
    const atCutoff = mag[i0]
    return {
      h,
      x: f.omega.map((w) => w / Math.PI),
      y: mag.map((m) => db(m, -120)),
      stopDb: db(stop, -120),
      cutoffDb: db(atCutoff, -120),
      nullAt: f.omega[j] / Math.PI,
    }
  }, [state.cutoff, state.taps, state.beta, state.win])

  const response = [{ name: '|H| (dB)', x: r.x, y: r.y, slot: 0 }] as const
  const impulse = [{ name: 'h[n]', x: r.h.map((_, n) => n), y: r.h, slot: 0 }] as const

  const xAxis = useAxis({ label: 'ω / π', range: [0, 1] })
  const yAxis = useAxis({ label: 'magnitude (dB)', range: [-120, 5] })
  const xAxis2 = useAxis({ label: 'n', hold: 'union' })
  const yAxis2 = useAxis({ label: 'h[n]', hold: 'union' })
  return (
    <Figure
      title="Designing an FIR low-pass filter with a window"
      state={state}
      caption="The ideal low-pass impulse response is a sinc, infinitely long. The window method keeps M + 1 samples of it, centred, and multiplies them by a window. Drag the cutoff. The window fixes the stopband level almost regardless of length: about −21 dB for rectangular, −44 dB for Hann, −53 dB for Hamming, −75 dB for Blackman, and a β-controlled trade-off for Kaiser. The length sets the width of the transition band. The gain at the cutoff is always close to −6 dB, half amplitude."

      readouts={
        <>
          <Readout label="gain at cutoff" value={`${formatNumber(r.cutoffDb)} dB`} />
          <Readout label="peak stopband level" value={`${formatNumber(r.stopDb)} dB`} />
          <Readout label="first null (×π)" value={formatNumber(r.nullAt)} />
          <Readout label="delay (samples)" value={(state.taps - 1) / 2} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...response[0]} />
        <Handle {...state.handle('cutoff', { label: 'cutoff' })} />
      </Plot>
      <Plot x={xAxis2} y={yAxis2} height={180}>
        <Bars {...impulse[0]} />
      </Plot>
    </Figure>
  )
}
