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
import { firLowpass } from '../_shared/design'

type Win = 'rectangular' | 'hann' | 'hamming' | 'blackman' | 'kaiser'

/**
 * Window-method low-pass design: the ideal sinc response, truncated to M + 1 taps and multiplied by a window. Drag the
 * cutoff; the window sets the stopband level, the length sets the transition width.
 */
export function FirDesignExplorer() {
  const cutoff = useParam(0.3, { min: 0.05, max: 0.9, step: 0.005 })
  const taps = useParam(41, { min: 5, max: 201, step: 2 })
  const beta = useParam(5.65, { min: 0, max: 12, step: 0.05 })
  const [win, setWin] = useState<Win>('hamming')

  const r = useMemo(() => {
    const h = firLowpass(taps.value, cutoff.value * Math.PI, win === 'kaiser' ? { kaiser: beta.value } : win)
    const f = freqz(h, [1], 2048)
    const mag = f.magnitude
    // Peak stopband level: the largest gain beyond the first null above the cutoff.
    const i0 = f.omega.findIndex((w) => w >= cutoff.value * Math.PI)
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
  }, [cutoff.value, taps.value, beta.value, win])

  const handles: Handle[] = [{ kind: 'x', at: cutoff.value, label: 'cutoff', onDrag: (x) => cutoff.set(x) }]
  const response: XYSeries[] = [{ name: '|H| (dB)', type: 'line', x: r.x, y: r.y, slot: 0 }]
  const impulse: XYSeries[] = [{ name: 'h[n]', type: 'bar', x: r.h.map((_, n) => n), y: r.h, slot: 0 }]

  return (
    <Interactive
      title="Designing an FIR low-pass filter with a window"
      caption="The ideal low-pass impulse response is a sinc, infinitely long. The window method keeps M + 1 samples of it, centred, and multiplies them by a window. Drag the cutoff. The window fixes the stopband level almost regardless of length: about −21 dB for rectangular, −44 dB for Hann, −53 dB for Hamming, −75 dB for Blackman, and a β-controlled trade-off for Kaiser. The length sets the width of the transition band. The gain at the cutoff is always close to −6 dB, half amplitude."
      controls={
        <>
          <ParamChoice
            label="window"
            value={win}
            onChange={setWin}
            options={[
              { value: 'rectangular', label: 'rect' },
              { value: 'hann', label: 'Hann' },
              { value: 'hamming', label: 'Hamming' },
              { value: 'blackman', label: 'Blackman' },
              { value: 'kaiser', label: 'Kaiser' },
            ]}
          />
          <ParamSlider label="taps M + 1" param={taps} format={(v) => String(v)} withArrows />
          <ParamSlider label="cutoff ω_c (×π)" param={cutoff} />
          {win === 'kaiser' && <ParamSlider label="Kaiser β" param={beta} />}
        </>
      }
      readout={
        <>
          <Readout label="gain at cutoff" value={`${formatNumber(r.cutoffDb)} dB`} />
          <Readout label="peak stopband level" value={`${formatNumber(r.stopDb)} dB`} />
          <Readout label="first null (×π)" value={formatNumber(r.nullAt)} />
          <Readout label="delay (samples)" value={(taps.value - 1) / 2} />
        </>
      }
    >
      <XYChart
        series={response}
        xLabel="ω / π"
        yLabel="magnitude (dB)"
        xRange={[0, 1]}
        yRange={[-120, 5]}
        handles={handles}
        height={300}
      />
      <XYChart series={impulse} xLabel="n" yLabel="h[n]" height={180} />
    </Interactive>
  )
}
