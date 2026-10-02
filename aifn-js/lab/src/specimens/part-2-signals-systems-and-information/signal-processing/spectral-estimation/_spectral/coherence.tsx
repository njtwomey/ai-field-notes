import { useMemo } from 'react'
import { stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { coupledProcesses } from 'aifn-applied/data/signals'
import type { SpectralTruth } from 'aifn-applied/data'
import { coherence, coherenceThreshold, crossSpectralDelay, csd, welch } from 'aifn/signal/spectral'
import { magnitude, phase } from 'aifn/signal'
import { Figure } from '@lab/layout'
import { choice, row, slider, toggle, useFigureState } from '@lab/state'
import { Annotation, Curve, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'

// ---------------------------------------------------------------------------------------------------------------------
// Coherence between two signals.

const db = (v: number) => 10 * Math.log10(Math.max(v, 1e-30))
export function CoherenceSpecimen() {
  const state = useFigureState({
    input: row('1 · input x (AR(2))', {
      radius: slider(0, 0.98, 0.9, { label: 'pole radius', step: 0.01 }),
      frequency: slider(0.02, 0.45, 0.1, { label: 'pole frequency', step: 0.005 }),
      n: choice([512, 1024, 2048, 4096, 8192], 2048, { label: 'samples n' }),
    }),
    coupling: row('2 · y = gain · lowpass(x delayed) + v', {
      delay: slider(0, 30, 5, { label: 'delay (samples)', step: 1 }),
      gain: slider(0, 3, 1, { label: 'gain', step: 0.05 }),
      noise: slider(0, 5, 1, { label: 'noise sd of v', step: 0.05 }),
    }),
    welch: row('3 · Welch averaging', {
      nperseg: choice([16, 32, 64, 128, 256, 512, 1024], 128, { label: 'segment length L' }),
      overlap: choice([0, 50], 50, { label: 'overlap %' }),
    }),
    reveal: row('4 · reveal', { truth: toggle(true, 'true values'), threshold: toggle(true, '95% null threshold') }),
  })
  const { radius, frequency, n } = state.input
  const { delay, gain, noise } = state.coupling
  const { nperseg, overlap } = state.welch
  const data = useMemo(
    () => coupledProcesses(stream('coherence'), { n, radius, frequency, delay, gain, noise }),
    [n, radius, frequency, delay, gain, noise],
  )
  const truth = data.meta.truth as SpectralTruth
  const est = useMemo(() => {
    const x = data.signal
    const y = data.partner!
    const o = { nperseg: Math.min(nperseg, n), noverlap: Math.floor((Math.min(nperseg, n) * overlap) / 100) }
    const pxy = csd(x, y, o)
    const c = coherence(x, y, o)
    const f = toFlat(c.f)
    const est = {
      f,
      pxx: toFlat(welch(x, o).values).map(db),
      pyy: toFlat(welch(y, o).values).map(db),
      mag: toFlat(magnitude(pxy)).map(db),
      phase: toFlat(phase(pxy)),
      coh: toFlat(c.values),
      segments: c.segments,
      delay: crossSpectralDelay(pxy, c, { minCoherence: 0.5 }),
    }
    const coupled = truth.coupled!
    const tc = toFlat(coupled.crossSpectrum(f))
    const trueMag = f.map((_, i) => db(Math.hypot(tc[2 * i], tc[2 * i + 1])))
    const truePhase = f.map((_, i) => Math.atan2(tc[2 * i + 1], tc[2 * i]))
    // Break the wrapped true phase where it jumps by 2π.
    const tp = truePhase.map((v, i) => (i > 0 && Math.abs(v - truePhase[i - 1]) > Math.PI ? NaN : v))
    const trueCoh = toFlat(coupled.coherence(f))
    const err = est.coh.reduce((a, v, i) => a + Math.abs(v - trueCoh[i]), 0) / f.length
    // Display only: H has a zero at Nyquist, so clip magnitudes 60 dB below the true peak.
    const floor = Math.max(...trueMag.filter(Number.isFinite)) - 60
    const clip = (v: number) => Math.max(v, floor)
    return {
      ...est,
      mag: est.mag.map(clip),
      truePxx: toFlat(truth.psd(f)).map(db),
      truePyy: toFlat(coupled.psd(f)).map(db),
      trueMag: trueMag.map(clip),
      truePhase: tp,
      trueCoh,
      err,
    }
  }, [data, truth, nperseg, overlap, n])
  // Overlapping segments are correlated: count them as about half as many independent ones at 50%.
  const kEff = overlap === 50 ? Math.max(2, est.segments / 1.9) : est.segments
  const threshold = coherenceThreshold(kEff)

  const freq = useAxis({ label: 'frequency (cycles per sample)', range: [0, 0.5] })
  const cohAxis = useAxis({ label: 'magnitude-squared coherence', range: [0, 1.02] })
  const phaseAxis = useAxis({ label: 'phase of S_xy (rad)', range: [-Math.PI, Math.PI] })
  const magAxis = useAxis({ label: '|S_xy| (dB)', hold: 'initial', key: `${radius}-${gain}` })
  const psdAxis = useAxis({ label: 'PSD (dB)', hold: 'initial', key: `${radius}-${gain}-${noise}` })

  return (
    <Figure
      title="Coherence between two signals"
      purpose="The cross-spectrum's phase is the coupling filter's phase (a straight line whose slope is the delay), and the coherence is the share of y's power at each frequency that x explains linearly; both need averaging over segments, and one segment gives coherence 1 everywhere."
      state={state}
      defaultSize="XL"
      readouts={{
        averaging: (
          <>
            <Readout label="segments K" value={est.segments} />
            <Readout label="95% null threshold" value={threshold.toFixed(3)} />
            <Readout label="mean |Ĉ − C|" value={est.err.toFixed(3)} />
          </>
        ),
        delay: (
          <>
            <Readout
              label="from the phase slope"
              value={
                Number.isFinite(est.delay.delay)
                  ? `${est.delay.delay.toFixed(2)} samples (${est.delay.bins} bins)`
                  : '—'
              }
            />
            <Readout label="true group delay" value={`${delay + 1} samples`} />
          </>
        ),
      }}
      caption="aifn/signal/spectral csd, coherence, welch, coherenceThreshold and crossSpectralDelay on aifn-applied/data/signals coupledProcesses, whose truth gives S_xy = H S_xx and C = |H|²S_xx/(|H|²S_xx + S_vv) exactly (ink); magnitudes are clipped 60 dB below the true peak, since the filter's zero at Nyquist sends |S_xy| to −∞. The filter is gain·[¼, ½, ¼] after the delay, so the true group delay is delay + 1 samples. Shorten the segments: estimates settle but blur the peak; lengthen them: the coherence estimate is biased up towards 1 and noisy, because fewer segments are averaged. The dashed level is the coherence two independent signals exceed 5% of the time with this many segments (50% overlap counted as about K/1.9 independent ones)."
    >
      <Plots rows={2} cols={2} hoverGroup>
        <Plot x={freq} y={cohAxis}>
          <Curve name="estimated coherence" x={est.f} y={est.coh} slot={0} />
          {state.reveal.truth && <Curve name="true coherence" x={est.f} y={est.trueCoh} emphasis />}
          {state.reveal.threshold && <Annotation y={threshold} dashed text="null 95%" />}
        </Plot>
        <Plot x={freq} y={phaseAxis}>
          <Points name="estimated phase" x={est.f} y={est.phase} slot={0} size={3} />
          {state.reveal.truth && <Curve name="true phase −2πf(delay + 1)" x={est.f} y={est.truePhase} emphasis />}
        </Plot>
        <Plot x={freq} y={magAxis}>
          <Curve name="estimated |S_xy|" x={est.f} y={est.mag} slot={0} />
          {state.reveal.truth && <Curve name="true |S_xy|" x={est.f} y={est.trueMag} emphasis />}
        </Plot>
        <Plot x={freq} y={psdAxis}>
          <Curve name="S_xx estimate" x={est.f} y={est.pxx} slot={1} />
          <Curve name="S_yy estimate" x={est.f} y={est.pyy} slot={2} />
          {state.reveal.truth && <Curve name="true S_xx" x={est.f} y={est.truePxx} emphasis dashed />}
          {state.reveal.truth && <Curve name="true S_yy" x={est.f} y={est.truePyy} emphasis />}
        </Plot>
      </Plots>
    </Figure>
  )
}
