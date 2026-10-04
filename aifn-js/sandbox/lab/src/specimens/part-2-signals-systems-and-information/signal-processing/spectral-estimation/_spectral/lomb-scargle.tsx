import { useMemo } from 'react'
import { stream } from 'aifn/foundation/random'
import { fromData, toFlat } from 'aifn/foundation/tensor'
import { unevenSinusoids, type UnevenSampling } from 'aifn-methods/data/signals'
import type { SpectralTruth } from 'aifn-methods/data'
import {
  falseAlarmLevel,
  falseAlarmProbability,
  gridSamples,
  lombScargle,
  lombScargleFrequencies,
  periodogram,
  spectralWindow,
} from 'aifn/signal/spectral'
import { Figure } from '@lab/layout'
import { choice, row, slider, toggle, useComputed, useFigureState } from '@lab/state'
import { Annotation, Curve, Handle, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'

// ---------------------------------------------------------------------------------------------------------------------
// Lomb–Scargle: spectra from uneven sampling.

const SAMPLING = [
  { value: 'random', label: 'random times' },
  { value: 'gaps', label: 'random, with gaps' },
  { value: 'seasonal', label: 'nightly and seasonal (astronomy)' },
] as const
const FMAX = 2.5

export function LombScargleSpecimen() {
  const state = useFigureState({
    data: row('1 · sampling', {
      sampling: choice(SAMPLING, 'random', { label: 'pattern' }),
      n: slider(20, 400, 120, { label: 'samples n', step: 5 }),
      gapFraction: slider(0, 0.8, 0.3, { label: 'gap fraction (clouded nights)', step: 0.05 }),
      noise: slider(0.05, 3, 0.7, { label: 'noise sd', step: 0.05 }),
    }),
    f0: slider(0.02, 2.4, 0.9, { label: 'true frequency f₀ (cycles per day)', step: 0.001 }),
    method: row('2 · estimates', {
      ls: choice(
        [
          { value: 'classic', label: 'classic' },
          { value: 'floating-mean', label: 'floating mean' },
          { value: 'generalised', label: 'generalised (weighted)' },
        ],
        'floating-mean',
        { label: 'Lomb–Scargle' },
      ),
      grid: choice(
        [
          { value: 'linear', label: 'linearly interpolated' },
          { value: 'zero-fill', label: 'zero-filled' },
        ],
        'linear',
        { label: 'regular-grid periodogram of' },
      ),
      heteroscedastic: toggle(false, 'unequal error bars'),
    }),
    reveal: row('3 · reveal', { fap: toggle(true, '1% false-alarm level'), aliases: toggle(true, 'alias positions') }),
  })
  const { sampling, n, gapFraction, noise } = state.data
  const { ls: method, grid, heteroscedastic } = state.method
  const f0 = state.f0
  // Seasonal sampling needs years to show its yearly alias.
  const span = sampling === 'seasonal' ? 730 : 100

  const data = useMemo(
    () =>
      unevenSinusoids(stream('lomb-scargle'), {
        n,
        span,
        sampling: sampling as UnevenSampling,
        gapFraction,
        frequency: f0,
        noise,
        heteroscedastic,
      }),
    [n, span, sampling, gapFraction, f0, noise, heteroscedastic],
  )
  const truth = data.meta.truth as SpectralTruth
  const t = useMemo(() => toFlat(data.x), [data])
  const y = useMemo(() => toFlat(data.signal.data), [data])
  const dy = useMemo(() => (data.dy ? toFlat(data.dy) : undefined), [data])

  const spectra = useComputed(() => {
    const freqs = lombScargleFrequencies(t, { samplesPerPeak: 4, maximum: FMAX })
    const ls = lombScargle(t, y, freqs, { method, dy })
    const window = spectralWindow(t, freqs)
    const level = method === 'classic' ? NaN : falseAlarmLevel(0.01, ls)
    const f = toFlat(ls.f)
    const p = toFlat(ls.values)
    const best = p.indexOf(Math.max(...p))
    // A regular grid fine enough for every frequency shown: spacing 1/(2 FMAX).
    const regular = gridSamples(t, y, { dt: 1 / (2 * FMAX), method: grid })
    const pg = periodogram(regular, { detrend: 'constant' })
    const pv = toFlat(pg.values)
    const pmax = Math.max(...pv.slice(1))
    const pf = toFlat(pg.f)
    const gridBest = pf[pv.indexOf(pmax)]
    return {
      f,
      power: p,
      window: toFlat(window.values),
      level,
      peak: f[best],
      fap: method === 'classic' ? NaN : falseAlarmProbability(p[best], ls),
      pf,
      pg: pv.map((v) => v / pmax),
      gridBest,
    }
  }, [t, y, dy, method, grid])
  const s = spectra.value
  const T = Math.max(...t) - Math.min(...t)
  const isTrue = (f: number) => Math.abs(f - f0) < 1 / T
  const aliasStep = sampling === 'seasonal' ? 1 : null
  const aliases = aliasStep ? [f0 - aliasStep, f0 + aliasStep, aliasStep - f0].filter((v) => v > 0 && v < FMAX) : []

  const time = useAxis({ label: 'time (days)', key: span })
  const value = useAxis({ label: 'y', hold: 'initial' })
  const freq = useAxis({ label: 'frequency (cycles per day)', range: [0, FMAX] })
  const lsAxis = useAxis({ label: 'Lomb–Scargle power', range: [0, 1] })
  const winAxis = useAxis({ label: 'spectral window', range: [0, 1.05] })
  const pgAxis = useAxis({ label: 'periodogram / peak', range: [0, 1.05] })
  const curve = useMemo(() => {
    const lo = Math.min(...t)
    const ts = Array.from({ length: 1200 }, (_, i) => lo + (i * T) / 1199)
    return { t: ts, y: toFlat(truth.expect(fromData(Float64Array.from(ts)))) }
  }, [t, T, truth])

  return (
    <Figure
      title="Lomb–Scargle: spectra from uneven sampling"
      purpose="Fitting a sinusoid at each frequency needs no regular grid, so the Lomb–Scargle periodogram finds the tone where an interpolated periodogram smears it; the sampling pattern still creates aliases, read off the spectral window."
      state={state}
      defaultSize="XL"
      readouts={{
        'Lomb–Scargle': (
          <>
            <Readout
              label="highest peak"
              value={`${s.peak.toFixed(4)} (${isTrue(s.peak) ? 'true' : 'alias or noise'})`}
            />
            <Readout
              label="false-alarm probability"
              value={
                Number.isFinite(s.fap)
                  ? s.fap < 1e-4
                    ? s.fap.toExponential(1)
                    : s.fap.toFixed(4)
                  : '— (needs a floating mean)'
              }
            />
          </>
        ),
        'regular grid': (
          <>
            <Readout
              label="highest peak"
              value={`${s.gridBest.toFixed(4)} (${isTrue(s.gridBest) ? 'true' : 'wrong'})`}
            />
          </>
        ),
        sampling: (
          <>
            <Readout label="samples" value={t.length} />
            <Readout label="span T" value={`${T.toFixed(0)} days`} />
            <Readout label="mean rate" value={`${(t.length / T).toFixed(2)} per day`} />
            <Readout label="peak width ≈ 1/T" value={(1 / T).toFixed(4)} />
          </>
        ),
      }}
      caption="aifn/signal/spectral lombScargle (with Baluev's falseAlarmLevel), spectralWindow and gridSamples + periodogram, on aifn-methods/data/signals unevenSinusoids. Drag f₀ on the Lomb–Scargle panel. Random sampling has no hard Nyquist limit: Lomb–Scargle finds f₀ well above the mean rate's half, while linear interpolation acts as a low-pass and the zero-filled grid inherits the sampling's sidelobes. Switch to nightly and seasonal sampling: the spectral window peaks at 1 cycle per day, so f₀ reappears at |1 − f₀| and 1 + f₀ (dashed), sometimes higher than the true peak."
    >
      <Plots rows={2} cols={2} widths={[1.4, 1]} heights={[1, 1.3]}>
        <Plot x={time} y={value}>
          <Curve name="true signal" x={curve.t} y={curve.y} emphasis thin />
          <Points name="samples" x={t} y={y} slot={0} size={4} />
        </Plot>
        <Plot x={freq} y={winAxis}>
          <Curve name="spectral window" x={s.f} y={s.window} slot={2} stale={spectra.stale} />
        </Plot>
        <Plot x={freq} y={lsAxis}>
          <Curve name={`Lomb–Scargle (${method})`} x={s.f} y={s.power} slot={0} stale={spectra.stale} />
          {state.reveal.fap && Number.isFinite(s.level) && <Annotation y={s.level} dashed text="FAP 1%" />}
          {state.reveal.aliases &&
            aliases.map((a, i) => <Annotation key={i} x={a} dashed text={i === 0 ? 'aliases' : undefined} />)}
          <Handle {...state.handle('f0', { label: 'f₀' })} />
        </Plot>
        <Plot x={freq} y={pgAxis}>
          <Curve name={`periodogram (${grid})`} x={s.pf} y={s.pg} slot={1} stale={spectra.stale} />
          <Annotation x={f0} dashed text="f₀" />
        </Plot>
      </Plots>
    </Figure>
  )
}
