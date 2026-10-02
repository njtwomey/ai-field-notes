import { useMemo } from 'react'
import { stream } from 'aifn/foundation/random'
import { imagPart, realPart, toFlat } from 'aifn/foundation/tensor'
import { roots } from 'aifn/numerics/polynomial'
import type { Spectrum } from 'aifn/foundation/contracts'
import { arProcess, sinusoidsInNoise, type SignalDataset } from 'aifn-applied/data/signals'
import type { SpectralTruth } from 'aifn-applied/data'
import {
  bartlett,
  blackmanTukey,
  logSpectralError,
  multitaper,
  peakDip,
  periodogram,
  replicateSpectralError,
  spectralConfidence,
  welch,
} from 'aifn/signal/spectral'
import { arPsd, esprit, music } from 'aifn/signal/statistical'
import type { WindowSpec } from 'aifn/signal/windows'
import { Button } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { choice, row, slider, toggle, useComputed, useFigureState, variants } from '@lab/state'
import { Annotation, Area, Bars, Curve, Handle, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'

// ---------------------------------------------------------------------------------------------------------------------
// Spectral estimation: estimates against the true spectrum.

/** Percival and Walden's AR(4) (1993, eq. 46a): two pairs of poles near the unit circle, a 65 dB dynamic range. */
const PW_AR4 = [2.7607, -3.8106, 2.6535, -0.9238]
const WINDOWS = [
  { value: 'boxcar', label: 'rectangular' },
  { value: 'hann', label: 'Hann' },
  { value: 'hamming', label: 'Hamming' },
  { value: 'blackman', label: 'Blackman' },
  { value: 'blackmanharris', label: 'Blackman–Harris' },
] as const
const SEGMENTS = [16, 32, 64, 128, 256, 512, 1024]
const LAGS = [4, 8, 16, 32, 64, 128, 256]
const NWS = [1, 1.5, 2, 3, 4, 6, 8]
const ORDERS = [1, 2, 4, 8, 16, 32]
const REPLICATES = 12
const CIRCLE = (() => {
  const a = Array.from({ length: 121 }, (_, i) => (2 * Math.PI * i) / 120)
  return { x: a.map(Math.cos), y: a.map(Math.sin) }
})()
const db = (v: number) => 10 * Math.log10(Math.max(v, 1e-30))

type Process =
  | { key: 'ar2'; re: number; im: number }
  | { key: 'ar4' }
  | { key: 'tones'; f1: number; gap: number; level: number; noise: number }

/** The pole of the AR(2) case, held inside radius 0.99 so the process stays stationary. */
function pole(re: number, im: number) {
  const r = Math.hypot(re, im)
  const scale = r > 0.99 ? 0.99 / r : 1
  return { radius: r * scale, frequency: Math.atan2(im, re) / (2 * Math.PI) }
}

/** One realisation of the chosen process (seeded), from the registered generators of aifn-applied/data/signals. */
function draw(p: Process, n: number, seed: string | number): SignalDataset {
  const s = stream(`spectral-estimation-${seed}`)
  if (p.key === 'ar2') return arProcess(s, { n, ...pole(p.re, p.im) })
  if (p.key === 'ar4') return arProcess(s, { n, ar: PW_AR4 })
  return sinusoidsInNoise(s, { n, f1: p.f1, a1: 1, f2: p.f1 + p.gap, a2: 10 ** (p.level / 20), noise: p.noise })
}

type Estimator =
  | { key: 'periodogram'; window: string }
  | { key: 'welch'; nperseg: number; overlap: number; window: string }
  | { key: 'bartlett'; nperseg: number }
  | { key: 'blackmanTukey'; maxLag: number; lagWindow: string }
  | { key: 'multitaper'; nw: number; tapers: number; adaptive: boolean }
  | { key: 'burg'; order: number; method: string }
  | { key: 'music'; sinusoids: number; order: number }

/** An estimate with its degrees of freedom (none for parametric estimates), on cycles per sample. */
function estimate(
  e: Estimator,
  x: Float64Array,
): { s: Spectrum; dof?: number | Float64Array; note: string; lineFreqs?: number[] } {
  const n = x.length
  switch (e.key) {
    case 'periodogram': {
      const s = periodogram(x, { window: e.window as WindowSpec })
      return { s, dof: s.dof, note: `bin 1/n = ${(1 / n).toFixed(4)}` }
    }
    case 'welch': {
      const nperseg = Math.min(e.nperseg, n)
      const s = welch(x, { nperseg, noverlap: Math.floor((nperseg * e.overlap) / 100), window: e.window as WindowSpec })
      return { s, dof: s.dof, note: `${s.segments} segments, bin 1/L = ${(1 / nperseg).toFixed(4)}` }
    }
    case 'bartlett': {
      const s = bartlett(x, { nperseg: Math.min(e.nperseg, n) })
      return { s, dof: s.dof, note: `${s.segments} segments, bin 1/L = ${(1 / Math.min(e.nperseg, n)).toFixed(4)}` }
    }
    case 'blackmanTukey': {
      const s = blackmanTukey(x, { maxLag: Math.min(e.maxLag, n - 1), lagWindow: e.lagWindow as WindowSpec })
      return { s, dof: s.dof, note: `resolution ≈ 1/M = ${(1 / s.maxLag).toFixed(4)}` }
    }
    case 'multitaper': {
      const k = Math.max(1, Math.min(e.tapers, Math.floor(2 * e.nw)))
      const s = multitaper(x, { nw: e.nw, k, adaptive: e.adaptive })
      return {
        s,
        dof: s.dof.data as Float64Array,
        note: `${k} tapers, bandwidth 2NW/n = ${((2 * e.nw) / n).toFixed(4)}`,
      }
    }
    case 'burg': {
      const s = arPsd(x, e.order, { method: e.method as 'burg', nfft: Math.max(512, n) })
      return { s, note: `AR(${e.order}) by ${e.method}` }
    }
    case 'music': {
      const s = music(x, { sinusoids: e.sinusoids, order: Math.min(e.order, Math.floor(n / 2)) })
      return { s, note: `correlation matrix ${s.order} × ${s.order}`, lineFreqs: toFlat(s.frequencies) }
    }
  }
}

/** The estimator's smoothing knob for the bias–variance sweep, its current value and the values to sweep. */
function sweepOf(e: Estimator, n: number) {
  switch (e.key) {
    case 'periodogram':
      return {
        label: 'window',
        values: WINDOWS.map((w) => w.value as string),
        current: e.window,
        at: (v: string | number) => ({ ...e, window: String(v) }),
      }
    case 'welch':
    case 'bartlett':
      return {
        label: 'segment length L',
        values: SEGMENTS.filter((L) => L <= n),
        current: Math.min(e.nperseg, n),
        at: (v: string | number) => ({ ...e, nperseg: Number(v) }),
      }
    case 'blackmanTukey':
      return {
        label: 'max lag M',
        values: LAGS.filter((M) => M < n),
        current: e.maxLag,
        at: (v: string | number) => ({ ...e, maxLag: Number(v) }),
      }
    case 'multitaper':
      return {
        label: 'NW (2NW − 1 tapers)',
        values: NWS,
        current: e.nw,
        at: (v: string | number) => ({ ...e, nw: Number(v), tapers: Math.max(1, Math.floor(2 * Number(v)) - 1) }),
      }
    case 'burg':
      return {
        label: 'AR order p',
        values: ORDERS.filter((p) => p < n / 4),
        current: e.order,
        at: (v: string | number) => ({ ...e, order: Number(v) }),
      }
    case 'music':
      return null
  }
}

const PRESETS = {
  biasVariance: {
    label: 'bias–variance: Welch segment length',
    values: {
      process: 'ar2',
      'process.re': 0.67,
      'process.im': 0.69,
      'data.n': 2048,
      estimator: 'welch',
      'estimator.nperseg': 32,
      'estimator.overlap': 50,
      'estimator.window': 'hann',
    },
  },
  leakage: {
    label: 'resolution against leakage: windows',
    values: {
      process: 'tones',
      'process.f1': 0.2,
      'process.gap': 0.012,
      'process.level': -50,
      'process.noise': 0.001,
      'data.n': 512,
      estimator: 'periodogram',
      'estimator.window': 'boxcar',
    },
  },
  dynamicRange: {
    label: 'dynamic range: AR(4)',
    values: { process: 'ar4', 'data.n': 1024, estimator: 'periodogram', 'estimator.window': 'boxcar' },
  },
} as const

export function SpectralEstimationSpecimen() {
  const state = useFigureState({
    process: variants(
      {
        ar2: {
          label: 'AR(2): one pole pair',
          params: {
            re: slider(-0.99, 0.99, 0.67, { label: 'pole Re z', step: 0.005 }),
            im: slider(0.005, 0.99, 0.69, { label: 'pole Im z', step: 0.005 }),
          },
        },
        ar4: { label: 'AR(4) of Percival and Walden', params: {} },
        tones: {
          label: 'two sinusoids in noise',
          params: {
            f1: slider(0.02, 0.4, 0.2, { label: 'f₁ (cycles/sample)', step: 0.001 }),
            gap: slider(0.001, 0.08, 0.01, { label: 'separation f₂ − f₁', step: 0.001 }),
            level: slider(-80, 0, -6, { label: 'second tone (dB)', step: 1 }),
            noise: slider(0.001, 2, 0.3, { label: 'noise sd', step: 0.001 }),
          },
        },
      },
      { label: '1 · process', choiceLabel: 'process' },
    ),
    data: row('2 · data', {
      n: choice([128, 256, 512, 1024, 2048, 4096], 512, { label: 'samples n' }),
      seed: slider(1, 50, 1, { label: 'realisation', step: 1 }),
    }),
    estimator: variants(
      {
        periodogram: { label: 'periodogram', params: { window: choice(WINDOWS, 'boxcar', { label: 'window' }) } },
        welch: {
          label: 'Welch',
          params: {
            nperseg: choice(SEGMENTS, 128, { label: 'segment length L' }),
            overlap: choice([0, 25, 50, 75], 50, { label: 'overlap %' }),
            window: choice(WINDOWS, 'hann', { label: 'window' }),
          },
        },
        bartlett: { label: 'Bartlett', params: { nperseg: choice(SEGMENTS, 64, { label: 'segment length L' }) } },
        blackmanTukey: {
          label: 'Blackman–Tukey',
          params: {
            maxLag: slider(2, 512, 32, { label: 'max lag M', step: 1 }),
            lagWindow: choice(
              [
                { value: 'bartlett', label: 'Bartlett (triangle)' },
                { value: 'hann', label: 'Hann' },
                { value: 'boxcar', label: 'rectangular' },
              ],
              'bartlett',
              { label: 'lag window' },
            ),
          },
        },
        multitaper: {
          label: 'multitaper',
          params: {
            nw: slider(1, 10, 4, { label: 'NW', step: 0.5 }),
            tapers: slider(1, 19, 7, { label: 'tapers k (≤ 2NW)', step: 1 }),
            adaptive: toggle(true, 'adaptive weights'),
          },
        },
        burg: {
          label: 'AR (parametric)',
          params: {
            order: slider(1, 40, 4, { label: 'order p', step: 1 }),
            method: choice(
              [
                { value: 'burg', label: 'Burg' },
                { value: 'yule-walker', label: 'Yule–Walker' },
                { value: 'least-squares', label: 'least squares' },
              ],
              'burg',
              { label: 'fit' },
            ),
          },
        },
        music: {
          label: 'MUSIC',
          params: {
            sinusoids: slider(1, 4, 2, { label: 'sinusoids K', step: 1 }),
            order: slider(6, 80, 30, { label: 'matrix size m', step: 1 }),
          },
        },
      },
      { label: '3 · estimator', choiceLabel: 'estimator', initial: 'welch' },
    ),
    reveal: row('4 · reveal', {
      band: toggle(true, '95% confidence band'),
      truth: toggle(true, 'true spectrum'),
    }),
  })

  const process = useMemo(
    () => ({ key: state.process.key, ...state.process.values }) as Process,
    [state.process.key, state.process.values],
  )
  const est = useMemo(
    () => ({ key: state.estimator.key, ...state.estimator.values }) as Estimator,
    [state.estimator.key, state.estimator.values],
  )
  const { n, seed } = state.data

  const data = useMemo(() => draw(process, n, seed), [process, n, seed])
  const truth = data.meta.truth as SpectralTruth
  const x = useMemo(() => toFlat(data.signal.data) as unknown as Float64Array, [data])
  const result = useMemo(() => estimate(est, Float64Array.from(x)), [est, x])
  const view = useMemo(() => {
    const f = toFlat(result.s.f)
    const v = toFlat(result.s.values)
    const trueValues = toFlat(truth.psd(f))
    const lineBand = truth.lines.map((l) => [l.frequency - 0.005, l.frequency + 0.005] as const)
    // MUSIC's pseudospectrum has no power scale: shift it so its median sits on the true floor's median.
    const isMusic = est.key === 'music'
    const median = (a: number[]) => [...a].sort((p, q) => p - q)[Math.floor(a.length / 2)]
    const shift = isMusic ? median(trueValues.map(db)) - median(v.map(db)) : 0
    const ci =
      result.dof !== undefined && !isMusic
        ? spectralConfidence(result.s, typeof result.dof === 'number' ? result.dof : result.dof)
        : null
    const error = isMusic ? null : logSpectralError(result.s, trueValues, { band: [0.005, 0.495], exclude: lineBand })
    return {
      f,
      est: v.map((u) => db(u) + shift),
      truth: trueValues.map(db),
      lower: ci ? toFlat(ci.lower).map(db) : null,
      upper: ci ? toFlat(ci.upper).map(db) : null,
      error,
    }
  }, [result, truth, est.key])

  // Line powers, in dB of the expected periodogram peak (A²/2 · n, density units at fs = 1) for reference marks.
  const lines = truth.lines.map((l) => ({ f: l.frequency, power: (l.amplitude * l.amplitude) / 2 }))
  const resolution = useMemo(() => {
    if (process.key !== 'tones') return null
    const f1 = process.f1
    const f2 = process.f1 + process.gap
    if (est.key === 'music') {
      const fr = result.lineFreqs ?? []
      const ok = fr.length >= 2 && Math.abs(fr[0] - f1) < process.gap / 3 && Math.abs(fr[1] - f2) < process.gap / 3
      return { resolved: ok, text: fr.map((v) => v.toFixed(4)).join(', ') }
    }
    const d = peakDip(result.s, f1, f2)
    return { resolved: d.resolved, text: `dip ${d.dip.toFixed(1)} dB` }
  }, [process, est.key, result])
  const espritFreqs = useMemo(() => {
    if (process.key !== 'tones') return null
    try {
      return toFlat(esprit(Float64Array.from(x), { sinusoids: 2, order: Math.min(40, Math.floor(n / 3)) }).frequencies)
    } catch {
      return null
    }
  }, [process.key, x, n])

  // The bias–variance sweep over the estimator's smoothing knob, over replicate realisations (on release).
  const sweep = useComputed(
    () => {
      const plan = sweepOf(est, n)
      if (!plan) return null
      const reps = Array.from({ length: REPLICATES }, (_, r) =>
        Float64Array.from(toFlat(draw(process, n, `${seed}-${r}`).signal.data)),
      )
      const lineBand = truth.lines.map((l) => [l.frequency - 0.005, l.frequency + 0.005] as const)
      const rows = plan.values.map((v) => {
        const estimates = reps.map((xr) => estimate(plan.at(v) as Estimator, xr).s)
        const f = toFlat(estimates[0].f)
        const e = replicateSpectralError(
          estimates.map((s) => toFlat(s.values)),
          toFlat(truth.psd(f)),
          { f, band: [0.005, 0.495], exclude: lineBand },
        )
        return { v, ...e }
      })
      return { label: plan.label, current: plan.current, rows }
    },
    [est, process, n, seed, truth],
    { mode: 'release' },
  )

  const freq = useAxis({ label: 'frequency (cycles per sample)', range: [0, 0.5] })
  const dbRange = useMemo(() => {
    const t = view.truth.filter(Number.isFinite)
    const top = Math.max(...t, ...lines.map((l) => db(l.power * n)))
    const bottom = Math.min(...t)
    return [Math.floor(bottom - 25), Math.ceil(top + 10)] as const
    // Refit on a new process or length, not on every estimate.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [process.key, n, truth])
  const power = useAxis({ label: 'PSD (dB)', range: dbRange, key: `${process.key}-${n}` })
  const re = useAxis({ label: 'Re z', range: [-1.15, 1.15] })
  const im = useAxis({ label: 'Im z', range: [-1.15, 1.15], equal: re })
  const SHORT: Record<string, string> = {
    boxcar: 'rect.',
    hann: 'Hann',
    hamming: 'Hamm.',
    blackman: 'Black.',
    blackmanharris: 'B–H',
  }
  const sweepCats = sweep.value ? sweep.value.rows.map((r) => SHORT[String(r.v)] ?? String(r.v)) : ['—']
  const knob = useAxis({ label: sweep.value?.label ?? 'knob', categories: sweepCats })
  const err = useAxis({ label: 'dB²', hold: 'union', key: `${est.key}-${process.key}` })
  const sampleAxis = useAxis({ label: 'sample t', range: [0, Math.min(n, 256)] })
  const valueAxis = useAxis({ label: 'x_t', hold: 'initial', key: process.key })

  const poles = useMemo(() => {
    // The process's poles: the roots of z^p − Σφᵢz^{p−i} (aifn/numerics/polynomial), or, for sinusoids, points on the
    // unit circle at ±2πfₖ.
    const ar = truth.model.noise.ar
    if (ar.length) {
      const r = roots([1, ...ar.map((v) => -v)])
      return { x: toFlat(realPart(r)), y: toFlat(imagPart(r)) }
    }
    return {
      x: truth.lines.flatMap((l) => [Math.cos(2 * Math.PI * l.frequency), Math.cos(2 * Math.PI * l.frequency)]),
      y: truth.lines.flatMap((l) => [Math.sin(2 * Math.PI * l.frequency), -Math.sin(2 * Math.PI * l.frequency)]),
    }
  }, [truth])

  const sweepIndex = sweep.value ? sweep.value.rows.findIndex((r) => String(r.v) === String(sweep.value!.current)) : -1
  const fmt = (v: number | undefined, d = 2) => (v === undefined || !Number.isFinite(v) ? '—' : v.toFixed(d))

  return (
    <Figure
      title="Spectral estimation: estimates against the true spectrum"
      purpose="Every nonparametric estimate trades variance for bias: averaging or smoothing narrows the confidence band but blurs peaks; windows trade resolution for leakage; parametric estimates are smooth but only as good as their model."
      state={state}
      defaultSize="XL"
      controls={
        <ControlRow label="presets">
          <div className="flex flex-wrap items-center gap-2">
            {(Object.keys(PRESETS) as (keyof typeof PRESETS)[]).map((key) => (
              <Button
                key={key}
                size="sm"
                variant="outline"
                aria-label={`preset: ${PRESETS[key].label}`}
                onClick={() => {
                  for (const [path, value] of Object.entries(PRESETS[key].values)) state.set(path, value)
                }}
              >
                {PRESETS[key].label}
              </Button>
            ))}
          </div>
        </ControlRow>
      }
      readouts={{
        estimate: (
          <>
            <Readout label="estimator" value={result.note} />
            <Readout
              label="degrees of freedom ν"
              value={
                result.dof === undefined
                  ? '— (parametric)'
                  : typeof result.dof === 'number'
                    ? result.dof.toFixed(1)
                    : `${Math.min(...result.dof).toFixed(1)}–${Math.max(...result.dof).toFixed(1)}`
              }
            />
          </>
        ),
        'against the truth (this realisation)': (
          <>
            <Readout label="bias (mean dB error)" value={view.error ? `${fmt(view.error.bias)} dB` : '—'} />
            <Readout label="spread (sd of dB error)" value={view.error ? `${fmt(view.error.sd)} dB` : '—'} />
            <Readout label="log-spectral distance" value={view.error ? `${fmt(view.error.distance)} dB` : '—'} />
          </>
        ),
        [`over ${REPLICATES} realisations`]: (
          <>
            <Readout label="bias²" value={sweepIndex >= 0 ? `${fmt(sweep.value!.rows[sweepIndex].bias2)} dB²` : '—'} />
            <Readout
              label="variance"
              value={sweepIndex >= 0 ? `${fmt(sweep.value!.rows[sweepIndex].variance)} dB²` : '—'}
            />
          </>
        ),
        ...(resolution
          ? {
              resolution: (
                <>
                  <Readout
                    label="two tones separate"
                    value={`${resolution.resolved ? 'yes' : 'no'} (${resolution.text})`}
                  />
                  <Readout
                    label="ESPRIT frequencies"
                    value={espritFreqs ? espritFreqs.map((v) => v.toFixed(4)).join(', ') : '—'}
                  />
                </>
              ),
            }
          : {}),
      }}
      caption="aifn/signal/spectral (periodogram, welch, bartlett, blackmanTukey, multitaper, spectralConfidence, replicateSpectralError) and aifn/signal/statistical (arPsd, music, esprit) on series from aifn-applied/data/signals, whose truth gives the exact PSD (ink). Drag the pole in the z-plane (AR(2)) or the tone f₁ on the spectrum. The band is the χ² 95% interval from the estimator's equivalent degrees of freedom (it holds away from spectral lines). The lower-right panel repeats the estimate on 12 fresh realisations at each setting of the smoothing knob (computed on release): bias² rises and variance falls as the estimate smooths more. Vertical dashed lines mark the true sinusoids; MUSIC's pseudospectrum is shifted to the floor's level, since its height carries no power."
    >
      <Plots rows={2} cols={2} widths={[2.2, 1]} heights={[1.6, 1]}>
        <Plot x={freq} y={power}>
          {state.reveal.band && view.lower && view.upper && (
            <Area name="95% band" x={view.f} y={view.upper} base={view.lower} slot={0} opacity={0.18} line={false} />
          )}
          <Curve name={est.key === 'music' ? 'MUSIC pseudospectrum' : 'estimate'} x={view.f} y={view.est} slot={0} />
          {state.reveal.truth && <Curve name="true PSD" x={view.f} y={view.truth} emphasis />}
          {state.reveal.truth && lines.map((l, i) => <Annotation key={i} x={l.f} dashed />)}
          {process.key === 'tones' && <Handle {...state.handle('process.f1', { label: 'f₁' })} />}
        </Plot>
        <Plot x={re} y={im} legend={false}>
          <Curve name="unit circle" x={CIRCLE.x} y={CIRCLE.y} muted />
          <Points name="poles" x={poles.x} y={poles.y} slot={1} />
          {process.key === 'ar2' && <Handle {...state.handle(['process.re', 'process.im'], { label: 'pole' })} />}
        </Plot>
        <Plot x={sampleAxis} y={valueAxis}>
          <Curve
            name="x (first 256 samples)"
            x={Array.from({ length: Math.min(n, 256) }, (_, i) => i)}
            y={Array.from(x.slice(0, 256))}
            slot={2}
          />
        </Plot>
        <Plot x={knob} y={err}>
          {sweep.value && (
            <>
              <Bars
                name="bias²"
                x={sweep.value.rows.map((_, i) => i - 0.18)}
                y={sweep.value.rows.map((r) => r.bias2)}
                slot={3}
                width={0.34}
                stale={sweep.stale}
              />
              <Bars
                name="variance"
                x={sweep.value.rows.map((_, i) => i + 0.18)}
                y={sweep.value.rows.map((r) => r.variance)}
                slot={4}
                width={0.34}
                stale={sweep.stale}
              />
              {sweepIndex >= 0 && <Annotation x={sweepIndex} dashed text="current" />}
            </>
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}
