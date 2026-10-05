/**
 * "Filter design gallery": every registered design family with its responses and a pole–zero plot whose poles and
 * zeros can be dragged; the IIR families compared at equal order; a design applied causally and with zero phase.
 */
import { useMemo, useState } from 'react'
import {
  bessel,
  butter,
  cheby1,
  cheby2,
  ellip,
  filtfilt,
  firwin,
  freqz,
  groupDelay,
  lfilter,
  equiripple,
} from 'aifn-compute/signal/filters'
import { unwrapPhase } from 'aifn-compute/signal'
import { convert, toZerosPolesGain, zerosPolesGain } from 'aifn-compute/systems'
import {
  angle,
  complexAbs,
  toComplexFlat,
  toFlat,
  type ComplexNumber,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import type { LtiSystem } from 'aifn-compute/foundation/contracts'
import { Figure } from 'aifn-render/layout'
import { choice, row, slider, toggle, variants, useFigureState } from 'aifn-render/state'
import { Annotation, Curve, Handle, Plot, Plots, Points, Readout, Segments, useAxis } from 'aifn-render/viz'
import { db, fmt, noise, range, stems } from './common'

const FS = 1000
type Band = 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'
const BANDS = ['lowpass', 'highpass', 'bandpass', 'bandstop'] as const
const order = (initial = 4) => slider(1, 10, initial, { label: 'order n', step: 1 })
const rp = slider(0.1, 3, 1, { label: 'passband ripple r_p (dB)', step: 0.1 })
const rs = slider(20, 80, 40, { label: 'stopband attenuation r_s (dB)', step: 1 })
const taps = slider(4, 50, 20, { label: 'half-length h (taps N = 2h + 1)', step: 1 })

/** The families of the registry's filter designs and their parameters. */
const FAMILY = variants(
  {
    butter: { label: 'Butterworth', params: { order: order() } },
    cheby1: { label: 'Chebyshev I', params: { order: order(), rp } },
    cheby2: { label: 'Chebyshev II', params: { order: order(), rs } },
    ellip: { label: 'Elliptic', params: { order: order(), rp, rs } },
    bessel: { label: 'Bessel', params: { order: order() } },
    firwin: {
      label: 'FIR, window method',
      params: { taps, window: choice(['hamming', 'hann', 'blackman', 'rectangular'], 'hamming', { label: 'window' }) },
    },
    remez: {
      label: 'FIR, Parks–McClellan',
      params: { taps, transition: slider(10, 150, 50, { label: 'transition width (Hz)', step: 5 }) },
    },
  },
  { label: '1 · family', choiceLabel: 'design', initial: 'ellip' },
)

type Params = Record<string, number | string>

/** A design as an LTI system at fs = 1 kHz: IIR in zeros–poles–gain form, FIR as its taps. */
function design(family: string, p: Params, band: Band, f1: number, f2: number): LtiSystem {
  const wn = band === 'lowpass' || band === 'highpass' ? f1 : ([f1, f2] as const)
  const n = Number(p.order)
  const opts = { btype: band, fs: FS }
  if (family === 'butter') return butter(n, wn, opts)
  if (family === 'cheby1') return cheby1(n, Number(p.rp), wn, opts)
  if (family === 'cheby2') return cheby2(n, Number(p.rs), wn, opts)
  if (family === 'ellip') return ellip(n, Number(p.rp), Number(p.rs), wn, opts)
  if (family === 'bessel') return bessel(n, wn, opts)
  // FIR: an odd length, which every band type allows.
  const N = 2 * Number(p.taps) + 1
  if (family === 'firwin')
    return firwin(N, band === 'lowpass' || band === 'highpass' ? f1 : [f1, f2], {
      fs: FS,
      passZero: band,
      window: p.window as 'hamming',
    })
  return equiripple(N, band === 'lowpass' || band === 'highpass' ? f1 : [f1, f2], Number(p.transition), {
    btype: band,
    fs: FS,
  })
}

/** Magnitude (dB), unwrapped phase, group delay, impulse and step responses of a discrete system. */
function responses(sys: LtiSystem) {
  const h = freqz(sys, { n: 512 })
  const f = toFlat(h.f)
  const mag = toFlat(complexAbs(h.values) as Tensor)
  const phase = toFlat(unwrapPhase(toFlat(angle(h.values) as Tensor)))
  const gd = toFlat(groupDelay(sys, { n: 512 }).delay)
  const runnable = sys.repr.form === 'zpk' ? convert(sys, 'sos') : sys
  const len = 120
  const impulse = toFlat(
    lfilter(
      runnable,
      range(len).map((i) => (i === 0 ? 1 : 0)),
    ).y as Tensor,
  )
  const step = toFlat(
    lfilter(
      runnable,
      range(len).map(() => 1),
    ).y as Tensor,
  )
  return {
    f,
    db: mag.map((v) => db(v, -140)),
    mag,
    phase,
    gd: gd.map((v) => (Number.isFinite(v) ? v : NaN)),
    impulse,
    step,
  }
}

const UNIT = range(201).map((i) => (2 * Math.PI * i) / 200)

// ── 1. The gallery ────────────────────────────────────────────────────────────────────────────────────────────────

export function FilterGalleryFigure() {
  const state = useFigureState({
    family: FAMILY,
    band: row('2 · band', {
      btype: choice(BANDS, 'lowpass', { label: 'band type' }),
      f1: slider(10, 480, 120, { label: 'edge f₁ (Hz)', step: 1 }),
      f2: slider(20, 490, 250, {
        label: 'edge f₂ (Hz, band types)',
        step: 1,
        when: (v) => String(v.btype).startsWith('band'),
      }),
    }),
  })
  const fam = state.family
  const params = fam.values as unknown as Params
  const { btype, f1: e1, f2: e2 } = state.band
  const [f1, f2] = btype.startsWith('band') ? [Math.min(e1, e2 - 5), Math.max(e2, e1 + 5)] : [e1, e2]
  const key = JSON.stringify([fam.key, params, btype, f1, f2])
  const base = useMemo(() => {
    const [k, p, b, a, c] = JSON.parse(key) as [string, Params, Band, number, number]
    return design(k, p, b, a, c)
  }, [key])
  const baseZpk = useMemo(() => {
    const r = toZerosPolesGain(base).repr
    return { z: toComplexFlat(r.zeros), p: toComplexFlat(r.poles), k: r.gain }
  }, [base])
  const [edit, setEdit] = useState<{ key: string; z: ComplexNumber[]; p: ComplexNumber[] } | null>(null)
  const edited = edit?.key === key
  const zpk = useMemo(() => (edited && edit ? { ...baseZpk, z: edit.z, p: edit.p } : baseZpk), [baseZpk, edited, edit])
  const sys = useMemo(() => {
    if (!edited) return base
    // Rescale an edited design so its largest gain is 1 again.
    const raw = zerosPolesGain(zpk.z, zpk.p, zpk.k, { dt: 1 / FS })
    const peak = Math.max(...toFlat(complexAbs(freqz(raw, { n: 512 }).values) as Tensor))
    return zerosPolesGain(zpk.z, zpk.p, zpk.k / (peak || 1), { dt: 1 / FS })
  }, [base, edited, zpk])
  const r = useMemo(() => responses(sys), [sys])
  const radius = Math.max(0, ...zpk.p.map((p) => Math.hypot(p.re, p.im)))
  /** Moves root i of a list (and its conjugate partner) to a new upper-half-plane position; real roots stay real. */
  const move = (which: 'z' | 'p', i: number, [x, y]: [number, number]) => {
    const list = (which === 'z' ? zpk.z : zpk.p).map((c) => ({ ...c }))
    const r0 = list[i]
    if (Math.abs(r0.im) < 1e-9) list[i] = { re: x, im: 0 }
    else {
      const j = list.findIndex((c, k) => k !== i && Math.abs(c.re - r0.re) < 1e-9 && Math.abs(c.im + r0.im) < 1e-9)
      const im = Math.max(1e-3, Math.abs(y))
      list[i] = { re: x, im }
      if (j >= 0) list[j] = { re: x, im: -im }
    }
    setEdit({ key, z: which === 'z' ? list : zpk.z, p: which === 'p' ? list : zpk.p })
  }
  const draggable = (which: 'z' | 'p') =>
    (which === 'z' ? zpk.z : zpk.p)
      .map((c, i) => ({ c, i }))
      .filter(({ c }) => c.im >= -1e-9 && Math.hypot(c.re, c.im) > 1e-6 && Math.hypot(c.re, c.im) < 3)
  const freq = useAxis({ label: 'frequency (Hz)', range: [0, FS / 2] })
  const gain = useAxis({ label: 'gain (dB)', range: [-100, 5] })
  const phase = useAxis({ label: 'phase (rad, unwrapped)', hold: 'union', key })
  const delay = useAxis({ label: 'group delay (samples)', hold: 'union', key })
  const re = useAxis({ label: 'Re z', range: [-1.5, 1.5] })
  const im = useAxis({ label: 'Im z', range: [-1.5, 1.5], equal: re })
  const n = useAxis({ label: 'sample n', range: [0, 120] })
  const h = useAxis({ label: 'impulse response', hold: 'union', key })
  const stepAxis = useAxis({ label: 'step response', hold: 'union', key })
  const showZ = draggable('z')
  const showP = draggable('p')
  return (
    <Figure
      title="Filter design gallery"
      purpose="Each classical design places its poles and zeros by a different rule: Butterworth for a flat passband, Chebyshev and elliptic for a sharper edge bought with ripple, Bessel for a constant delay, and FIR designs for exactly linear phase."
      defaultSize="XL"
      state={state}
      readouts={{
        design: (
          <>
            <Readout label="poles, zeros" value={`${zpk.p.length}, ${zpk.z.length}`} />
            <Readout label="largest |pole|" value={fmt(radius, 4)} />
            <Readout label="stable" value={radius < 1 ? 'yes' : 'no'} />
            <Readout label="edited by hand" value={edited ? 'yes' : 'no'} />
          </>
        ),
        response: (
          <>
            <Readout label="gain at f₁" value={`${fmt(r.db[Math.round((f1 / (FS / 2)) * 512)] ?? NaN, 2)} dB`} />
            <Readout label="max group delay" value={`${fmt(Math.max(...r.gd.filter(Number.isFinite)), 1)} samples`} />
          </>
        ),
      }}
      caption="fs = 1 kHz. Pick a family and band type; drag the edge lines on the magnitude panel. On the pole–zero plot (diamonds: poles, circles: zeros, each under a drag handle; the unit circle in grey) drag any pole or zero in the upper half-plane: its conjugate follows, real ones slide along the axis, and every response updates (the gain is rescaled to a peak of 0 dB). A pole dragged outside the circle makes the impulse response grow. Changing any control restores the design. FIR designs keep their poles at the origin, so only their zeros move; their group delay is constant, (N − 1)/2."
    >
      <Plots rows={3} cols={2} widths={[1.4, 1]} heights={[1.6, 1, 1]}>
        <Plot x={freq} y={gain}>
          <Curve name="|H| (dB)" x={r.f} y={r.db} slot={0} />
          <Handle kind="x" at={e1} label="f₁" onDrag={(v) => state.set('band.f1', v)} />
          {btype.startsWith('band') && <Handle kind="x" at={e2} label="f₂" onDrag={(v) => state.set('band.f2', v)} />}
        </Plot>
        <Plot x={re} y={im}>
          <Curve name="unit circle" x={UNIT.map(Math.cos)} y={UNIT.map(Math.sin)} muted thin />
          <Points name="zeros" x={zpk.z.map((c) => c.re)} y={zpk.z.map((c) => c.im)} slot={1} shape={0} size={15} />
          <Points name="poles" x={zpk.p.map((c) => c.re)} y={zpk.p.map((c) => c.im)} slot={0} shape={3} size={16} />
          {showP.map(({ c, i }) => (
            <Handle key={`p${i}`} kind="point" at={[c.re, c.im]} onDrag={(v) => move('p', i, v)} />
          ))}
          {showZ.map(({ c, i }) => (
            <Handle key={`z${i}`} kind="point" at={[c.re, c.im]} onDrag={(v) => move('z', i, v)} />
          ))}
        </Plot>
        <Plot x={freq} y={phase}>
          <Curve name="phase" x={r.f} y={r.phase} slot={0} />
        </Plot>
        <Plot x={freq} y={delay}>
          <Curve name="group delay" x={r.f} y={r.gd.map((v) => (Math.abs(v) > 500 ? NaN : v))} slot={0} />
        </Plot>
        <Plot x={n} y={h}>
          <Segments name="impulse response" segments={stems(range(r.impulse.length), r.impulse)} slot={0} />
        </Plot>
        <Plot x={n} y={stepAxis}>
          <Curve name="step response" x={range(r.step.length)} y={r.step} slot={0} />
          <Annotation y={1} dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 2. The IIR families at equal order ────────────────────────────────────────────────────────────────────────────

const IIR = [
  { key: 'butter', label: 'Butterworth' },
  { key: 'cheby1', label: 'Chebyshev I' },
  { key: 'cheby2', label: 'Chebyshev II' },
  { key: 'ellip', label: 'elliptic' },
  { key: 'bessel', label: 'Bessel' },
] as const

export function FamiliesFigure() {
  const state = useFigureState({
    spec: row('1 · specification', {
      order: slider(1, 10, 5, { label: 'order n (every family)', step: 1 }),
      rp: slider(0.1, 3, 1, { label: 'r_p (dB)', step: 0.1 }),
      rs: slider(20, 80, 50, { label: 'r_s (dB)', step: 1 }),
    }),
    cutoff: slider(20, 450, 150, { label: 'cutoff (Hz)', step: 1, onChart: true }),
    log: row('2 · view', { zoom: toggle(false, 'zoom on the passband') }),
  })
  const { order: n, rp: p, rs: s } = state.spec
  const fc = state.cutoff
  const all = useMemo(
    () => IIR.map(({ key }) => responses(design(key, { order: n, rp: p, rs: s }, 'lowpass', fc, fc))),
    [n, p, s, fc],
  )
  const freq = useAxis({ label: 'frequency (Hz)', range: [0, FS / 2] })
  const gain = useAxis({ label: 'gain (dB)', range: state.log.zoom ? [-4, 0.5] : [-100, 5] })
  const delay = useAxis({ label: 'group delay (samples)', hold: 'union', key: `${n}` })
  const t = useAxis({ label: 'sample n', range: [0, 60] })
  const st = useAxis({ label: 'step response', range: [0, 1.4] })
  const ph = useAxis({ label: 'phase (rad)', hold: 'union', key: `${n}` })
  const width = all.map((r) => {
    // Transition: from the last frequency within 3 dB of the peak to the first 40 dB down.
    const pass = r.f.filter((_, i) => r.db[i] > -3).pop() ?? 0
    const stop = r.f.find((_, i) => r.f[i] > pass && r.db[i] < -40) ?? NaN
    return stop - pass
  })
  return (
    <Figure
      title="The IIR families at equal order"
      purpose="At one order and cutoff, ripple buys steepness: the elliptic filter falls fastest, Butterworth sits between, and Bessel falls slowest but has the flattest group delay and a step response without overshoot."
      defaultSize="L"
      state={state}
      readouts={{
        'width from −3 dB to −40 dB': (
          <>
            {IIR.map((f, i) => (
              <Readout
                key={f.key}
                label={f.label}
                value={Number.isFinite(width[i]) ? `${fmt(width[i], 0)} Hz` : '> Nyquist'}
              />
            ))}
          </>
        ),
      }}
      caption="Low-pass designs at fs = 1 kHz; drag the cutoff line. Chebyshev I and elliptic edges are passband edges, Chebyshev II's the stopband edge, Butterworth's and Bessel's the −3 dB and phase-matched points, so the curves cross the cutoff at different levels. Zoom on the passband to see the ripple of r_p dB."
    >
      <Plots rows={2} cols={2} heights={[1.4, 1]}>
        <Plot x={freq} y={gain}>
          {all.map((r, i) => (
            <Curve key={IIR[i].key} name={IIR[i].label} x={r.f} y={r.db} slot={i} />
          ))}
          <Handle {...state.handle('cutoff', { label: 'cutoff' })} />
        </Plot>
        <Plot x={freq} y={delay}>
          {all.map((r, i) => (
            <Curve
              key={IIR[i].key}
              name={IIR[i].label}
              x={r.f}
              y={r.gd.map((v) => (Math.abs(v) > 200 ? NaN : v))}
              slot={i}
            />
          ))}
        </Plot>
        <Plot x={t} y={st}>
          {all.map((r, i) => (
            <Curve key={IIR[i].key} name={IIR[i].label} x={range(60)} y={r.step.slice(0, 60)} slot={i} />
          ))}
          <Annotation y={1} dashed />
        </Plot>
        <Plot x={freq} y={ph}>
          {all.map((r, i) => (
            <Curve key={IIR[i].key} name={IIR[i].label} x={r.f} y={r.phase} slot={i} />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 3. Applying a design: causal against zero phase ───────────────────────────────────────────────────────────────

export function ApplyFilterFigure() {
  const state = useFigureState({
    family: choice(
      [
        ...IIR.map((f) => ({ value: f.key, label: f.label })),
        { value: 'remez', label: 'FIR, Parks–McClellan (61 taps)' },
      ],
      'ellip',
      { label: 'family' },
    ),
    order: slider(1, 8, 4, { label: 'order n (IIR)', step: 1 }),
    cutoff: slider(10, 200, 40, { label: 'cutoff (Hz)', step: 1, onChart: true }),
    reveal: row('reveal', { zero: toggle(true, 'filtfilt (zero phase)'), clean: toggle(true, 'the clean signal') }),
  })
  const fs = FS
  const len = 600
  const sig = useMemo(() => {
    const t = range(len).map((i) => i / fs)
    const clean = t.map((s) => Math.sin(2 * Math.PI * 4 * s) + 0.5 * Math.sin(2 * Math.PI * 11 * s + 1))
    const w = noise('gallery-apply', len, 0.25)
    const x = t.map((s, i) => clean[i] + 0.6 * Math.sin(2 * Math.PI * 120 * s) + w[i])
    return { t, clean, x }
  }, [fs])
  const sys = useMemo(
    () =>
      state.family === 'remez'
        ? design('remez', { taps: 30, transition: 60 }, 'lowpass', state.cutoff, 0)
        : convert(design(state.family, { order: state.order, rp: 1, rs: 50 }, 'lowpass', state.cutoff, 0), 'sos'),
    [state.family, state.order, state.cutoff],
  )
  const causal = useMemo(() => toFlat(lfilter(sys, sig.x).y as Tensor), [sys, sig])
  const zero = useMemo(() => toFlat(filtfilt(sys, sig.x) as Tensor), [sys, sig])
  const rmse = (y: number[]) => Math.sqrt(y.reduce((a, v, i) => a + (v - sig.clean[i]) ** 2, 0) / len)
  // Delay of the causal output: the lag in samples maximising its correlation with the clean signal.
  const lag = useMemo(() => {
    let best = 0
    let bestV = -Infinity
    for (let L = 0; L < 80; L++) {
      let s = 0
      for (let i = 0; i + L < len; i++) s += sig.clean[i] * causal[i + L]
      if (s > bestV) [bestV, best] = [s, L]
    }
    return best
  }, [sig, causal])
  const t = useAxis({ label: 'time (s)' })
  const y = useAxis({ label: 'signal', range: [-2.8, 2.8] })
  const f = useAxis({ label: 'frequency (Hz)', range: [0, 300] })
  const g = useAxis({ label: 'gain (dB)', range: [-80, 5] })
  const resp = useMemo(() => responses(sys), [sys])
  return (
    <Figure
      title="Causal filtering delays; zero-phase filtering does not"
      purpose="A causal filter delays each frequency by its group delay, so the output lags the clean signal; filtering forwards then backwards cancels the phase (squaring the magnitude) and lines the output up with the input."
      defaultSize="L"
      state={state}
      readouts={
        <>
          <Readout label="causal lag" value={`${lag} samples = ${lag} ms`} />
          <Readout label="rms error, causal" value={fmt(rmse(causal), 3)} />
          <Readout label="rms error, filtfilt" value={fmt(rmse(zero), 3)} />
        </>
      }
      caption="Input: 4 Hz and 11 Hz tones (the clean signal), a 120 Hz hum of amplitude 0.6 and white noise of sd 0.25, at 1 kHz. Drag the cutoff on the response panel. The causal output lags by roughly the passband group delay (largest for the elliptic and Chebyshev designs near their edge); filtfilt removes the lag at any order."
    >
      <Plots rows={2} heights={[1, 1.6]}>
        <Plot x={f} y={g}>
          <Curve name="|H| (dB)" x={resp.f} y={resp.db} slot={0} />
          <Handle {...state.handle('cutoff', { label: 'cutoff' })} />
        </Plot>
        <Plot x={t} y={y}>
          <Curve name="input" x={sig.t} y={sig.x} muted thin />
          <Curve name="lfilter (causal)" x={sig.t} y={causal} slot={1} />
          {state.reveal.zero && <Curve name="filtfilt (zero phase)" x={sig.t} y={zero} slot={2} />}
          {state.reveal.clean && <Curve name="clean" x={sig.t} y={sig.clean} emphasis dashed />}
        </Plot>
      </Plots>
    </Figure>
  )
}
