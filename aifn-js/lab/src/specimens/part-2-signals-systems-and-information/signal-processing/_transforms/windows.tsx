/**
 * "Windows and the leakage trade-off": every registered window's shape and spectrum, with its main-lobe width and
 * peak side lobe measured here against the registry's stated values, and the trade-off across the whole registry.
 */
import { useMemo } from 'react'
import { rfft } from 'aifn/foundation/fourier'
import { toComplexFlat, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { windowRegistry } from 'aifn/signal/windows'
import { Figure } from '@lab/layout'
import { choice, row, slider, useFigureState, when } from '@lab/state'
import { Annotation, Curve, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'
import { amplitudeSpectrum, db, fmt, range } from './common'

const N = 64
const PAD = 32
const ENTRIES = Object.values(windowRegistry)
const OPTIONS = ENTRIES.map((e) => ({ value: e.info.key, label: e.info.name }))

/** The window's normalised spectrum in dB over bins −16 … 16, and its measured first-null width and peak side lobe. */
function measure(w: number[]) {
  const total = w.reduce((a, b) => a + b, 0)
  const X = toComplexFlat(rfft(w, { n: N * PAD }) as Tensor).map((z) => Math.hypot(z.re, z.im) / total)
  const half = X.map((_, k) => k / PAD)
  let k = 1
  while (k < X.length - 1 && !(X[k] <= X[k - 1] && X[k] <= X[k + 1])) k++
  const nullBin = half[k]
  const side = Math.max(...X.slice(k))
  const keep = half.map((b, i) => [b, i] as const).filter(([b]) => b <= 16)
  const bins = [...keep.map(([b]) => -b).reverse(), ...keep.slice(1).map(([b]) => b)]
  const levels = [...keep.map(([, i]) => db(X[i])).reverse(), ...keep.slice(1).map(([, i]) => db(X[i]))]
  return { bins, levels, width: 2 * nullBin, sideDb: db(side) }
}

export function WindowFigure() {
  const state = useFigureState({
    window: row('1 · window', {
      key: choice(OPTIONS, 'hann', { label: 'window' }),
      beta: slider(0, 20, 8.6, { label: 'Kaiser β', step: 0.1, when: when('key', 'kaiser') }),
      std: slider(2, 32, 8, { label: 'Gaussian σ (samples)', step: 0.5, when: when('key', 'gaussian') }),
      alpha: slider(0, 1, 0.5, { label: 'Tukey α', step: 0.01, when: when('key', 'tukey') }),
    }),
  })
  const { key, beta, std, alpha } = state.window
  const entry = windowRegistry[key]
  const w = useMemo(() => toFlat(entry(N, { periodic: true, beta, std, alpha })), [entry, beta, std, alpha])
  const m = useMemo(() => measure(w), [w])
  const rect = useMemo(() => measure(range(N).map(() => 1)), [])
  const coherent = w.reduce((a, b) => a + b, 0) / N
  const enbw = (N * w.reduce((a, b) => a + b * b, 0)) / w.reduce((a, b) => a + b, 0) ** 2
  const n = useAxis({ label: 'sample n', range: [0, N] })
  const v = useAxis({ label: 'w[n]', range: [0, 1.05] })
  const bins = useAxis({ label: 'frequency (bins)', range: [-16, 16] })
  const level = useAxis({ label: 'gain (dB, 0 at the peak)', range: [-140, 5] })
  return (
    <Figure
      title="A window's shape and its spectrum"
      purpose="A window that tapers smoothly to zero has side lobes that fall fast, so little leakage reaches distant bins, but its main lobe is wider, so two close tones blur together; the registry records both numbers for every window."
      defaultSize="L"
      state={state}
      readouts={{
        'measured here': (
          <>
            <Readout label="main lobe (null to null)" value={`${fmt(m.width, 2)} bins`} />
            <Readout label="peak side lobe" value={`${fmt(m.sideDb, 1)} dB`} />
            <Readout label="coherent gain" value={fmt(coherent, 3)} />
            <Readout label="noise bandwidth" value={`${fmt(enbw, 2)} bins`} />
          </>
        ),
        registry: (
          <>
            <Readout
              label="main lobe"
              value={entry.info.mainLobeWidth === undefined ? 'depends on σ' : `${entry.info.mainLobeWidth} bins`}
            />
            <Readout
              label="side lobe"
              value={entry.info.sideLobeDb === undefined ? 'depends on σ' : `${entry.info.sideLobeDb} dB`}
            />
          </>
        ),
      }}
      caption={`Periodic windows of N = ${N} samples (aifn windowRegistry), spectra by a DFT zero-padded ×${PAD} and normalised to 0 dB at DC. The rectangular window's spectrum is drawn in grey for comparison. The measured values use the current parameter; the registry's are at its default, so they agree there.`}
    >
      <Plots rows={1} cols={2} widths={[1, 1.8]}>
        <Plot x={n} y={v}>
          <Curve name={entry.info.name} x={range(N)} y={w} slot={0} showPoints />
        </Plot>
        <Plot x={bins} y={level}>
          <Curve name="rectangular" x={rect.bins} y={rect.levels} muted thin />
          <Curve name={entry.info.name} x={m.bins} y={m.levels} slot={0} />
          <Annotation x={-m.width / 2} dashed />
          <Annotation x={m.width / 2} dashed text="first null" />
          <Annotation y={m.sideDb} dashed text="peak side lobe" />
        </Plot>
      </Plots>
    </Figure>
  )
}

export function TradeOffFigure() {
  const state = useFigureState({
    key: choice(OPTIONS, 'blackman', { label: 'highlight' }),
    gap: slider(1, 12, 8, { label: 'weak tone: bins above the strong one', step: 0.25 }),
  })
  const listed = ENTRIES.filter((e) => e.info.mainLobeWidth !== undefined && e.info.sideLobeDb !== undefined)
  const pick = windowRegistry[state.key]
  // A strong tone half way between bins and one 60 dB down, seen through the highlighted window.
  const x = useMemo(
    () =>
      range(N).map(
        (i) => Math.cos((2 * Math.PI * 10.5 * i) / N) + 1e-3 * Math.cos((2 * Math.PI * (10.5 + state.gap) * i) / N),
      ),
    [state.gap],
  )
  const w = useMemo(() => toFlat(pick(N, { periodic: true })), [pick])
  const s = useMemo(
    () =>
      amplitudeSpectrum(
        x.map((v, i) => v * w[i]),
        'rectangular',
        16,
      ),
    [x, w],
  )
  const sr = useMemo(() => amplitudeSpectrum(x, 'rectangular', 16), [x])
  const width = useAxis({ label: 'main-lobe width (bins, null to null)', range: [1.5, 10.5] })
  const side = useAxis({ label: 'peak side lobe (dB)', range: [-100, -5] })
  const bins = useAxis({ label: 'frequency (bins)', range: [0, 32] })
  const lv = useAxis({ label: 'amplitude (dB)', range: [-120, 5] })
  const norm = w.reduce((a, b) => a + b, 0) / N
  return (
    <Figure
      title="The trade-off across the registry"
      purpose="Lower side lobes cost a wider main lobe: no window is best at both, so the choice depends on whether the weak component sits far from a strong one (low side lobes win) or close to it (a narrow main lobe wins)."
      defaultSize="L"
      state={state}
      readouts={
        <>
          <Readout
            label={pick.info.name}
            value={`${pick.info.mainLobeWidth ?? '—'} bins, ${pick.info.sideLobeDb ?? '—'} dB`}
          />
        </>
      }
      caption="Left: each window's main-lobe width and peak side-lobe level as stated in aifn's window registry (Harris, 1978). Right: a unit tone at 10.5 bins and a tone 60 dB weaker above it, under the rectangular window (grey) and the highlighted one. Move the weak tone close: Blackman–Harris loses it inside its main lobe; move it far: the rectangular window buries it in leakage."
    >
      <Plots rows={1} cols={2}>
        <Plot x={width} y={side}>
          <Points
            name="windows"
            x={listed.map((e) => e.info.mainLobeWidth!)}
            y={listed.map((e) => e.info.sideLobeDb!)}
            muted
            size={8}
          />
          {listed.map((e) => (
            <Annotation key={e.info.key} at={[e.info.mainLobeWidth!, e.info.sideLobeDb!]} text={e.info.name} />
          ))}
          {pick.info.mainLobeWidth !== undefined && (
            <Points
              name={pick.info.name}
              x={[pick.info.mainLobeWidth]}
              y={[pick.info.sideLobeDb!]}
              emphasis
              size={12}
            />
          )}
        </Plot>
        <Plot x={bins} y={lv}>
          <Curve name="rectangular" x={sr.f.map((f) => f * N)} y={sr.amp.map((a) => db(a))} muted />
          <Curve name={pick.info.name} x={s.f.map((f) => f * N)} y={s.amp.map((a) => db(a / norm))} slot={0} />
          <Annotation x={10.5 + state.gap} dashed text="weak tone" />
        </Plot>
      </Plots>
    </Figure>
  )
}
