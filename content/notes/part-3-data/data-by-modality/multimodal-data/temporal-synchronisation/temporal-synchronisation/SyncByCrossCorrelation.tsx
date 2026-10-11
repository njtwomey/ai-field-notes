import { useContext, useMemo, useState, type ReactNode } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  FrameContext,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { child, normals, stream } from 'aifn-compute/foundation/random'
import { add, toFlat } from 'aifn-compute/foundation/tensor'
import { crossCorrelation } from 'aifn-compute/probability/stats'
import { sincInterpolate } from 'aifn-compute/signal/multirate'
import { arSeries } from 'aifn-methods/data/synthetic'

/** Samples per second of both devices, and samples per recording (12 s). */
const FS = 50
const N = 600
/** The shared source covers -3 s to 15 s, so that every time either device samples lies inside it. */
const SOURCE_T0 = -3
const SOURCE_N = 900
/** Largest offset searched, in samples (±1.5 s), and the length of the early and late windows (3 s). */
const MAX_LAG = 75
const WINDOW = 150
const PANEL = 200

const times = Array.from({ length: N }, (_, k) => k / FS)

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** The cross-correlation of two equal-length records, as a curve over offsets in seconds, and its peak. */
function correlate(a: number[], b: number[]) {
  const { lags, values } = crossCorrelation(a, b, { maxLag: MAX_LAG })
  // A peak at lag k means b follows a by k samples; b sampling the source τ seconds later puts the peak at -τ fs.
  const offset = toFlat(lags).map((l) => -l / FS)
  const r = toFlat(values)
  let best = 0
  r.forEach((v, i) => {
    if (v > r[best]) best = i
  })
  return { offset, r, peak: offset[best], height: r[best] }
}

/**
 * Two devices record one source. Device B starts τ seconds later on the source's clock and its clock runs fast by a
 * drift δ, so its k-th sample is taken at true time τ + (k / fs)(1 + δ). Cross-correlation estimates the offset over
 * the whole record and in an early and a late window; the two window offsets give the drift.
 */
export function SyncByCrossCorrelation() {
  const state = useFigureState({
    drift: slider(0, 2, 0.5, { step: 0.05, label: 'clock drift of B (%)' }),
    noise: slider(0, 1.5, 0.4, { step: 0.05, label: 'sensor noise sd' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { drift, noise, seed } = state
  // The true offset has a natural place on the correlation chart, so it is set by dragging there.
  const [tau, setTau] = useState(0.6)

  const view = useMemo(() => {
    const s = stream(`temporal-synchronisation/${seed}`)
    const source = arSeries(child(s, 'source'), { coefficients: [0.9], n: SOURCE_N })
    const signal = { kind: 'signal' as const, data: source.y, fs: FS, t0: SOURCE_T0 }
    const delta = drift / 100
    const trueB = times.map((t) => tau + t * (1 + delta))
    const a = toFlat(add(sincInterpolate(signal, times), normals(child(s, 'noise-a'), N, 0, noise)))
    const b = toFlat(add(sincInterpolate(signal, trueB), normals(child(s, 'noise-b'), N, 0, noise)))
    const whole = correlate(a, b)
    const early = correlate(a.slice(0, WINDOW), b.slice(0, WINDOW))
    const late = correlate(a.slice(N - WINDOW), b.slice(N - WINDOW))
    // Window centres on B's clock; the offset grows by δ per second of recording.
    const cEarly = WINDOW / 2 / FS
    const cLate = (N - WINDOW / 2) / FS
    const driftHat = (late.peak - early.peak) / (cLate - cEarly)
    const tauHat = early.peak - driftHat * cEarly
    const corrected = times.map((t) => tauHat + t * (1 + driftHat))
    return { a, b, whole, early, late, driftHat, tauHat, corrected }
  }, [tau, drift, noise, seed])

  const t = useAxis({ label: 'time on its own clock (s)', range: [0, 12] })
  const t2 = useAxis({ label: 'time on A’s clock (s)', range: [0, 12] })
  const v = useAxis({ label: 'value', range: [-5, 5] })
  const v2 = useAxis({ label: 'value', range: [-5, 5] })
  const off = useAxis({ label: 'offset of B (s)', range: [-1.5, 1.5] })
  const r = useAxis({ label: 'correlation', range: [-0.4, 1] })

  return (
    <Figure
      title="Finding an offset and a drift by cross-correlation"
      state={state}
      hoverReadout={false}
      caption={`Two devices sample one source at ${FS} samples per second for 12 s. Device B starts later by the true offset τ and its clock runs fast by the drift. Top: each record against its own clock; the two do not line up. Bottom left: the cross-correlation of the whole records (thick) and of the first and last 3 s (thin); each peak marks an offset estimate. Drag the dashed line to set the true offset. With drift, the early and late peaks separate, and the whole-record peak sits at the offset in the middle of the recording and flattens. Bottom right: B's samples placed at the corrected times τ̂ + t(1 + δ̂) from the two window estimates, now on top of A. Real clocks drift by tens of parts per million, not per cent; the drift here is exaggerated so that 12 s show it. Estimates move in steps of one sample (0.02 s), so the drift estimate moves in steps of about 0.2%.`}
      readouts={
        <>
          <Readout label="true offset τ" value={`${formatNumber(tau)} s`} />
          <Readout label="whole-record peak" value={`${formatNumber(view.whole.peak)} s`} />
          <Readout
            label="early, late window peaks"
            value={`${formatNumber(view.early.peak)} s, ${formatNumber(view.late.peak)} s`}
          />
          <Readout label="estimated drift δ̂" value={`${formatNumber(100 * view.driftHat)}%`} />
          <Readout label="peak correlation" value={formatNumber(view.whole.height)} />
        </>
      }
    >
      <FixedHeight height={PANEL}>
        <Plot x={t} y={v} title="As recorded">
          <Curve name="device A" x={times} y={view.a} slot={0} thin />
          <Curve name="device B" x={times} y={view.b} slot={1} thin />
        </Plot>
      </FixedHeight>
      <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={off} y={r} title="Cross-correlation">
            <Curve name="first 3 s" x={view.early.offset} y={view.early.r} slot={2} thin />
            <Curve name="last 3 s" x={view.late.offset} y={view.late.r} slot={3} thin />
            <Curve name="whole record" x={view.whole.offset} y={view.whole.r} slot={1} emphasis />
            <Points name="peak" x={[view.whole.peak]} y={[view.whole.height]} slot={1} size={7} />
            <Handle
              kind="x"
              at={tau}
              label="true offset τ"
              onDrag={(x) => setTau(Math.max(-1.2, Math.min(1.2, Math.round(x * 100) / 100)))}
            />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={t2} y={v2} title="B on corrected timestamps">
            <Curve name="device A" x={times} y={view.a} slot={0} thin />
            <Curve name="device B, corrected" x={view.corrected} y={view.b} slot={1} thin />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
