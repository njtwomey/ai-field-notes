import { useContext, useMemo, type ReactNode } from 'react'
import {
  Annotation,
  Bars,
  Curve,
  Figure,
  FrameContext,
  int,
  Plot,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { crossCorrelation } from 'aifn-compute/probability/stats'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { coupledProcesses } from 'aifn-methods/data/signals'

const N = 400
const SHOWN = 120
const MAX_LAG = 30
const PANEL = 200

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** Two sensors that record the same process with an unknown delay; the cross-correlation finds it. */
export function SyncByCrossCorrelation() {
  const state = useFigureState({
    delay: int(8, { ge: 0, le: 25, label: 'true delay (samples)' }),
    noise: slider(0, 40, 2, { step: 0.5, label: 'noise sd of the second sensor' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { delay, noise, seed } = state
  const view = useMemo(() => {
    const d = coupledProcesses(stream(`sync/${seed}`), { n: N, delay, noise, frequency: 0.05, radius: 0.95 })
    const a = toFlat(d.signal.data)
    const b = toFlat(d.partner!.data)
    const { lags, values } = crossCorrelation(a, b, { maxLag: MAX_LAG })
    const k = toFlat(lags)
    const r = toFlat(values)
    const best = k[r.indexOf(Math.max(...r))]
    const ts = Array.from({ length: SHOWN }, (_, i) => i)
    // The second stream moved back by the estimated lag, so its sample t + lag sits at t.
    const aligned = ts.map((t) => (t + best >= 0 && t + best < b.length ? b[t + best] : NaN))
    return { ts, a: a.slice(0, SHOWN), b: b.slice(0, SHOWN), aligned, k, r, best, peak: Math.max(...r) }
  }, [delay, noise, seed])
  const t1 = useAxis({ label: 'sample t', range: [0, SHOWN - 1] })
  const y1 = useAxis({ label: 'value' })
  const t2 = useAxis({ label: 'sample t', range: [0, SHOWN - 1] })
  const y2 = useAxis({ label: 'value' })
  const kAxis = useAxis({ label: 'lag k (samples)', range: [-MAX_LAG - 1, MAX_LAG + 1] })
  const rAxis = useAxis({ label: 'r(k)', range: [-1, 1] })
  return (
    <Figure
      title="Aligning two streams by cross-correlation"
      state={state}
      hoverReadout={false}
      caption="Sensor A records a process; sensor B records a smoothed copy of it some samples later, with its own noise. Top: the raw streams, offset in time. Bottom left: their cross-correlation r(k) at each lag k; the peak is the lag at which B best matches A. The smoothing filter adds one sample of delay, so the peak sits at the true delay plus one. Bottom right: B moved back by the estimated lag, on top of A. The signal itself has a standard deviation of about 6. While the noise is no larger than that, the peak stays at the right lag and only its height falls; at several times that size the peak flattens and the estimate wanders."
      readouts={
        <>
          <Readout label="estimated lag" value={`${view.best} samples`} />
          <Readout label="peak correlation" value={view.peak.toFixed(2)} />
        </>
      }
    >
      <FixedHeight height={PANEL}>
        <Plot x={t1} y={y1} title="raw streams">
          <Curve name="sensor A" x={view.ts} y={view.a} slot={0} />
          <Curve name="sensor B" x={view.ts} y={view.b} slot={1} />
        </Plot>
      </FixedHeight>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={kAxis} y={rAxis} title="cross-correlation">
            <Bars name="r(k)" x={view.k} y={view.r} slot={2} width={0.8} />
            <Annotation x={view.best} text={`k = ${view.best}`} dashed />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={t2} y={y2} title="after alignment">
            <Curve name="sensor A" x={view.ts} y={view.a} slot={0} />
            <Curve name="sensor B, moved back" x={view.ts} y={view.aligned} slot={1} />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
