import { useContext, useMemo, type ReactNode } from 'react'
import { Curve, Figure, FrameContext, Handle, int, Plot, Readout, Segments, useAxis, useFigureState } from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { arRegimes } from 'aifn-methods/data/synthetic'

const N = 300
const CHANGEPOINTS = [70, 125, 205, 240]
const ROWS = 6

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

type Seg = { from: [number, number]; to: [number, number] }

/** Sliding windows over a labelled series: the windows that straddle a change of label have no single label. */
export function WindowLayout() {
  const state = useFigureState({
    length: int(40, { ge: 4, le: 150, suggestions: [20, 40, 80], label: 'window length L' }),
    stride: int(20, { ge: 1, le: 150, suggestions: [5, 20, 40], label: 'stride s' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { length, stride, seed } = state
  const series = useMemo(() => {
    const d = arRegimes(stream(`windows/${seed}`), { n: N, changepoints: CHANGEPOINTS })
    const x = toFlat(d.x)
    const regime = toFlat(d.y!).map((segment) => segment % 2)
    const t = x.map((_, i) => i)
    // One curve per label, broken (NaN) where the other label holds, so the colour shows the label.
    const part = (k: number) => x.map((v, i) => (regime[i] === k || regime[i - 1] === k ? v : NaN))
    return { t, x, regime, slow: part(0), fast: part(1) }
  }, [seed])
  const windows = useMemo(() => {
    const pure: Seg[][] = [[], []]
    const mixed: Seg[] = []
    let count = 0
    for (let start = 0; start + length <= N; start += stride) {
      const labels = new Set(series.regime.slice(start, start + length))
      const row = ROWS - (count % ROWS)
      const seg: Seg = { from: [start, row], to: [start + length - 1, row] }
      if (labels.size > 1) mixed.push(seg)
      else pure[[...labels][0]].push(seg)
      count++
    }
    return { pure, mixed, count }
  }, [series, length, stride])
  const t1 = useAxis({ label: 'time step t', range: [0, N] })
  const y1 = useAxis({ label: 'x' })
  const t2 = useAxis({ label: 'time step t', range: [0, N] })
  const y2 = useAxis({ label: 'window', range: [0, ROWS + 1] })
  const overlap = Math.max(0, 1 - stride / length)
  return (
    <Figure
      title="Sliding windows over a labelled series"
      state={state}
      hoverReadout={false}
      caption={`Top: a series that switches between a slowly wandering activity and a rapidly alternating one at ${CHANGEPOINTS.length} labelled changes. Bottom: the windows, staggered over ${ROWS} rows so that overlapping ones can be told apart. A window inside one activity takes that activity's colour; a window that straddles a change, in grey, has no single label. Drag the end of the first window to set its length L and the start of the second to set the stride s. Longer windows give more context per example and more mixed windows; a stride shorter than the length makes neighbouring windows share samples.`}
      readouts={
        <>
          <Readout label="windows" value={String(windows.count)} />
          <Readout
            label="mixed windows"
            value={`${windows.mixed.length} (${Math.round((100 * windows.mixed.length) / Math.max(1, windows.count))}%)`}
          />
          <Readout label="overlap between neighbours" value={`${Math.round(100 * overlap)}%`} />
        </>
      }
    >
      <FixedHeight height={180}>
        <Plot x={t1} y={y1} title="series and its labels">
          <Curve name="slow activity" x={series.t} y={series.slow} slot={0} />
          <Curve name="fast activity" x={series.t} y={series.fast} slot={1} />
        </Plot>
      </FixedHeight>
      <FixedHeight height={200}>
        <Plot x={t2} y={y2} title="windows">
          {windows.pure[0].length > 0 && <Segments name="slow windows" segments={windows.pure[0]} slot={0} width={4} />}
          {windows.pure[1].length > 0 && <Segments name="fast windows" segments={windows.pure[1]} slot={1} width={4} />}
          {windows.mixed.length > 0 && <Segments name="mixed windows" segments={windows.mixed} muted width={4} />}
          <Handle
            kind="x"
            at={length}
            label="L"
            onDrag={(x) => state.set('length', Math.max(4, Math.min(150, Math.round(x))))}
          />
          <Handle
            kind="x"
            at={stride}
            label="s"
            onDrag={(x) => state.set('stride', Math.max(1, Math.min(150, Math.round(x))))}
          />
        </Plot>
      </FixedHeight>
    </Figure>
  )
}
