import { useContext, useMemo, type ReactNode } from 'react'
import { Curve, Figure, FrameContext, Handle, int, Plot, Readout, Segments, useAxis, useFigureState } from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { seasonalSeries } from 'aifn-methods/data/synthetic'

const T = 120

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

type Seg = { from: [number, number]; to: [number, number] }

/** How one series becomes many training examples: a look-back window as input and the next H values as target. */
export function ForecastExamples() {
  const state = useFigureState({
    lookback: int(24, { ge: 1, le: 60, suggestions: [12, 24, 48], label: 'look-back L' }),
    horizon: int(6, { ge: 1, le: 24, suggestions: [1, 6, 12], label: 'horizon H' }),
    stride: int(3, { ge: 1, le: 24, suggestions: [1, 3, 12], label: 'stride between origins' }),
    origin: int(70, { ge: 1, le: T - 1, label: 'forecast origin' }),
  })
  const { lookback, horizon, stride, origin } = state
  const series = useMemo(() => {
    const y = toFlat(
      seasonalSeries(stream('forecast-examples'), { n: T, period: 12, amplitudes: [1.5], trend: 0.03, noise: 0.4 }).y,
    )
    return { t: y.map((_, i) => i), y }
  }, [])
  const view = useMemo(() => {
    const o = Math.max(lookback, Math.min(T - horizon, origin))
    const mask = (from: number, to: number) => series.y.map((v, i) => (i >= from && i < to ? v : NaN))
    const inputs: Seg[] = []
    const targets: Seg[] = []
    let count = 0
    // Origin o: input y[o - L .. o - 1], target y[o .. o + H - 1].
    for (let start = lookback; start + horizon <= T; start += stride) {
      const row = count + 1
      inputs.push({ from: [start - lookback, row], to: [start - 1, row] })
      targets.push({ from: [start, row], to: [start + horizon - 1, row] })
      count++
    }
    return { o, input: mask(o - lookback, o), target: mask(o, o + horizon), inputs, targets, count }
  }, [series, lookback, horizon, stride, origin])
  const t1 = useAxis({ label: 'time t', range: [0, T] })
  const y1 = useAxis({ label: 'y' })
  const t2 = useAxis({ label: 'time t', range: [0, T] })
  const y2 = useAxis({ label: 'example', key: view.count })
  return (
    <Figure
      title="From one series to training examples"
      state={state}
      hoverReadout={false}
      caption={`Top: a monthly series of ${T} values. At the forecast origin, the input is the look-back window of the last L values and the target is the next H values. Drag the origin to move the example. Bottom: every example the series yields when the origin moves in steps of the stride, one row per example, the first at the bottom; inputs in one colour, targets in the other. Neighbouring examples overlap whenever the stride is shorter than L + H, and a target value of one example is an input value of a later one, so examples must be split by time, not at random.`}
      readouts={
        <>
          <Readout label="examples" value={String(view.count)} />
          <Readout label="values per example" value={`${lookback} in, ${horizon} out`} />
        </>
      }
    >
      <FixedHeight height={200}>
        <Plot x={t1} y={y1} title="one example">
          <Curve name="series" x={series.t} y={series.y} muted />
          <Curve name="look-back input" x={series.t} y={view.input} slot={0} width={3} />
          <Curve name="horizon target" x={series.t} y={view.target} slot={1} width={3} />
          <Handle
            kind="x"
            at={view.o}
            label="origin"
            onDrag={(x) => state.set('origin', Math.max(1, Math.min(T - 1, Math.round(x))))}
          />
        </Plot>
      </FixedHeight>
      <FixedHeight height={200}>
        <Plot x={t2} y={y2} title="all examples">
          {view.inputs.length > 0 && <Segments name="inputs" segments={view.inputs} slot={0} width={2} />}
          {view.targets.length > 0 && <Segments name="targets" segments={view.targets} slot={1} width={2} />}
        </Plot>
      </FixedHeight>
    </Figure>
  )
}
