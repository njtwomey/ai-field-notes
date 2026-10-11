import { useMemo } from 'react'
import { Annotation, choice, Curve, Figure, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { seasonalSeries } from 'aifn-methods/data/synthetic'
import { standardScaler } from 'aifn-methods/learning/preprocessing'

const N = 120
const TRAIN = 84
const SERIES = [
  { name: 'large store', level: 1000, amplitude: 200, trend: 4, noise: 40 },
  { name: 'medium store', level: 120, amplitude: 30, trend: -0.3, noise: 8 },
  { name: 'small store', level: 12, amplitude: 4, trend: 0.15, noise: 1.5 },
] as const
const MODES = [
  { value: 'raw', label: 'raw values' },
  { value: 'global', label: 'global: one mean and scale for all series' },
  { value: 'series', label: 'per series: each series by its own training values' },
  { value: 'window', label: 'per window: each window by its own values' },
] as const
type Mode = (typeof MODES)[number]['value']

const column = (values: number[]) => tensor(values.map((v) => [v]))

/** Fit a standard scaler on `fit` and apply it to `apply`. */
function scaled(fit: number[], apply: number[]) {
  return toFlat(
    standardScaler()
      .fit({ x: column(fit) })
      .transform(column(apply)),
  )
}

/** Three series of very different size, put on one scale in four ways. */
export function NormaliseSeries() {
  const state = useFigureState({
    mode: choice(MODES, 'raw', { label: 'normalisation' }),
    window: int(24, { ge: 4, le: 60, suggestions: [12, 24, 48], label: 'window length L (per window)' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { mode, window, seed } = state
  const raw = useMemo(
    () =>
      SERIES.map(({ level, amplitude, trend, noise }, k) =>
        toFlat(
          seasonalSeries(stream(`normalise/${seed}/${k}`), {
            n: N,
            period: 12,
            level,
            amplitudes: [amplitude],
            trend,
            noise,
          }).y,
        ),
      ),
    [seed],
  )
  const view = useMemo(() => {
    const t = Array.from({ length: N }, (_, i) => i)
    const transform: Record<Mode, () => number[][]> = {
      raw: () => raw,
      global: () => {
        const pooled = raw.flatMap((y) => y.slice(0, TRAIN))
        return raw.map((y) => scaled(pooled, y))
      },
      series: () => raw.map((y) => scaled(y.slice(0, TRAIN), y)),
      window: () =>
        raw.map((y) => {
          const out: number[] = []
          for (let start = 0; start < N; start += window) {
            const piece = y.slice(start, start + window)
            out.push(...scaled(piece, piece))
          }
          return out
        }),
    }
    const ys = transform[mode]()
    // How far the test stretch leaves the range the training stretch occupied, per series.
    const beyond = ys.map((y) => {
      const train = y.slice(0, TRAIN)
      const [lo, hi] = [Math.min(...train), Math.max(...train)]
      return y.slice(TRAIN).filter((v) => v < lo || v > hi).length / (N - TRAIN)
    })
    return { t, ys, beyond }
  }, [raw, mode, window])
  const tAxis = useAxis({ label: 'month t', range: [0, N] })
  const yAxis = useAxis({ label: mode === 'raw' ? 'sales' : 'normalised value', key: mode })
  return (
    <Figure
      title="Putting series of different sizes on one scale"
      state={state}
      hoverReadout={false}
      caption={`Monthly sales of three stores that differ in size by a factor of about a hundred, each with a yearly season and its own trend. The first ${TRAIN} months are the training stretch and the rest the test stretch. Raw, the small stores are flat lines at the bottom. A global standardisation uses one mean and scale for all three and leaves them as different as before. Per-series standardisation, fitted on each series' training stretch, puts all three on one scale; the two stores whose trend keeps rising then leave their training range in the test stretch. Per-window standardisation rescales every window by its own mean and standard deviation, so each window starts from the same scale and level whatever the trend has done; the level and scale it removes must be restored to the forecasts.`}
      readouts={
        <>
          {SERIES.map((s, k) => (
            <Readout
              key={s.name}
              label={`${s.name}: test months outside the training range`}
              value={`${Math.round(100 * view.beyond[k])}%`}
            />
          ))}
        </>
      }
    >
      <Plot x={tAxis} y={yAxis}>
        {SERIES.map((s, k) => (
          <Curve key={s.name} name={s.name} x={view.t} y={view.ys[k]} slot={k} />
        ))}
        <Annotation x={TRAIN} text="training | test" dashed />
      </Plot>
    </Figure>
  )
}
