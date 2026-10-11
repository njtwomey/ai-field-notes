import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { seasonalSeries } from 'aifn-methods/data/synthetic'
import { oneHotEncoder } from 'aifn-methods/learning/preprocessing'
import { linearRegression } from 'aifn-methods/learning/linear'
import { rootMeanSquaredError } from 'aifn-compute/learning/metrics'
import { stream } from 'aifn-compute/foundation/random'
import { concat, fromData, slice, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

const WEEKS = 20
const DAYS = 7 * WEEKS
/** The last four weeks are held out: the model is fitted on the past and scored on the future. */
const TRAIN = DAYS - 28
const days = Array.from({ length: DAYS }, (_, t) => t)

const ENCODINGS = [
  { value: 'none', label: 'elapsed days only' },
  { value: 'number', label: '+ weekday as a number 0–6' },
  { value: 'indicators', label: '+ weekday indicators' },
] as const
type Encoding = (typeof ENCODINGS)[number]['value']

/** The design matrix for each encoding: elapsed days, plus the weekday read off the day index. */
function design(encoding: Encoding) {
  const elapsed = fromData(Float64Array.from(days), [DAYS, 1])
  // Day 0 is a Monday; the weekday is the day index modulo 7.
  const weekday = days.map((t) => t % 7)
  if (encoding === 'none') return elapsed
  if (encoding === 'number') return concat([elapsed, fromData(Float64Array.from(weekday), [DAYS, 1])], 1)
  const indicators = oneHotEncoder({ drop: 'first' }).fit({ x: weekday }).transform(weekday)
  return concat([elapsed, indicators], 1)
}

/** A daily series with a trend and a weekly cycle, fitted by least squares with three encodings of the weekday. */
export function WeekdayFeatures() {
  const state = useFigureState({
    encoding: choice(ENCODINGS, 'indicators', { label: 'features' }),
    second: slider(0, 1, 0.6, { step: 0.05, label: 'second weekly harmonic' }),
    noise: slider(0, 1, 0.3, { step: 0.05, label: 'noise sd' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { encoding, second, noise, seed } = state
  const y = useMemo(
    () =>
      seasonalSeries(stream(`date-and-time-features/${seed}`), {
        n: DAYS,
        period: 7,
        amplitudes: [1, second],
        trend: 0.02,
        noise,
      }).y as Tensor,
    [second, noise, seed],
  )
  const fits = useMemo(
    () =>
      Object.fromEntries(
        ENCODINGS.map(({ value }) => {
          const x = design(value)
          const model = linearRegression().fit({
            kind: 'dataset',
            x: slice(x, [0, TRAIN]) as Tensor,
            y: slice(y, [0, TRAIN]) as Tensor,
          })
          const fitted = model.decide(x)
          const test = rootMeanSquaredError(slice(y, [TRAIN, null]) as Tensor, slice(fitted, [TRAIN, null]) as Tensor)
          return [value, { fitted: Array.from(toFlat(fitted)), test }]
        }),
      ) as Record<Encoding, { fitted: number[]; test: number }>,
    [y],
  )
  const values = Array.from(toFlat(y))
  const t = useAxis({ label: 'day (day 0 is a Monday)', range: [0, DAYS - 1] })
  const v = useAxis({ label: 'value' })
  return (
    <Figure
      title="Weekday features for a daily series"
      state={state}
      hoverReadout={false}
      caption="A daily series with a linear trend and a weekly cycle made of two harmonics, so the weekly shape is not a single sine wave. Least squares is fitted on the first sixteen weeks (coloured points) and scored on the last four (grey points). With elapsed days only, the model fits the trend and misses the cycle. The weekday as one number 0 to 6 forces the weekly effect to be a straight line from Monday to Sunday. Six indicator columns, one per weekday after Monday, let each weekday have its own level and capture any weekly shape."
      readouts={
        <>
          {ENCODINGS.map(({ value, label }) => (
            <Readout key={value} label={`held-out RMSE, ${label}`} value={formatNumber(fits[value].test)} />
          ))}
        </>
      }
    >
      <Plot x={t} y={v}>
        <Points name="training days" x={days.slice(0, TRAIN)} y={values.slice(0, TRAIN)} slot={0} />
        <Points name="held-out days" x={days.slice(TRAIN)} y={values.slice(TRAIN)} muted />
        <Curve name="fitted and forecast" x={days} y={fits[encoding].fitted} emphasis />
      </Plot>
    </Figure>
  )
}
