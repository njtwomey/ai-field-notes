import { useMemo } from 'react'
import {
  Bars,
  choice,
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  type Segment,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

const N = 20000
const LO = -50
const HI = 30
const BIN = 1

type Format = 'fp16' | 'bf16'
const FORMATS = [
  { value: 'fp16', label: 'fp16' },
  { value: 'bf16', label: 'bf16' },
] as const

/** log2 of the smallest subnormal, smallest normal and largest finite value. */
const LIMITS: Record<Format, { sub: number; normal: number; max: number }> = {
  fp16: { sub: -24, normal: -14, max: Math.log2(65504) },
  bf16: { sub: -133, normal: -126, max: 128 },
}

/** Illustrative gradient magnitudes: log2 |g| roughly normal around 2^-22, as for activation gradients deep in a net. */
const LOG_G = (() => {
  const r = stream(3)
  return Array.from({ length: N }, () => -22 + 5 * normal(r))
})()

export function LossScaling() {
  const state = useFigureState({
    format: choice<Format>(FORMATS, 'fp16', { label: 'format' }),
    k: int(0, { min: 0, max: 32, step: 1, label: 'loss scale exponent k (scale 2^k)' }),
  })
  const lim = LIMITS[state.format]

  const { series, under, sub, over } = useMemo(() => {
    const bins = Math.round((HI - LO) / BIN)
    const counts = [0, 1, 2].map(() => new Array<number>(bins).fill(0))
    let under = 0
    let sub = 0
    let over = 0
    for (const v of LOG_G) {
      const x = v + state.k
      // Magnitudes below half the smallest subnormal round to zero.
      const cls = x < lim.sub - 1 ? 0 : x > lim.max ? 2 : 1
      if (cls === 0) under++
      else if (cls === 2) over++
      else if (x < lim.normal) sub++
      const i = Math.floor((Math.min(Math.max(x, LO), HI - 1e-9) - LO) / BIN)
      counts[cls][i]++
    }
    const xs = counts[0].map((_, i) => LO + (i + 0.5) * BIN)
    const series = [
      { name: 'rounds to zero', x: xs, y: counts[0].map((c) => c / N), muted: true },
      { name: 'representable', x: xs, y: counts[1].map((c) => c / N), slot: 0 },
      { name: 'overflows to inf', x: xs, y: counts[2].map((c) => c / N), slot: 1 },
    ] as const
    return { series, under: under / N, sub: sub / N, over: over / N }
  }, [state.k, lim])

  const bounds = useMemo((): Segment[] => {
    const lines = [lim.sub, lim.normal, lim.max].filter((v) => v > LO && v < HI)
    return lines.map((v) => ({ from: [v, 0], to: [v, 0.09] }))
  }, [lim])

  const xAxis = useAxis({ label: 'log2 |scaled gradient|', range: [LO, HI] })
  const yAxis = useAxis({ label: 'fraction of values', range: [0, 0.1] })
  return (
    <Figure
      title="Loss scaling shifts gradients into the representable range"
      purpose="Change the loss scale and the number format to see which scaled gradients underflow, lose precision or overflow."
      state={state}
      caption="A histogram of 20,000 illustrative gradient magnitudes on a log2 scale, after multiplying the loss by 2^k. Grey lines mark the smallest subnormal, the smallest normal and the largest finite number of the format. In fp16 many unscaled gradients round to zero; a scale of about 2^8 to 2^16 rescues them without overflow. bf16 has the exponent range of fp32 and needs no scaling."

      readouts={
        <>
          <Readout label="round to zero" value={`${formatNumber(100 * under)}%`} />
          <Readout label="subnormal (reduced precision)" value={`${formatNumber(100 * sub)}%`} />
          <Readout label="overflow" value={`${formatNumber(100 * over)}%`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Bars {...series[0]} />
        <Bars {...series[1]} />
        <Bars {...series[2]} />
        <Segments segments={bounds} />
      </Plot>
    </Figure>
  )
}
