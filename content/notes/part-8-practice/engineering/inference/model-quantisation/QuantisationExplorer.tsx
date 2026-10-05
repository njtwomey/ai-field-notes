import { useMemo } from 'react'
import {
  Bars,
  Button,
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  type Segment,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

const N = 5000
const BINS = 70
const RANGE = 6

type Shape = 'gauss' | 'laplace' | 'outliers'
const SHAPES = [
  { value: 'gauss', label: 'Gaussian' },
  { value: 'laplace', label: 'Laplace' },
  { value: 'outliers', label: 'Gaussian + outliers' },
] as const

/** Unit-variance weights, fixed by the seed so that only the quantiser changes between renders. */
function sampleWeights(shape: Shape): number[] {
  const r = stream(7)
  return Array.from({ length: N }, (_, i) => {
    if (shape === 'laplace') {
      const u = uniform(r) - 0.5
      return (-Math.sign(u) * Math.log(1 - 2 * Math.abs(u))) / Math.SQRT2
    }
    // One weight in 200 is an outlier of about ±5 standard deviations.
    if (shape === 'outliers' && i % 200 === 0) return (i % 400 === 0 ? 1 : -1) * (5 + 0.3 * normal(r))
    return normal(r)
  })
}

/** Symmetric uniform quantiser with clipping range [-c, c]: split the error into rounding and clipping parts. */
function quantise(w: number[], bits: number, c: number) {
  const qmax = 2 ** (bits - 1) - 1
  const s = c / qmax
  let rounding = 0
  let clipping = 0
  let inside = 0
  let power = 0
  for (const x of w) {
    const q = Math.max(-qmax, Math.min(qmax, Math.round(x / s)))
    const e = (s * q - x) ** 2
    if (Math.abs(x) > c) clipping += e
    else {
      rounding += e
      inside++
    }
    power += x * x
  }
  // Rounding error is averaged over the unclipped weights, where the s²/12 model applies; the total is per weight.
  const total = (rounding + clipping) / w.length
  return {
    s,
    qmax,
    rounding: rounding / Math.max(inside, 1),
    clipping: clipping / w.length,
    total,
    power: power / w.length,
  }
}

export function QuantisationExplorer() {
  const state = useFigureState({
    shape: choice<Shape>(SHAPES, 'gauss', { label: 'weight distribution' }),
    bits: int(4, { min: 2, max: 8, step: 1, label: 'bits b' }),
    clip: float(3, { min: 0.2, max: RANGE, step: 0.05, label: 'clipping range c (standard deviations)' }),
  })
  const weights = useMemo(() => sampleWeights(state.shape), [state.shape])
  const res = useMemo(() => quantise(weights, state.bits, state.clip), [weights, state.bits, state.clip])

  const histogram = useMemo(() => {
    const width = (2 * RANGE) / BINS
    const counts = new Array(BINS).fill(0)
    for (const x of weights) {
      const i = Math.floor((x + RANGE) / width)
      if (i >= 0 && i < BINS) counts[i]++
    }
    return [
      {
        name: 'weights',
        x: counts.map((_, i) => -RANGE + (i + 0.5) * width),
        y: counts.map((k) => k / (N * width)),
        slot: 0,
      },
    ] as const
  }, [weights])

  // One vertical tick per representable value; at 7 or 8 bits they merge into a band.
  const levels = useMemo((): Segment[] => {
    const out: Segment[] = []
    for (let q = -res.qmax; q <= res.qmax; q++) out.push({ from: [q * res.s, 0], to: [q * res.s, 0.62] })
    return out
  }, [res.qmax, res.s])

  const bestClip = () => {
    let best = state.bind('clip').min
    let bestMse = Infinity
    for (let c = state.bind('clip').min; c <= state.bind('clip').max; c += state.bind('clip').step) {
      const r = quantise(weights, state.bits, c)
      if (r.total < bestMse) {
        bestMse = r.total
        best = c
      }
    }
    state.set('clip', best)
  }

  const mse = res.total

  const xAxis = useAxis({ label: 'weight value (standard deviations)', range: [-RANGE, RANGE] })
  const yAxis = useAxis({ label: 'density', range: [0, 0.75] })
  return (
    <Figure
      title="Quantising 5,000 unit-variance weights"
      purpose="Change the bit width and drag the clipping range to trade clipping error against rounding error."
      state={state}
      caption="Grey ticks are the representable values of a symmetric b-bit quantiser with clipping range [−c, c]. Drag either dashed line to move c. A wide range keeps outliers but spaces the levels far apart; a narrow range packs the levels together but clips the tails. The best c shrinks as the bit width falls."
      controls={
        <>
          <Button variant="outline" size="sm" onClick={bestClip}>
            Minimise MSE over c
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label="step s = c / (2^(b−1) − 1)" value={formatNumber(res.s)} />
          <Readout label="rounding MSE inside [−c, c]" value={formatNumber(res.rounding)} />
          <Readout label="s²/12 prediction" value={formatNumber((res.s * res.s) / 12)} />
          <Readout label="clipping MSE per weight" value={formatNumber(res.clipping)} />
          <Readout label="total MSE" value={formatNumber(mse)} />
          <Readout label="SQNR" value={`${formatNumber(10 * Math.log10(res.power / mse))} dB`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Bars {...histogram[0]} />
        <Segments segments={levels} />
        <Handle kind="x" at={state.clip} label="+c" onDrag={(x) => state.set('clip', Math.abs(x))} />
        <Handle kind="x" at={-state.clip} label="−c" onDrag={(x) => state.set('clip', Math.abs(x))} />
      </Plot>
    </Figure>
  )
}
