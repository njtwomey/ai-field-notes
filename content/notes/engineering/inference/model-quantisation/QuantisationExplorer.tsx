import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type Segment,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'

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
  const r = rng(7)
  return Array.from({ length: N }, (_, i) => {
    if (shape === 'laplace') {
      const u = r.uniform() - 0.5
      return (-Math.sign(u) * Math.log(1 - 2 * Math.abs(u))) / Math.SQRT2
    }
    // One weight in 200 is an outlier of about ±5 standard deviations.
    if (shape === 'outliers' && i % 200 === 0) return (i % 400 === 0 ? 1 : -1) * (5 + 0.3 * r.normal())
    return r.normal()
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
  const [shape, setShape] = useState<Shape>('gauss')
  const [bits, setBits] = useState(4)
  const clip = useParam(3, { min: 0.2, max: RANGE, step: 0.05 })
  const weights = useMemo(() => sampleWeights(shape), [shape])
  const res = useMemo(() => quantise(weights, bits, clip.value), [weights, bits, clip.value])

  const histogram = useMemo((): XYSeries[] => {
    const width = (2 * RANGE) / BINS
    const counts = new Array(BINS).fill(0)
    for (const x of weights) {
      const i = Math.floor((x + RANGE) / width)
      if (i >= 0 && i < BINS) counts[i]++
    }
    return [
      {
        name: 'weights',
        type: 'bar',
        x: counts.map((_, i) => -RANGE + (i + 0.5) * width),
        y: counts.map((k) => k / (N * width)),
        slot: 0,
      },
    ]
  }, [weights])

  // One vertical tick per representable value; at 7 or 8 bits they merge into a band.
  const levels = useMemo((): Segment[] => {
    const out: Segment[] = []
    for (let q = -res.qmax; q <= res.qmax; q++) out.push({ from: [q * res.s, 0], to: [q * res.s, 0.62] })
    return out
  }, [res.qmax, res.s])

  const bestClip = () => {
    let best = clip.min
    let bestMse = Infinity
    for (let c = clip.min; c <= clip.max; c += clip.step) {
      const r = quantise(weights, bits, c)
      if (r.total < bestMse) {
        bestMse = r.total
        best = c
      }
    }
    clip.set(best)
  }

  const mse = res.total
  const handles: Handle[] = [
    { kind: 'x', at: clip.value, label: '+c', onDrag: (x) => clip.set(Math.abs(x)) },
    { kind: 'x', at: -clip.value, label: '−c', onDrag: (x) => clip.set(Math.abs(x)) },
  ]

  return (
    <Interactive
      title="Quantising 5,000 unit-variance weights"
      caption="Grey ticks are the representable values of a symmetric b-bit quantiser with clipping range [−c, c]. Drag either dashed line to move c. A wide range keeps outliers but spaces the levels far apart; a narrow range packs the levels together but clips the tails. The best c shrinks as the bit width falls."
      controls={
        <>
          <ParamChoice label="weight distribution" value={shape} onChange={setShape} options={SHAPES} />
          <ParamSlider label="bits b" value={bits} onChange={setBits} min={2} max={8} step={1} />
          <ParamSlider label="clipping range c (standard deviations)" param={clip} />
          <ParamButton onClick={bestClip}>Minimise MSE over c</ParamButton>
        </>
      }
      readout={
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
      <XYChart
        series={histogram}
        segments={levels}
        xLabel="weight value (standard deviations)"
        yLabel="density"
        xRange={[-RANGE, RANGE]}
        yRange={[0, 0.75]}
        handles={handles}
        height={300}
      />
    </Interactive>
  )
}
