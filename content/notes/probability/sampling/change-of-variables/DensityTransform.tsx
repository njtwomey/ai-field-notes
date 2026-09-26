import { useMemo, useState } from 'react'
import {
  Interactive,
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
import { linspace, rng, sigmoid } from '@/lib/math'
import { normalPdf } from '@/lib/math/special'

type MapId = 'affine' | 'exp' | 'logistic' | 'square'

type Transform = {
  label: string
  g: (x: number) => number
  /** |g'(x)|, the local stretch factor. */
  stretch: (x: number) => number
  /** Density of Y = g(X) for X ~ N(0, 1), from the change-of-variables formula. */
  density: (y: number) => number
  yRange: [number, number]
  /** Top of the density axis when the density is unbounded, e.g. near y = 0 for y = x². */
  densityMax?: number
}

const TRANSFORMS: Record<MapId, Transform> = {
  affine: {
    label: 'y = 2x + 1',
    g: (x) => 2 * x + 1,
    stretch: () => 2,
    density: (y) => normalPdf((y - 1) / 2) / 2,
    yRange: [-6, 8],
  },
  exp: {
    label: 'y = eˣ',
    g: Math.exp,
    stretch: Math.exp,
    density: (y) => (y > 0 ? normalPdf(Math.log(y)) / y : 0),
    yRange: [0, 6],
  },
  logistic: {
    label: 'y = σ(2x)',
    g: (x) => sigmoid(2 * x),
    stretch: (x) => 2 * sigmoid(2 * x) * (1 - sigmoid(2 * x)),
    density: (y) => (y > 0 && y < 1 ? normalPdf(Math.log(y / (1 - y)) / 2) / (2 * y * (1 - y)) : 0),
    yRange: [0, 1],
  },
  square: {
    label: 'y = x² (not one-to-one)',
    g: (x) => x * x,
    stretch: (x) => 2 * Math.abs(x),
    // Two preimages, ±√y, each contribute φ(√y) / (2√y).
    density: (y) => (y > 0 ? normalPdf(Math.sqrt(y)) / Math.sqrt(y) : 0),
    yRange: [0, 5],
    densityMax: 1.5,
  },
}

const X_RANGE: [number, number] = [-3, 3]
const WIDTH = 0.25
const BINS = 50

/**
 * Left: the map y = g(x), with a short interval [x, x + 0.25] carried to its image. Right: a histogram of Y = g(X) for
 * X ~ N(0, 1) against the density the change-of-variables formula predicts.
 */
export function DensityTransform() {
  const [id, setId] = useState<MapId>('exp')
  const x = useParam(0.5, { min: -2.75, max: 2.5, step: 0.05 })
  const t = TRANSFORMS[id]
  const [y0, y1] = t.yRange

  const curve = useMemo((): XYSeries[] => {
    const xs = linspace(X_RANGE[0], X_RANGE[1], 241)
    return [{ name: 'g(x)', type: 'line', x: xs, y: xs.map(t.g), slot: 0 }]
  }, [t])

  const histogram = useMemo((): XYSeries[] => {
    const n = 20_000
    const r = rng(7)
    const width = (y1 - y0) / BINS
    const counts = new Array(BINS).fill(0)
    for (let i = 0; i < n; i++) {
      const bin = Math.floor((t.g(r.normal()) - y0) / width)
      if (bin >= 0 && bin < BINS) counts[bin]++
    }
    const ys = linspace(y0 + 1e-3, y1 - 1e-3, 300)
    return [
      {
        name: 'samples of g(X)',
        type: 'bar',
        x: counts.map((_, i) => y0 + (i + 0.5) * width),
        y: counts.map((c) => c / (n * width)),
        slot: 0,
      },
      { name: 'formula p_Y(y)', type: 'line', x: ys, y: ys.map(t.density), slot: 1 },
    ]
  }, [t, y0, y1])

  // The interval [x, x + WIDTH] on the x-axis, carried up to the curve and across to the y-axis.
  const a = x.value
  const b = a + WIDTH
  const clampY = (v: number) => Math.min(Math.max(v, y0), y1)
  const segments: Segment[] = [a, b].flatMap((v): Segment[] => [
    { from: [v, y0], to: [v, clampY(t.g(v))] },
    { from: [v, clampY(t.g(v))], to: [X_RANGE[0], clampY(t.g(v))] },
  ])
  const handles: Handle[] = [{ kind: 'x', at: a, label: 'x', onDrag: x.set }]
  const imageWidth = Math.abs(t.g(b) - t.g(a))

  return (
    <Interactive
      title="Stretching space thins out density"
      caption="X is standard Gaussian. On the left, the interval [x, x + 0.25] is carried through g to an interval on the y-axis. Probability is conserved, so where g stretches the interval the density of Y falls by the same factor. Drag the line labelled x to move the interval. On the right, a histogram of 20,000 transformed draws matches the density from the formula. For y = x², two values of x reach each y, and the formula adds both contributions."
      controls={
        <>
          <ParamChoice
            label="map"
            value={id}
            onChange={setId}
            options={(Object.keys(TRANSFORMS) as MapId[]).map((k) => ({ value: k, label: TRANSFORMS[k].label }))}
          />
          <ParamSlider label="x" param={x} />
        </>
      }
      readout={
        <>
          <Readout label="p_X(x)" value={formatNumber(normalPdf(a))} />
          <Readout label="stretch |g′(x)|" value={formatNumber(t.stretch(a))} />
          <Readout label="image width / 0.25" value={formatNumber(imageWidth / WIDTH)} />
          <Readout label="p_X(x) / |g′(x)|" value={formatNumber(normalPdf(a) / t.stretch(a))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          height={300}
          series={curve}
          segments={segments}
          xRange={X_RANGE}
          yRange={t.yRange}
          xLabel="x"
          yLabel="y = g(x)"
          handles={handles}
        />
        <XYChart
          height={300}
          series={histogram}
          xRange={t.yRange}
          yRange={[0, t.densityMax]}
          xLabel="y"
          yLabel="density"
        />
      </div>
    </Interactive>
  )
}
