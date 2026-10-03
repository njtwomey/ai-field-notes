import { useMemo } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type Segment,
} from 'aifn-render'
import { linspace } from '@/lib/math'
import { eigSym } from '@/lib/math/mat2'

const T = 400
const SHOWN = 200
const ARROW = 2

/** Two unit-variance sources: a sine wave and a sawtooth with a different period. */
const SOURCES = (() => {
  const s1 = Array.from({ length: T }, (_, t) => Math.SQRT2 * Math.sin((2 * Math.PI * t) / 50))
  const s2 = Array.from({ length: T }, (_, t) => Math.sqrt(3) * (2 * (((t + 11) / 37) % 1) - 1))
  return [s1, s2]
})()

/** Mixtures x = A s, centred and whitened: z = Λ^(-1/2) Eᵀ x has identity covariance. */
function whitened() {
  const a = [
    [1, 0.6],
    [0.4, 1],
  ]
  const x = SOURCES[0].map((_, t) => [
    a[0][0] * SOURCES[0][t] + a[0][1] * SOURCES[1][t],
    a[1][0] * SOURCES[0][t] + a[1][1] * SOURCES[1][t],
  ])
  const m = [0, 1].map((j) => x.reduce((s, p) => s + p[j], 0) / T)
  const c = x.map((p) => [p[0] - m[0], p[1] - m[1]])
  const sxx = c.reduce((s, p) => s + p[0] * p[0], 0) / T
  const sxy = c.reduce((s, p) => s + p[0] * p[1], 0) / T
  const syy = c.reduce((s, p) => s + p[1] * p[1], 0) / T
  const { values, vectors } = eigSym(sxx, sxy, syy)
  return c.map((p) => vectors.map((v, j) => (v[0] * p[0] + v[1] * p[1]) / Math.sqrt(values[j])) as [number, number])
}

const rad = (deg: number) => (deg * Math.PI) / 180
const fold = (deg: number) => ((((deg + 90) % 180) + 180) % 180) - 90

/** Excess kurtosis of the projection yₜ = wᵀzₜ; the projection has unit variance because z is white. */
function kurtosis(z: [number, number][], w: [number, number]) {
  return z.reduce((s, p) => s + (w[0] * p[0] + w[1] * p[1]) ** 4, 0) / z.length - 3
}

const corr = (a: number[], b: number[]) => {
  const ma = a.reduce((s, v) => s + v, 0) / a.length
  const mb = b.reduce((s, v) => s + v, 0) / b.length
  let ab = 0
  let aa = 0
  let bb = 0
  a.forEach((v, i) => {
    ab += (v - ma) * (b[i] - mb)
    aa += (v - ma) ** 2
    bb += (b[i] - mb) ** 2
  })
  return ab / Math.sqrt(aa * bb)
}

export function UnmixingDirection() {
  const angle = useParam(20, { min: -90, max: 90, step: 1 })
  const data = useMemo(() => {
    const z = whitened()
    const thetas = linspace(-90, 90, 181)
    const kurt = thetas.map((t) => kurtosis(z, [Math.cos(rad(t)), Math.sin(rad(t))]))
    return { z, thetas, kurt }
  }, [])
  const w: [number, number] = [Math.cos(rad(angle.value)), Math.sin(rad(angle.value))]
  const y = data.z.map((p) => w[0] * p[0] + w[1] * p[1])
  const k = kurtosis(data.z, w)
  const c1 = corr(y, SOURCES[0])
  const c2 = corr(y, SOURCES[1])

  // One FastICA fixed-point step with the kurtosis contrast: w ← E[z (wᵀz)³] − 3w, then normalise.
  const fastIcaStep = () => {
    const next = [0, 1].map((j) => data.z.reduce((s, p, t) => s + p[j] * y[t] ** 3, 0) / T - 3 * w[j])
    angle.set(fold((Math.atan2(next[1], next[0]) * 180) / Math.PI))
  }

  const vectors: Segment[] = [{ from: [0, 0], to: [ARROW * w[0], ARROW * w[1]] }]
  const tip: Handle[] = [
    {
      kind: 'point',
      at: [ARROW * w[0], ARROW * w[1]],
      label: 'w',
      onDrag: ([px, py]) => angle.set(fold((Math.atan2(py, px) * 180) / Math.PI)),
    },
  ]
  const onCurve: Handle[] = [{ kind: 'x', at: angle.value, label: 'direction', onDrag: (x) => angle.set(x) }]
  const ts = Array.from({ length: SHOWN }, (_, t) => t)

  return (
    <Interactive
      title="Finding an independent component by non-Gaussianity"
      caption="A sine wave and a sawtooth are mixed linearly, then whitened, which leaves the scatter a rotated square. Drag the tip of w, or the line on the kurtosis curve, or press FastICA step. The projection wᵀz is a single source exactly where its kurtosis is most negative: along the square's sides. Midway between them it is an equal mix, whose kurtosis is closer to the Gaussian value 0."
      controls={
        <>
          <ParamSlider label="direction of w (degrees)" param={angle} format={(v) => `${v}°`} />
          <ParamButton onClick={fastIcaStep}>FastICA step</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="excess kurtosis of wᵀz" value={formatNumber(k)} />
          <Readout label="|corr| with sine" value={formatNumber(Math.abs(c1))} />
          <Readout label="|corr| with sawtooth" value={formatNumber(Math.abs(c2))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          equalAspect
          xRange={[-2.5, 2.5]}
          yRange={[-2.5, 2.5]}
          xLabel="z₁ (whitened)"
          yLabel="z₂ (whitened)"
          vectors={vectors}
          handles={tip}
          series={[
            {
              name: 'whitened mixtures',
              type: 'scatter',
              x: data.z.map((p) => p[0]),
              y: data.z.map((p) => p[1]),
              slot: 0,
            },
          ]}
        />
        <XYChart
          height={320}
          xLabel="direction of w (degrees)"
          yLabel="excess kurtosis"
          xRange={[-90, 90]}
          handles={onCurve}
          series={[
            { name: 'kurtosis of wᵀz', type: 'line', x: data.thetas, y: data.kurt, slot: 1 },
            { name: 'current w', type: 'scatter', x: [angle.value], y: [k], emphasis: true },
          ]}
        />
      </div>
      <XYChart
        height={200}
        xLabel="time"
        yLabel="wᵀz"
        series={[{ name: 'projection wᵀzₜ', type: 'line', x: ts, y: y.slice(0, SHOWN), slot: 2 }]}
      />
    </Interactive>
  )
}
