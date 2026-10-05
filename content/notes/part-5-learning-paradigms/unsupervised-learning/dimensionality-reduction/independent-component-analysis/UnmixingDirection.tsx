import { useMemo } from 'react'
import {
  Button,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  type Segment,
  slider,
  useAxis,
  useFigureState,
  Vectors,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { eigh2 } from 'aifn-compute/numerics/linalg'

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
  const { values, vectors } = eigh2([
    [sxx, sxy],
    [sxy, syy],
  ])
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
  const state = useFigureState({
    angle: slider(-90, 90, 20, { step: 1, label: 'direction of w (degrees)', format: (v) => `${v}°` }),
  })
  const data = useMemo(() => {
    const z = whitened()
    const thetas = toFlat(linspace(-90, 90, 181))
    const kurt = thetas.map((t) => kurtosis(z, [Math.cos(rad(t)), Math.sin(rad(t))]))
    return { z, thetas, kurt }
  }, [])
  const w: [number, number] = [Math.cos(rad(state.angle)), Math.sin(rad(state.angle))]
  const y = data.z.map((p) => w[0] * p[0] + w[1] * p[1])
  const k = kurtosis(data.z, w)
  const c1 = corr(y, SOURCES[0])
  const c2 = corr(y, SOURCES[1])

  // One FastICA fixed-point step with the kurtosis contrast: w ← E[z (wᵀz)³] − 3w, then normalise.
  const fastIcaStep = () => {
    const next = [0, 1].map((j) => data.z.reduce((s, p, t) => s + p[j] * y[t] ** 3, 0) / T - 3 * w[j])
    state.set('angle', fold((Math.atan2(next[1], next[0]) * 180) / Math.PI))
  }

  const vectors: Segment[] = [{ from: [0, 0], to: [ARROW * w[0], ARROW * w[1]] }]
  const ts = Array.from({ length: SHOWN }, (_, t) => t)

  const xAxis = useAxis({ label: 'z₁ (whitened)', range: [-2.5, 2.5] })
  const yAxis = useAxis({ label: 'z₂ (whitened)', range: [-2.5, 2.5], equal: xAxis })
  const xAxis2 = useAxis({ label: 'direction of w (degrees)', range: [-90, 90] })
  const yAxis2 = useAxis({ label: 'excess kurtosis', hold: 'union' })
  const xAxis3 = useAxis({ label: 'time', hold: 'union' })
  const yAxis3 = useAxis({ label: 'wᵀz', hold: 'union' })
  return (
    <Figure
      title="Finding an independent component by non-Gaussianity"
      state={state}
      caption="A sine wave and a sawtooth are mixed linearly, then whitened, which leaves the scatter a rotated square. Drag the tip of w, or the line on the kurtosis curve, or press FastICA step. The projection wᵀz is a single source exactly where its kurtosis is most negative: along the square's sides. Midway between them it is an equal mix, whose kurtosis is closer to the Gaussian value 0."
      controls={
        <>
          <Button variant="outline" size="sm" onClick={fastIcaStep}>
            FastICA step
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label="excess kurtosis of wᵀz" value={formatNumber(k)} />
          <Readout label="|corr| with sine" value={formatNumber(Math.abs(c1))} />
          <Readout label="|corr| with sawtooth" value={formatNumber(Math.abs(c2))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          <Points name="whitened mixtures" x={data.z.map((p) => p[0])} y={data.z.map((p) => p[1])} slot={0} />
          <Vectors vectors={vectors} />
          <Handle
            kind="point"
            at={[ARROW * w[0], ARROW * w[1]]}
            label="w"
            onDrag={([px, py]) => state.set('angle', fold((Math.atan2(py, px) * 180) / Math.PI))}
          />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          <Curve name="kurtosis of wᵀz" x={data.thetas} y={data.kurt} slot={1} />
          <Points name="current w" x={[state.angle]} y={[k]} emphasis />
          <Handle {...state.handle('angle', { label: 'direction' })} />
        </Plot>
      </div>
      <Plot x={xAxis3} y={yAxis3} height={200}>
        <Curve name="projection wᵀzₜ" x={ts} y={y.slice(0, SHOWN)} slot={2} />
      </Plot>
    </Figure>
  )
}
