import { useMemo, useState } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
  type Vec2,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { stream, uniform } from 'aifn-compute/foundation/random'

/** Bands of u₁ that map to rings of the Gaussian plane: small u₁ gives a large radius. */
const BANDS = [1 / 3, 2 / 3, 1]
const BAND_NAMES = ['u₁ ≤ 1/3', '1/3 < u₁ ≤ 2/3', 'u₁ > 2/3']
const band = (u1: number) => BANDS.findIndex((b) => u1 <= b)

const boxMuller = ([u1, u2]: Vec2): Vec2 => {
  const r = Math.sqrt(-2 * Math.log(u1))
  return [r * Math.cos(2 * Math.PI * u2), r * Math.sin(2 * Math.PI * u2)]
}

/** The inverse map: u₁ = exp(−r²/2), u₂ = angle / 2π. */
const inverse = ([z1, z2]: Vec2): Vec2 => {
  const u1 = Math.exp(-(z1 * z1 + z2 * z2) / 2)
  const angle = Math.atan2(z2, z1) / (2 * Math.PI)
  return [u1, angle < 0 ? angle + 1 : angle]
}

const clampU = (v: number) => Math.min(Math.max(v, 0.002), 0.998)

const circle = (r: number, name: string): SeriesSpec => {
  const t = toFlat(linspace(0, 2 * Math.PI, 97))
  return {
    name,
    type: 'line',
    x: t.map((a) => r * Math.cos(a)),
    y: t.map((a) => r * Math.sin(a)),
    muted: true,
    dashed: true,
  }
}

/** Uniform pairs in the unit square on the left, their Box–Muller images on the right, coloured by the band of u₁. */
export function BoxMuller() {
  const state = useFigureState({
    logN: float(3, {
      min: 2,
      max: 3.5,
      step: 0.1,
      label: 'pairs',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => Math.round(10 ** v).toLocaleString(),
    }),
  })
  const [u, setU] = useState<Vec2>([0.25, 0.1])

  const { uniforms, normals, moments } = useMemo(() => {
    const n = Math.round(10 ** state.logN)
    const r = stream(5)
    const us: Vec2[] = Array.from({ length: n }, () => [Math.max(uniform(r), 1e-12), uniform(r)])
    const zs = us.map(boxMuller)
    const groups = us.map(([u1]) => band(u1))
    const m = zs.reduce((s, z) => s + z[0], 0) / n
    const v = zs.reduce((s, z) => s + (z[0] - m) ** 2, 0) / (n - 1)
    const c = zs.reduce((s, z) => s + z[0] * z[1], 0) / n
    return {
      uniforms: {
        name: 'uniform pairs',
        type: 'scatter',
        x: us.map((p) => p[0]),
        y: us.map((p) => p[1]),
        group: groups,
        groupNames: BAND_NAMES,
      } satisfies SeriesSpec,
      normals: {
        name: 'normal pairs',
        type: 'scatter',
        x: zs.map((p) => p[0]),
        y: zs.map((p) => p[1]),
        group: groups,
        groupNames: BAND_NAMES,
      } satisfies SeriesSpec,
      moments: { mean: m, variance: v, product: c },
    }
  }, [state.logN])

  // The radii that the band edges u₁ = 1/3 and u₁ = 2/3 map to.
  const rings = useMemo(
    () => [
      circle(Math.sqrt(-2 * Math.log(2 / 3)), 'r at u₁ = 2/3'),
      circle(Math.sqrt(2 * Math.log(3)), 'r at u₁ = 1/3'),
    ],
    [],
  )
  const z = boxMuller(u)

  const xAxis = useAxis({ label: 'u₁', range: [0, 1] })
  const yAxis = useAxis({ label: 'u₂', range: [0, 1], equal: xAxis })
  const xAxis2 = useAxis({ label: 'z₁', range: [-4, 4] })
  const yAxis2 = useAxis({ label: 'z₂', range: [-4, 4], equal: xAxis2 })
  return (
    <Figure
      title="Two uniforms in, two Gaussians out"
      state={state}
      caption="Each uniform pair (u₁, u₂) on the left maps to one point on the right. The band of u₁ sets the ring: small u₁ gives a large radius √(−2 log u₁). u₂ sets the angle 2πu₂. The dashed circles are the images of u₁ = 1/3 and u₁ = 2/3. Drag the black point on either side to see where a single pair goes."

      readouts={
        <>
          <Readout label="(u₁, u₂)" value={`(${formatNumber(u[0])}, ${formatNumber(u[1])})`} />
          <Readout label="radius √(−2 log u₁)" value={formatNumber(Math.hypot(z[0], z[1]))} />
          <Readout label="(z₁, z₂)" value={`(${formatNumber(z[0])}, ${formatNumber(z[1])})`} />
          <Readout label="sample mean of z₁" value={formatNumber(moments.mean)} />
          <Readout label="sample variance of z₁" value={formatNumber(moments.variance)} />
          <Readout label="mean of z₁z₂" value={formatNumber(moments.product)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers([uniforms])}
          <Handle kind="point" at={u} onDrag={(p) => setU([clampU(p[0]), clampU(p[1])])} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          {seriesLayers([normals, ...rings])}
          <Handle kind="point" at={z} onDrag={(p) => setU(((v) => [clampU(v[0]), clampU(v[1])] as Vec2)(inverse(p)))} />
        </Plot>
      </div>
    </Figure>
  )
}
