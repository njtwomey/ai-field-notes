import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type Handle,
  type Vec2,
  type XYSeries,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'

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

const circle = (r: number, name: string): XYSeries => {
  const t = linspace(0, 2 * Math.PI, 97)
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
  const [logN, setLogN] = useState(3)
  const [u, setU] = useState<Vec2>([0.25, 0.1])

  const { uniforms, normals, moments } = useMemo(() => {
    const n = Math.round(10 ** logN)
    const r = rng(5)
    const us: Vec2[] = Array.from({ length: n }, () => [Math.max(r.uniform(), 1e-12), r.uniform()])
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
      } satisfies XYSeries,
      normals: {
        name: 'normal pairs',
        type: 'scatter',
        x: zs.map((p) => p[0]),
        y: zs.map((p) => p[1]),
        group: groups,
        groupNames: BAND_NAMES,
      } satisfies XYSeries,
      moments: { mean: m, variance: v, product: c },
    }
  }, [logN])

  // The radii that the band edges u₁ = 1/3 and u₁ = 2/3 map to.
  const rings = useMemo(
    () => [
      circle(Math.sqrt(-2 * Math.log(2 / 3)), 'r at u₁ = 2/3'),
      circle(Math.sqrt(2 * Math.log(3)), 'r at u₁ = 1/3'),
    ],
    [],
  )
  const z = boxMuller(u)
  const leftHandles: Handle[] = [{ kind: 'point', at: u, onDrag: (p) => setU([clampU(p[0]), clampU(p[1])]) }]
  const rightHandles: Handle[] = [
    { kind: 'point', at: z, onDrag: (p) => setU(((v) => [clampU(v[0]), clampU(v[1])] as Vec2)(inverse(p))) },
  ]

  return (
    <Interactive
      title="Two uniforms in, two Gaussians out"
      caption="Each uniform pair (u₁, u₂) on the left maps to one point on the right. The band of u₁ sets the ring: small u₁ gives a large radius √(−2 log u₁). u₂ sets the angle 2πu₂. The dashed circles are the images of u₁ = 1/3 and u₁ = 2/3. Drag the black point on either side to see where a single pair goes."
      controls={
        <ParamSlider
          label="pairs"
          value={logN}
          onChange={setLogN}
          min={2}
          max={3.5}
          step={0.1}
          format={(v) => Math.round(10 ** v).toLocaleString()}
        />
      }
      readout={
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
        <XYChart
          series={[uniforms]}
          xRange={[0, 1]}
          yRange={[0, 1]}
          equalAspect
          xLabel="u₁"
          yLabel="u₂"
          handles={leftHandles}
        />
        <XYChart
          series={[normals, ...rings]}
          xRange={[-4, 4]}
          yRange={[-4, 4]}
          equalAspect
          xLabel="z₁"
          yLabel="z₂"
          handles={rightHandles}
        />
      </div>
    </Interactive>
  )
}
