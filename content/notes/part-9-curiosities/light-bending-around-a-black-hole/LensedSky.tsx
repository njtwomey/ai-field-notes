import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Raster,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { B_CRIT, checkerLight, imageRadius, lensedSky, R_OBS, starField, starLight, sweepTable } from './blackHole'

/** The image spans [−HALF, HALF] in both pixel coordinates (r_obs tan of the angle from the view centre). */
const HALF = 25
/** The hole may be dragged within [−LIMIT, LIMIT]. */
const LIMIT = 18
const STARS = 2000

const ring = (cx: number, cy: number, r: number) => {
  const a = Array.from({ length: 121 }, (_, i) => (2 * Math.PI * i) / 120)
  return { x: a.map((v) => cx + r * Math.cos(v)), y: a.map((v) => cy + r * Math.sin(v)) }
}
const clamp = (v: number) => Math.min(LIMIT, Math.max(-LIMIT, v))

/**
 * A ray-traced view of the sky behind a black hole. One table of the azimuth Φ(b) that a ray sweeps from the observer
 * to infinity is built once; each pixel then needs one spline lookup, so dragging the hole redraws at once.
 */
export function LensedSky() {
  const state = useFigureState({
    hx: float(9, { min: -LIMIT, max: LIMIT, label: 'hole, x', onChart: true }),
    hy: float(5, { min: -LIMIT, max: LIMIT, label: 'hole, y', onChart: true }),
    sky: choice(['stars', 'checkerboard'], 'stars', { label: 'background' }),
    pixels: int(200, { min: 50, max: 400, label: 'pixels across', suggestions: [100, 200, 300] }),
    seed: int(1, { min: 0, max: 999, label: 'star field seed' }),
  })
  const { hx, hy, sky, pixels, seed } = state

  const table = useMemo(() => sweepTable(), [])
  const field = useMemo(() => starField(seed, STARS), [seed])
  const image = useComputed(
    () =>
      lensedSky(table, sky === 'stars' ? (x, y, z) => starLight(field, x, y, z) : checkerLight, hx, hy, pixels, HALF),
    [table, field, sky, hx, hy, pixels],
  )
  const ringRadius = imageRadius(table.einsteinB)
  const shadowRadius = imageRadius(B_CRIT)
  const einstein = useMemo(() => ring(hx, hy, ringRadius), [hx, hy, ringRadius])
  const shadow = useMemo(() => ring(hx, hy, shadowRadius), [hx, hy, shadowRadius])

  const xa = useAxis({ label: 'image x (units of M at the hole)', range: [-HALF, HALF] })
  const ya = useAxis({ label: 'image y', range: [-HALF, HALF], equal: xa })

  return (
    <Figure
      title="The lensed sky"
      state={state}
      defaultSize="L"
      aspect={1}
      readouts={{
        geometry: (
          <>
            <Readout label="observer distance r_obs" value={formatNumber(R_OBS)} />
            <Readout label="hole to bright star, in the image" value={formatNumber(Math.hypot(hx, hy))} />
            <Readout
              label="Einstein ring radius (b_E)"
              value={`${formatNumber(ringRadius)} (${formatNumber(table.einsteinB)})`}
            />
            <Readout label="shadow radius (b_c)" value={`${formatNumber(shadowRadius)} (${formatNumber(B_CRIT)})`} />
          </>
        ),
        cost: (
          <>
            <Readout label="table of Φ(b), built once" value={`${table.ms.toFixed(0)} ms`} />
            <Readout label="this image" value={`${image.ms.toFixed(0)} ms`} />
          </>
        ),
      }}
      caption={
        <>
          The view of an observer at r_obs = {R_OBS} (units G = c = M = 1) looking at a black hole against a distant
          sky: a seeded star field with one bright star at the centre of the view, or a checkerboard of 4° squares.
          Image coordinates are r_obs tan(angle from the view centre), so near the hole they read as impact parameters.
          Drag the hole. Each pixel&apos;s ray is traced backwards: it stays in the plane through the hole and the pixel
          direction and sweeps the azimuth Φ(b) from a table built once with Dormand–Prince, then picks up the sky in
          the direction it leaves. The colour scale runs from the page colour (no light) to the strongest blue. The
          shadow inside the solid circle gets no light: its rays have b &lt; b_c and end on the horizon. Drag the hole
          onto the bright star (the image centre) and its light spreads over the dashed circle, the Einstein ring. Every
          star also has a second image inside the ring, on the opposite side of the hole, and a thin ring of the whole
          sky hugs the shadow.
        </>
      }
    >
      <Plot x={xa} y={ya}>
        <Raster
          x={image.value.x}
          y={image.value.y}
          z={image.value.z}
          scale="sequential"
          range={[0, 1]}
          colorBar={false}
          valueLabel="light"
          stale={image.stale}
        />
        <Curve name="Einstein ring of a star behind the hole" x={einstein.x} y={einstein.y} muted dashed silent />
        <Curve name="shadow edge" x={shadow.x} y={shadow.y} muted silent />
        <Handle
          kind="point"
          at={[hx, hy]}
          label="black hole"
          onDrag={([x, y]) => {
            state.set('hx', clamp(x))
            state.set('hy', clamp(y))
          }}
        />
      </Plot>
    </Figure>
  )
}
