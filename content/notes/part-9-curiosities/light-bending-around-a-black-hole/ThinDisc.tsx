import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Raster,
  Readout,
  setting,
  slider,
  toggle,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { B_CRIT, DISC_INNER, DISC_OUTER, discImage, rayTable } from './blackHole'

/** The image spans [−HALF, HALF] in impact-parameter units. */
const HALF = 24

const ring = (r: number) => {
  const a = Array.from({ length: 121 }, (_, i) => (2 * Math.PI * i) / 120)
  return { x: a.map((v) => r * Math.cos(v)), y: a.map((v) => r * Math.sin(v)) }
}
const SHADOW = ring(B_CRIT)

/**
 * A thin disc from r = 6 to r = 20 around a Schwarzschild black hole, seen by a distant observer. A table of u(φ) for
 * a few hundred impact parameters is integrated once; each pixel finds where its ray crosses the disc's plane by a
 * table lookup, so the inclination slider redraws at once.
 */
export function ThinDisc() {
  const state = useFigureState({
    inclination: slider(0, 89, 80, { label: 'inclination (degrees)', step: 1 }),
    secondary: setting(true, 'show the secondary image'),
    doppler: toggle(false, 'Doppler beaming and gravitational redshift'),
    root: setting(false, 'square-root brightness scale'),
    pixels: int(200, { min: 50, max: 400, label: 'pixels across', suggestions: [100, 200, 300] }),
  })
  const { inclination, secondary, doppler, root, pixels } = state

  const table = useMemo(() => rayTable(), [])
  const image = useComputed(
    () => discImage(table, inclination, pixels, HALF, { secondary, doppler }),
    [table, inclination, pixels, secondary, doppler],
  )
  const shown = useMemo(
    () => (root ? image.value.z.map((row) => row.map((v) => Math.sqrt(v))) : image.value.z),
    [image.value, root],
  )
  const top = useMemo(() => Math.max(1e-9, ...shown.map((row) => Math.max(...row))), [shown])

  const xa = useAxis({ label: 'image x = b cos ψ (units of M)', range: [-HALF, HALF] })
  const ya = useAxis({ label: 'image y = b sin ψ', range: [-HALF, HALF], equal: xa })
  const v = image.value

  return (
    <Figure
      title="A black hole with a thin disc"
      state={state}
      defaultSize="L"
      aspect={1}
      readouts={{
        'frequency shift g = 1/(1 + z) on the disc': (
          <>
            <Readout label="smallest g (receding side)" value={Number.isFinite(v.gMin) ? formatNumber(v.gMin) : '–'} />
            <Readout
              label="largest g (approaching side)"
              value={Number.isFinite(v.gMax) ? formatNumber(v.gMax) : '–'}
            />
          </>
        ),
        pixels: (
          <>
            <Readout label="on the primary image" value={String(v.primary)} />
            <Readout label="on the secondary image" value={String(v.secondary)} />
          </>
        ),
        cost: (
          <>
            <Readout label="table of u(φ), built once" value={`${table.ms.toFixed(0)} ms`} />
            <Readout label="this image" value={`${image.ms.toFixed(0)} ms`} />
          </>
        ),
      }}
      caption={
        <>
          Units G = c = M = 1. A geometrically thin, opaque disc in the equatorial plane from r = {DISC_INNER} (the
          innermost stable circular orbit) to r = {DISC_OUTER}, rotating counter-clockwise seen from above, viewed by a
          distant observer at inclination i, the angle between the line of sight and the disc&apos;s axis (0° is
          face-on, 90° edge-on). Each pixel at impact parameter b and position angle ψ is a ray that crosses the
          disc&apos;s plane at azimuths φ₀, φ₀ + π, …; the first crossing that lands on the disc gives the pixel its
          light, the Page–Thorne flux at that radius: zero at r = 6, a peak at r = 9.55, and 35% of the peak at r = 20.
          The primary and secondary images of one radius have the same brightness until the frequency shift is on. Move
          the inclination towards 90°: the far side of the disc rises over the top of the shadow and its underside
          appears below it (the secondary image, light that passed under the hole). Switch on Doppler beaming and
          gravitational redshift to multiply each pixel by g⁴, where g = 1/(1 + z) is Luminet&apos;s redshift factor for
          gas on circular Keplerian orbits: the side moving towards the observer (left) brightens and the receding side
          dims. The dashed circle is the shadow&apos;s edge, b = 3√3. The colour scale runs linearly from the page
          colour (no light) to the strongest blue at the image&apos;s brightest pixel; the square-root scale lifts the
          dim receding side when the frequency shift is on.
        </>
      }
    >
      <Plot x={xa} y={ya}>
        <Raster
          x={v.x}
          y={v.y}
          z={shown}
          scale="sequential"
          range={[0, top]}
          colorBar={false}
          valueLabel={root ? '√(I / I_peak)' : 'I / I_peak'}
          stale={image.stale}
        />
        <Curve name="shadow edge, b = 3√3" x={SHADOW.x} y={SHADOW.y} muted dashed silent />
      </Plot>
    </Figure>
  )
}
