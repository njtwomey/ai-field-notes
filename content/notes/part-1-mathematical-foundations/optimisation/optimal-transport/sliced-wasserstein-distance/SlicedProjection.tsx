import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  type Segment,
  Segments,
  slider,
  useAxis,
  useFigureState,
  Vectors,
} from 'aifn-render'
import { costMatrix, hungarian, wass1dSamples, type Pt } from '../_shared/ot'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream } from 'aifn-compute/foundation/random'

const N = 30
const RANGE: [number, number] = [-3, 3]
const TIP = 2.6
const ANGLES = toFlat(linspace(0, 179, 180))

/** Two elongated clouds with the same centre: one stretched along x₁, the other along x₂. */
const CLOUDS = (() => {
  const r = stream(3)
  const src: Pt[] = Array.from({ length: N }, () => [1.2 * normal(r), 0.35 * normal(r)])
  const tgt: Pt[] = Array.from({ length: N }, () => [0.35 * normal(r), 1.2 * normal(r)])
  return { src, tgt }
})()

const project = (pts: Pt[], deg: number) => {
  const c = Math.cos((deg * Math.PI) / 180)
  const s = Math.sin((deg * Math.PI) / 180)
  return pts.map(([x, y]) => x * c + y * s)
}
const sliceCost = (deg: number) => wass1dSamples(project(CLOUDS.src, deg), project(CLOUDS.tgt, deg), 2)

const PROFILE = ANGLES.map(sliceCost)
const SW2 = PROFILE.reduce((s, v) => s + v, 0) / PROFILE.length
const W2 = (() => {
  const C = costMatrix(CLOUDS.src, CLOUDS.tgt, 2)
  return hungarian(C).reduce((s, j, i) => s + C[i][j], 0) / N
})()

/**
 * Projecting two 2-D samples onto a direction reduces transport to sorting. The sliced distance averages the 1-D
 * costs over directions; here the clouds differ in orientation, so some directions see no difference at all.
 */
export function SlicedProjection() {
  const state = useFigureState({
    angle: slider(0, 179, 20, { step: 1, label: 'direction θ (degrees)' }),
  })
  const dir = useMemo((): Pt => {
    const rad = (state.angle * Math.PI) / 180
    return [Math.cos(rad), Math.sin(rad)]
  }, [state.angle])

  const series = useMemo(
    () =>
      [
        { name: 'source', x: CLOUDS.src.map((p) => p[0]), y: CLOUDS.src.map((p) => p[1]), slot: 0 },
        { name: 'target', x: CLOUDS.tgt.map((p) => p[0]), y: CLOUDS.tgt.map((p) => p[1]), slot: 1 },
        {
          name: 'direction θ',
          x: [-TIP * dir[0], TIP * dir[0]],
          y: [-TIP * dir[1], TIP * dir[1]],
          muted: true,
        },
      ] as const,
    [dir],
  )
  // Each point joined to its foot on the projection line.
  const segments = useMemo((): Segment[] => {
    const feet = (pts: Pt[]) =>
      pts.map((p): Segment => {
        const t = p[0] * dir[0] + p[1] * dir[1]
        return { from: p, to: [t * dir[0], t * dir[1]] }
      })
    return [...feet(CLOUDS.src), ...feet(CLOUDS.tgt)]
  }, [dir])
  const vectors = useMemo((): Segment[] => [{ from: [0, 0], to: [TIP * dir[0], TIP * dir[1]] }], [dir])

  const profile = useMemo(
    () =>
      [
        { name: 'W₂² of the projections', x: ANGLES, y: PROFILE, slot: 2 },
        { name: 'average = SW₂²', x: [0, 179], y: [SW2, SW2], dashed: true, muted: true },
      ] as const,
    [],
  )

  const xAxis = useAxis({ label: 'x₁', range: RANGE })
  const yAxis = useAxis({ label: 'x₂', range: RANGE, equal: xAxis })
  const xAxis2 = useAxis({ label: 'direction θ (degrees)', range: [0, 179] })
  const yAxis2 = useAxis({ label: '1-D transport cost', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Slicing two point clouds"
      state={state}
      caption="Drag the arrow tip to choose a direction θ, or drag the vertical line on the right. Every point is projected onto the line; in one dimension the optimal plan pairs the sorted projections, so each slice costs one sort. The clouds share a centre but are stretched along different axes. Near 45° and 135° their projections have similar spreads and the slice cost is smallest. The sliced distance averages the slices and is smaller than the exact W₂²."

      readouts={
        <>
          <Readout label="W₂² along θ" value={formatNumber(PROFILE[Math.round(state.angle)])} />
          <Readout label="SW₂² (180 directions)" value={formatNumber(SW2)} />
          <Readout label="exact W₂² (assignment)" value={formatNumber(W2)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          <Points {...series[0]} />
          <Points {...series[1]} />
          <Curve {...series[2]} />
          <Segments segments={segments} />
          <Vectors vectors={vectors} />
          <Handle
            kind="point"
            at={[TIP * dir[0], TIP * dir[1]]}
            label="θ"
            onDrag={([x, y]) => {
              // A direction and its opposite give the same slice, so fold the angle into [0°, 180°).
              let deg = (Math.atan2(y, x) * 180) / Math.PI
              if (deg < 0) deg += 180
              state.set('angle', deg >= 179.5 ? 0 : deg)
            }}
          />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          <Curve {...profile[0]} />
          <Curve {...profile[1]} />
          <Handle {...state.handle('angle', { label: 'θ' })} />
        </Plot>
      </div>
    </Figure>
  )
}
