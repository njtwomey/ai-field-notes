import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useElementSize,
  useFigureState,
} from 'aifn-render'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'
import { FixedHeight } from '../_shared/FixedHeight'
import { circle, TAU } from '../_shared/geometry'

const N = 150
const CLASSES = ['inner ring', 'outer ring']

/**
 * Two concentric rings are not separable by a straight line in (x, y), but in polar coordinates (θ, r) the radius alone
 * separates them, and the circle r = r* becomes a horizontal line.
 */
export function PolarFeatures() {
  const state = useFigureState({
    noise: slider(0.05, 0.5, 0.2, { step: 0.01, label: 'ring width (sd of r)' }),
    cut: slider(0.2, 3, 1.4, { step: 0.01, onChart: true }),
  })
  const { noise, cut } = state

  const data = useMemo(() => {
    const s = stream('polar-rings')
    const x: number[] = []
    const y: number[] = []
    const r: number[] = []
    const t: number[] = []
    const group: number[] = []
    for (let k = 0; k < 2; k++) {
      for (let i = 0; i < N; i++) {
        const radius = Math.max(0.02, (k === 0 ? 1 : 2) + noise * normal(s))
        const angle = uniform(s, 0, TAU)
        x.push(radius * Math.cos(angle))
        y.push(radius * Math.sin(angle))
        r.push(radius)
        t.push(angle)
        group.push(k)
      }
    }
    return { x, y, r, t, group }
  }, [noise])

  const accuracy = useMemo(() => {
    let right = 0
    for (let i = 0; i < data.r.length; i++) if ((data.r[i] > cut ? 1 : 0) === data.group[i]) right++
    return right / data.r.length
  }, [data, cut])
  const ring = useMemo(() => circle(cut), [cut])

  const [cell, size] = useElementSize<HTMLDivElement>()
  const height = Math.max(240, Math.round(size.width))

  const cx = useAxis({ label: 'x', range: [-3.2, 3.2] })
  const cy = useAxis({ label: 'y', range: [-3.2, 3.2], equal: cx })
  const tx = useAxis({ label: 'θ (rad)', range: [0, TAU] })
  const ry = useAxis({ label: 'r', range: [0, 3.2] })

  return (
    <Figure
      title="Polar features make rings separable"
      state={state}
      caption="Two classes of points lie on concentric rings of radius 1 and 2. Left: in Cartesian coordinates no straight line separates them. Right: the same points in polar coordinates, angle θ across and radius r up. The classes now occupy two horizontal bands, and the horizontal line r = r* separates them; drag it. The dashed circle on the left is the same boundary. A linear classifier on the features (r, θ), or on r alone, solves a problem that a linear classifier on (x, y) cannot. Widen the rings to make them overlap."
      readouts={
        <>
          <Readout label="threshold r*" value={formatNumber(cut)} />
          <Readout label="accuracy of r > r* ⇒ outer" value={`${formatNumber(100 * accuracy)}%`} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div ref={cell}>
          <Plot x={cx} y={cy} fitHeight title="Cartesian (x, y)">
            <Points name="points" x={data.x} y={data.y} group={data.group} groupNames={CLASSES} size={5} />
            <Curve name="r = r*" x={ring.x} y={ring.y} emphasis dashed silent />
          </Plot>
        </div>
        <FixedHeight height={height}>
          <Plot x={tx} y={ry} title="polar (θ, r)" legend={false}>
            <Points name="points" x={data.t} y={data.r} group={data.group} groupNames={CLASSES} size={5} />
            <Handle {...state.handle('cut', { axis: 'y', label: 'r*' })} />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
