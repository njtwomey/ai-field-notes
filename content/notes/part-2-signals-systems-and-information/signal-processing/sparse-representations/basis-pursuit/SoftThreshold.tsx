import { useMemo } from 'react'
import { proxL1 } from 'aifn-compute/optim/proximal'
import { linspace, tensor, toFlat } from 'aifn-compute/foundation/tensor'
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
  useFigureState,
} from 'aifn-render'

const R = 3
const XS = toFlat(linspace(-R, R, 601))

/** Soft thresholding of each value at level λ, by the proximal operator of λ‖·‖₁ with unit step. */
const soft = (values: readonly number[], lambda: number) => toFlat(proxL1(lambda).prox(tensor(values), 1))

/**
 * The one-coordinate problem min_x ½(x − v)² + λ|x|: a parabola centred on v plus a V. Its minimiser is the soft
 * threshold of v, which is zero while |v| ≤ λ.
 */
export function SoftThreshold() {
  const state = useFigureState({
    v: slider(-R, R, 1.8, { step: 0.02, label: 'input v', onChart: true }),
    lambda: slider(0, 2, 0.8, { step: 0.02, label: 'λ' }),
  })
  const { v, lambda } = state

  const d = useMemo(() => {
    const quad = XS.map((x) => 0.5 * (x - v) ** 2)
    const pen = XS.map((x) => lambda * Math.abs(x))
    const total = XS.map((_, i) => quad[i] + pen[i])
    const xStar = soft([v], lambda)[0]
    const softMap = soft(XS, lambda)
    // NaN breaks the line at the two jumps.
    const hard = { x: [-R, -lambda, NaN, -lambda, lambda, NaN, lambda, R], y: [-R, -lambda, NaN, 0, 0, NaN, lambda, R] }
    return { quad, pen, total, xStar, fStar: 0.5 * (xStar - v) ** 2 + lambda * Math.abs(xStar), softMap, hard }
  }, [v, lambda])

  const ax = useAxis({ label: 'x', range: [-R, R] })
  const ay = useAxis({ label: 'objective', range: [0, 6] })
  const bx = useAxis({ label: 'input v', range: [-R, R] })
  const by = useAxis({ label: 'output', range: [-R, R] })
  return (
    <Figure
      title="Soft thresholding is the proximal step of the ℓ1 penalty"
      state={state}
      caption="Left: the one-coordinate problem ½(x − v)² + λ|x| (solid) is a parabola centred on the input v (drag the guide) plus a V of slope λ (both dashed). Its minimiser (the dot) is v moved towards zero by λ, and it is exactly zero while |v| ≤ λ, because the V's kink at zero is steeper than the parabola's slope there. Right: the map from v to the minimiser, soft thresholding, beside hard thresholding at the same level, which keeps a large input unchanged and zeroes a small one."
      readouts={
        <>
          <Readout label="minimiser S_λ(v)" value={formatNumber(d.xStar)} />
          <Readout label="shrinkage v − S_λ(v)" value={formatNumber(v - d.xStar)} />
          <Readout label="zero?" value={Math.abs(v) <= lambda ? 'yes: |v| ≤ λ' : 'no: |v| > λ'} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={ax} y={ay} height={300}>
          <Curve name="½(x − v)²" x={XS} y={d.quad} muted dashed />
          <Curve name="λ|x|" x={XS} y={d.pen} muted dashed />
          <Curve name="½(x − v)² + λ|x|" x={XS} y={d.total} slot={0} />
          <Points name="minimiser" x={[d.xStar]} y={[d.fStar]} emphasis size={10} live />
          <Handle {...state.handle('v', { label: 'v' })} />
        </Plot>
        <Plot x={bx} y={by} height={300}>
          <Curve name="identity" x={[-R, R]} y={[-R, R]} muted dashed />
          <Curve name="soft thresholding S_λ(v)" x={XS} y={d.softMap} slot={0} />
          <Curve name="hard thresholding at λ" x={d.hard.x} y={d.hard.y} slot={1} dashed />
          <Points name="this input" x={[v]} y={[d.xStar]} emphasis size={10} live />
        </Plot>
      </div>
    </Figure>
  )
}
