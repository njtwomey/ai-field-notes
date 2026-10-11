import { useContext, useMemo, type ReactNode } from 'react'
import {
  Figure,
  formatNumber,
  FrameContext,
  Plot,
  Points,
  Readout,
  Vectors,
  slider,
  useAxis,
  useFigureState,
  type AxisModel,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { mean, square, sub, sum, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { MultivariateNormal } from 'aifn-compute/probability/distributions'
import { whitening } from 'aifn-methods/learning/preprocessing'

const N = 300
const PANEL = 260
/** Rows drawn in colour in every panel, so that each can be followed through the transforms. */
const MARKED = [0, 1, 2]

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** The mean squared distance between each centred input row and its whitened row. */
function distanceToInput(xc: Tensor, z: Tensor) {
  return mean(sum(square(sub(xc, z)), 1)) as number
}

type PanelProps = { title: string; rows: number[][]; x: AxisModel; y: AxisModel; axes?: number[][] }

function Panel({ title, rows, x, y, axes }: PanelProps) {
  const xs = rows.map((r) => r[0])
  const ys = rows.map((r) => r[1])
  return (
    <FixedHeight height={PANEL}>
      <Plot x={x} y={y} title={title}>
        <Points name="rows" x={xs} y={ys} muted />
        {axes && (
          <Vectors
            name="principal axes"
            vectors={axes.map((a, k) => ({ from: [0, 0], to: [a[0], a[1]], label: `axis ${k + 1}` }))}
          />
        )}
        {MARKED.map((i, k) => (
          <Points key={i} name={`row ${'ABC'[k]}`} x={[xs[i]]} y={[ys[i]]} slot={k} />
        ))}
      </Plot>
    </FixedHeight>
  )
}

/** Correlated rows, and the same rows after PCA whitening and after ZCA whitening. */
export function WhiteningPanels() {
  const state = useFigureState({
    sd1: slider(0.3, 3, 2, { step: 0.1, label: 'standard deviation of x₁' }),
    sd2: slider(0.3, 3, 0.8, { step: 0.1, label: 'standard deviation of x₂' }),
    rho: slider(-0.95, 0.95, 0.7, { step: 0.05, label: 'correlation ρ' }),
    epsilon: slider(0, 1, 0, { step: 0.05, label: 'regulariser ε' }),
  })
  const { sd1, sd2, rho, epsilon } = state
  const view = useMemo(() => {
    const c = rho * sd1 * sd2
    const covariance = tensor([
      [sd1 * sd1, c],
      [c, sd2 * sd2],
    ])
    const x = MultivariateNormal(tensor([0, 0]), { covariance }).sample(stream('whitening/rows'), {
      shape: [N],
    }) as Tensor
    const pca = whitening({ method: 'pca', epsilon }).fit({ x })
    const zca = whitening({ method: 'zca', epsilon }).fit({ x })
    const xc = sub(x, pca.mean)
    const zp = pca.transform(x)
    const zz = zca.transform(x)
    const variances = Array.from(toFlat(pca.explainedVariance))
    // The principal axes (the columns of the components matrix), each drawn two standard deviations long.
    const v = toRows(pca.components) as number[][]
    const axes = variances.map((lambda, k) => [2 * Math.sqrt(lambda) * v[0][k], 2 * Math.sqrt(lambda) * v[1][k]])
    return {
      raw: toRows(xc) as number[][],
      pca: toRows(zp) as number[][],
      zca: toRows(zz) as number[][],
      axes,
      condition: variances[0] / variances[variances.length - 1],
      movedPca: distanceToInput(xc, zp),
      movedZca: distanceToInput(xc, zz),
    }
  }, [sd1, sd2, rho, epsilon])
  const rx = useAxis({ label: 'x₁', range: [-7, 7] })
  const ry = useAxis({ label: 'x₂', range: [-5, 5], equal: rx })
  const px = useAxis({ label: 'z₁', range: [-4, 4] })
  const py = useAxis({ label: 'z₂', range: [-4, 4], equal: px })
  const zx = useAxis({ label: 'z₁', range: [-4, 4] })
  const zy = useAxis({ label: 'z₂', range: [-4, 4], equal: zx })
  return (
    <Figure
      title="PCA and ZCA whitening of the same rows"
      state={state}
      hoverReadout={false}
      caption="Left: 300 rows from a two-dimensional normal distribution with the chosen standard deviations and correlation, centred, with the principal axes drawn two standard deviations long. Middle: PCA whitening rotates the rows onto the principal axes and divides each axis by its standard deviation. Right: ZCA whitening does the same and then rotates back to the original axes. Both clouds have identity covariance; the three coloured rows show that PCA whitening turns and may flip the picture while ZCA whitening keeps each row near where it started. A regulariser ε above 0 shrinks the low-variance axis less than full whitening would, so the whitened cloud stays elongated."
      readouts={
        <>
          <Readout label="condition number of the covariance" value={formatNumber(view.condition)} />
          <Readout label="mean squared move, PCA" value={formatNumber(view.movedPca)} />
          <Readout label="mean squared move, ZCA" value={formatNumber(view.movedZca)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Panel title="centred rows" rows={view.raw} x={rx} y={ry} axes={view.axes} />
        <Panel title="PCA-whitened" rows={view.pca} x={px} y={py} />
        <Panel title="ZCA-whitened" rows={view.zca} x={zx} y={zy} />
      </div>
    </Figure>
  )
}
