import { useMemo } from 'react'
import { samplePrior } from 'aifn-applied/learning/gaussian-processes'
import {
  gram,
  linearKernel,
  matern,
  periodic,
  polynomial,
  rationalQuadratic,
  rbf,
  type Kernel,
} from 'aifn/learning/kernels'
import { stream } from 'aifn/foundation/random'
import { linspace, tensor, toFlat, toRows } from 'aifn/foundation/tensor'
import { Figure } from '@lab/layout'
import { choice, row, slider, useFigureState, variants } from '@lab/state'
import { Curve, Handle, Plot, Plots, Raster, Readout, useAxis } from '@lab/viz'
import { formatValue } from '@lab/views'

const ell = (initial: number) => slider(0.05, 3, initial, { label: 'lengthscale ℓ', step: 0.05 })
const KERNELS = variants(
  {
    rbf: { label: 'RBF', params: { ell: ell(0.7) } },
    matern12: { label: 'Matérn ½', params: { ell: ell(0.7) } },
    matern32: { label: 'Matérn 3⁄2', params: { ell: ell(0.7) } },
    matern52: { label: 'Matérn 5⁄2', params: { ell: ell(0.7) } },
    rq: {
      label: 'rational quadratic',
      params: { ell: ell(0.7), alpha: slider(0.2, 4, 1.5, { label: 'shape α' }) },
    },
    periodic: {
      label: 'periodic',
      params: { ell: ell(0.7), period: slider(0.2, 4, 1.5, { label: 'period p' }) },
    },
    linear: { label: 'linear', params: {} },
    cubic: { label: 'cubic', params: {} },
  },
  {
    label: '1 · kernel',
    choiceLabel: 'kernel',
    initial: 'matern32',
    shared: { variance: slider(0.1, 3, 1, { label: 'variance σ²' }) },
  },
)

type Chosen = { key: string; values: { variance: number; ell?: number; alpha?: number; period?: number } }

function make(k: Chosen): Kernel {
  const { variance } = k.values
  const lengthscale = k.values.ell ?? 1
  switch (k.key) {
    case 'rbf':
      return rbf({ lengthscale, variance })
    case 'matern12':
      return matern(0.5, { lengthscale, variance })
    case 'matern32':
      return matern(1.5, { lengthscale, variance })
    case 'matern52':
      return matern(2.5, { lengthscale, variance })
    case 'rq':
      return rationalQuadratic({ lengthscale, variance, alpha: k.values.alpha! })
    case 'periodic':
      return periodic({ lengthscale, variance, period: k.values.period! })
    case 'linear':
      return linearKernel({ variance, bias: 0.1 })
    default:
      return polynomial(3, { variance, bias: 1 })
  }
}

const GRID = linspace(-3, 3, 241)
const GRID_X = toFlat(GRID)
const GRAM_GRID = linspace(-3, 3, 41)
const GRAM_AXIS = toFlat(GRAM_GRID)

/** One kernel at a time: its covariance with a reference point, the functions it draws, and its Gram matrix. */
export function KernelGallery() {
  const state = useFigureState({
    kernel: KERNELS,
    prior: row('2 · prior draws', { draws: choice([4, 12, 40], 4, { label: 'draws' }) }),
    anchor: slider(-3, 3, 0, { onChart: true, label: 'x₀' }),
  })
  const chosen = state.kernel as unknown as Chosen
  const kernel = useMemo(() => make(chosen), [state.kernel]) // eslint-disable-line react-hooks/exhaustive-deps -- chosen is state.kernel
  const anchor = state.anchor
  const count = state.prior.draws
  const stationary = kernel.stationary
  const covRow = useMemo(() => toFlat(gram(kernel, GRID, tensor([anchor]))), [kernel, anchor])
  const draws = useMemo(() => samplePrior(stream('kernel-gallery'), kernel, GRID, count), [kernel, count])
  const drawRows = useMemo(() => toRows(draws.draws), [draws])
  const K = useMemo(() => toRows(gram(kernel, GRAM_GRID)), [kernel])
  const x = useAxis({ label: 'x', range: [-3, 3] })
  const kAxis = useAxis({ label: 'k(x, x₀)', hold: 'union', key: state.kernel.key })
  const fAxis = useAxis({ label: 'f(x)', hold: 'union', key: state.kernel.key })
  const gx = useAxis({ label: 'xⱼ' })
  const gy = useAxis({ label: 'xᵢ', equal: gx })
  const many = count > 4
  return (
    <>
      <Figure
        title="Covariance with a point and the functions it implies"
        defaultSize="L"
        purpose="A kernel k(x, x′) is the covariance of a Gaussian process: its shape against one input sets how rough and how far-reaching the prior's draws are."
        state={state}
        readouts={{
          kernel: (
            <>
              <Readout label="stationary" value={String(stationary)} />
              <Readout label="k(x₀, x₀)" value={formatValue(toFlat(gram(kernel, tensor([anchor])))[0])} />
              <Readout label="jitter for the draws" value={formatValue(draws.jitter)} />
            </>
          ),
        }}
        caption="Drag the vertical line to move the reference point x₀. A stationary kernel's curve only slides with x₀; the linear and cubic kernels change shape. Matérn ½ draws are rough, the RBF's are infinitely smooth, and the periodic kernel repeats. The draws use fixed normals, so they deform continuously as ℓ changes."
      >
        <Plots rows={2} heights={[1, 1.4]} hoverGroup>
          <Plot x={x} y={kAxis}>
            <Curve name={`k(x, ${formatValue(anchor)})`} x={GRID_X} y={covRow} slot={0} />
            <Handle {...state.handle('anchor', { label: 'x₀' })} />
          </Plot>
          <Plot x={x} y={fAxis} legend={false}>
            {drawRows.map((y, i) =>
              many ? (
                <Curve key={i} name="draws" x={GRID_X} y={y} slot={0} thin silent />
              ) : (
                <Curve key={i} name={`draw ${i + 1}`} x={GRID_X} y={y} slot={i} />
              ),
            )}
          </Plot>
        </Plots>
      </Figure>
      <Figure
        title="The Gram matrix"
        purpose="k(xᵢ, xⱼ) on 41 evenly spaced inputs: a band for a short lengthscale, near rank one for a long one."
        caption="The same kernel as above. Stationary kernels give Toeplitz matrices (constant along diagonals)."
      >
        <Plot x={gx} y={gy}>
          <Raster x={GRAM_AXIS} y={GRAM_AXIS} z={K} valueLabel="k(xᵢ, xⱼ)" />
        </Plot>
      </Figure>
    </>
  )
}
