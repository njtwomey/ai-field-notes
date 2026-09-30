import { useMemo, useState } from 'react'
import { samplePrior } from 'aifn/gp'
import { gram, linear, matern, periodic, polynomial, rationalQuadratic, rbf, type Kernel } from 'aifn/kernels'
import { stream } from 'aifn/random'
import { linspace, tensor, toFlat, toRows } from 'aifn/tensor'
import { Select, Slider } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Heatmap, Panel, Readout, Subplots, XYChart, type XYSeries } from '@lab/viz'
import { formatValue } from '@lab/views'

const KINDS = ['rbf', 'matern12', 'matern32', 'matern52', 'rational quadratic', 'periodic', 'linear', 'cubic'] as const
type KindName = (typeof KINDS)[number]

function make(kind: KindName, lengthscale: number, variance: number, extra: number): Kernel {
  switch (kind) {
    case 'rbf':
      return rbf({ lengthscale, variance })
    case 'matern12':
      return matern(0.5, { lengthscale, variance })
    case 'matern32':
      return matern(1.5, { lengthscale, variance })
    case 'matern52':
      return matern(2.5, { lengthscale, variance })
    case 'rational quadratic':
      return rationalQuadratic({ lengthscale, variance, alpha: extra })
    case 'periodic':
      return periodic({ lengthscale, variance, period: extra })
    case 'linear':
      return linear({ variance, bias: 0.1 })
    case 'cubic':
      return polynomial(3, { variance, bias: 1 })
  }
}

const GRID = linspace(-3, 3, 241)
const GRID_X = toFlat(GRID)
const DRAWS = 4

/** One kernel at a time: its covariance with a reference point, the functions it draws, and its Gram matrix. */
export function KernelGallery() {
  const [kind, setKind] = useState<KindName>('matern32')
  const [lengthscale, setLengthscale] = useState(0.7)
  const [variance, setVariance] = useState(1)
  const [extra, setExtra] = useState(1.5)
  const [anchor, setAnchor] = useState(0)
  const kernel = useMemo(() => make(kind, lengthscale, variance, extra), [kind, lengthscale, variance, extra])
  const stationary = kernel.stationary
  const row = useMemo(() => toFlat(gram(kernel, GRID, tensor([anchor]))), [kernel, anchor])
  const draws = useMemo(() => samplePrior(stream('kernel-gallery'), kernel, GRID, DRAWS), [kernel])
  const K = useMemo(() => {
    const g = linspace(-3, 3, 41)
    return { axis: toFlat(g), z: toRows(gram(kernel, g)) }
  }, [kernel])
  const covSeries: XYSeries[] = [{ name: `k(x, ${formatValue(anchor)})`, type: 'line', x: GRID_X, y: row, slot: 0 }]
  const drawSeries: XYSeries[] = toRows(draws.draws).map((y, i) => ({
    name: `draw ${i + 1}`,
    type: 'line' as const,
    x: GRID_X,
    y,
    slot: i,
  }))
  const extraLabel = kind === 'periodic' ? 'period p' : 'shape α'
  return (
    <>
      <Figure
        title="Covariance with a point and the functions it implies"
        defaultSize="L"
        description="A kernel k(x, x′) is the covariance of a Gaussian process: its shape against one input sets how rough and how far-reaching the prior's draws are."
        controls={
          <>
            <ControlRow label="Kernel">
              <Select label="kernel" value={kind} onChange={setKind} options={[...KINDS]} />
              <Slider label="variance σ²" value={variance} onChange={setVariance} min={0.1} max={3} />
            </ControlRow>
            <ControlRow label="Scale">
              <Slider
                label="lengthscale ℓ"
                value={lengthscale}
                onChange={setLengthscale}
                min={0.05}
                max={3}
                disabled={!stationary}
              />
              {(kind === 'periodic' || kind === 'rational quadratic') && (
                <Slider label={extraLabel} value={extra} onChange={setExtra} min={0.2} max={4} />
              )}
            </ControlRow>
          </>
        }
        readouts={
          <>
            <Readout label="stationary" value={String(stationary)} />
            <Readout label="k(x₀, x₀)" value={formatValue(toFlat(gram(kernel, tensor([anchor])))[0])} />
            <Readout label="jitter for the draws" value={formatValue(draws.jitter)} />
          </>
        }
        caption="Drag the vertical line to move the reference point x₀. A stationary kernel's curve only slides with x₀; the linear and cubic kernels change shape. Matérn ½ draws are rough, the RBF's are infinitely smooth, and the periodic kernel repeats. The draws use fixed normals, so they deform continuously as ℓ changes."
      >
        <Subplots rows={2} sharex heightRatios={[1, 1.4]}>
          <Panel>
            <XYChart
              series={covSeries}
              xLabel="x"
              yLabel="k(x, x₀)"
              handles={[{ kind: 'x', at: anchor, label: 'x₀', onDrag: (v) => setAnchor(Math.max(-3, Math.min(3, v))) }]}
              rescaleOnChange={false}
              axisKey={kind}
              holdFit="union"
            />
          </Panel>
          <Panel>
            <XYChart series={drawSeries} xLabel="x" yLabel="f(x)" legend={false} />
          </Panel>
        </Subplots>
      </Figure>
      <Figure
        title="The Gram matrix"
        description="k(xᵢ, xⱼ) on 41 evenly spaced inputs: a band for a short lengthscale, near rank one for a long one."
        caption="The same kernel as above. Stationary kernels give Toeplitz matrices (constant along diagonals)."
      >
        <Heatmap x={K.axis} y={K.axis} z={K.z} xLabel="xⱼ" yLabel="xᵢ" equalAspect />
      </Figure>
    </>
  )
}
