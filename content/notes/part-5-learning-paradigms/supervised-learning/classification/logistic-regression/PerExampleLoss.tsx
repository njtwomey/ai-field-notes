import { useMemo } from 'react'
import {
  choice,
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
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { sigmoid } from 'aifn/numerics/special'

type Label = '1' | '0'

const Z_MIN = -6
const Z_MAX = 6
const GRID = toFlat(linspace(Z_MIN, Z_MAX, 241))

/** log(1 + e^z), without overflow for large z. */
const softplus = (z: number) => Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)))

/**
 * One example's cross-entropy ℓ(z) = log(1 + e^z) − yz as a function of its score z, with ℓ′ = σ(z) − y and
 * ℓ″ = σ(z)(1 − σ(z)). The quadratic model at the chosen score is the one Newton's method minimises; its minimiser is
 * the IRLS working response.
 */
export function PerExampleLoss() {
  const state = useFigureState({
    label: choice<Label>(
      [
        { value: '1', label: 'y = 1' },
        { value: '0', label: 'y = 0' },
      ],
      '1',
      { label: 'label y' },
    ),
    score: float(-2, { min: Z_MIN, max: Z_MAX, step: 0.05, label: 'score z' }),
  })
  const y = state.label === '1' ? 1 : 0
  const z0 = state.score
  const p = sigmoid(z0)
  const loss = (z: number) => softplus(z) - y * z
  const g = p - y
  const h = p * (1 - p)
  const working = z0 - g / h

  const curves = useMemo(
    (): SeriesSpec[] => [
      { name: 'loss ℓ(z)', type: 'line', x: GRID, y: GRID.map((z) => softplus(z) - y * z), slot: 0 },
      { name: 'gradient ℓ′(z) = σ(z) − y', type: 'line', x: GRID, y: GRID.map((z) => sigmoid(z) - y), slot: 1 },
      {
        name: 'curvature ℓ″(z) = σ(z)(1 − σ(z))',
        type: 'line',
        x: GRID,
        y: GRID.map((z) => sigmoid(z) * (1 - sigmoid(z))),
        slot: 2,
      },
    ],
    [y],
  )

  const series = useMemo((): SeriesSpec[] => {
    const l0 = softplus(z0) - y * z0
    const model = GRID.map((z) => l0 + g * (z - z0) + 0.5 * h * (z - z0) ** 2)
    const shown = working >= Z_MIN && working <= Z_MAX
    return [
      ...curves,
      { name: 'quadratic model at z', type: 'line', x: GRID, y: model, slot: 3, dashed: true },
      ...(shown
        ? [
            {
              name: 'working response',
              type: 'scatter' as const,
              x: [working],
              y: [l0 - (0.5 * g * g) / h],
              emphasis: true,
            },
          ]
        : []),
    ]
  }, [curves, z0, y, g, h, working])

  const xAxis = useAxis({ label: 'score z', range: [Z_MIN, Z_MAX] })
  const yAxis = useAxis({ label: 'value', range: [-1.1, 4] })
  return (
    <Figure
      title="One example: loss, gradient and curvature"
      state={state}
      caption="The cross-entropy of a single example as a function of its score z = wᵀx, with its first and second derivatives. Drag the vertical line, or use the slider, to move the score. The dashed curve is the quadratic model that Newton's method minimises at that score. The diamond is its minimiser, the IRLS working response z + (y − p)/(p(1 − p)). A confidently wrong example (y = 1, z far below 0) has gradient near −1 but curvature near 0, so its quadratic model is almost flat and its working response lies far away. A confidently right example has both near 0."

      readouts={
        <>
          <Readout label="p = σ(z)" value={formatNumber(p)} />
          <Readout label="loss" value={formatNumber(loss(z0))} />
          <Readout label="gradient p − y" value={formatNumber(g)} />
          <Readout label="curvature p(1 − p)" value={formatNumber(h)} />
          <Readout label="working response" value={Math.abs(working) < 1e6 ? formatNumber(working) : '∞'} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        {seriesLayers(series)}
        <Handle kind="x" at={z0} label="score z" onDrag={(v: number) => state.set('score', v)} />
      </Plot>
    </Figure>
  )
}
