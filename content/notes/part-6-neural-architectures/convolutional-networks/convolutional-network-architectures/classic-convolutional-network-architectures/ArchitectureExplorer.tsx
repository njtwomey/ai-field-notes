import { useMemo } from 'react'
import { Bars, choice, Figure, formatNumber, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { networkCost } from 'aifn-compute/nn/functional'
import { modelZoo } from '../_shared/architectures.ts'

const ZOO = modelZoo().map((model) => ({ model, cost: networkCost(model.stages, model.input) }))

const millions = (v: number) => `${formatNumber(v / 1e6)} M`
const billions = (v: number) => `${formatNumber(v / 1e9)} G`
const side = (n: number) => (Number.isFinite(n) ? `${n} px` : 'whole image')

/**
 * Eleven image classifiers from LeNet-5 to ConvNeXt-T, each counted layer by layer from its specification: parameters
 * against multiply-adds per image, and the share of either in each stage of the chosen network.
 */
export function ArchitectureExplorer() {
  const state = useFigureState({
    model: choice(
      ZOO.map(({ model }) => ({ value: model.id, label: `${model.name} (${model.year})` })),
      'vgg-16',
      { label: 'network' },
    ),
    metric: choice(
      [
        { value: 'params', label: 'parameters' },
        { value: 'multiplies', label: 'multiply-adds per image' },
      ],
      'params',
      { label: 'stage breakdown' },
    ),
  })
  const chosen = ZOO.find((z) => z.model.id === state.model) ?? ZOO[0]
  const byParams = state.metric === 'params'

  const scatter = useMemo(
    () => ({
      x: ZOO.map((z) => z.cost.multiplies / 1e9),
      y: ZOO.map((z) => z.cost.params / 1e6),
      labels: ZOO.map((z) => z.model.name),
    }),
    [],
  )
  const rows = chosen.cost.layers
  const labels = rows.map((r) => r.label)
  const values = rows.map((r) => (byParams ? r.params / 1e6 : r.multiplies / 1e9))
  const features = rows.length > 1 ? rows[rows.length - 2] : rows[0]

  const cx = useAxis({ label: 'multiply-adds per image (billions)', log: true })
  const cy = useAxis({ label: 'parameters (millions)', log: true })
  const sx = useAxis({ label: byParams ? 'parameters (millions)' : 'multiply-adds (billions)', range: [0, undefined] })
  const sy = useAxis({ label: chosen.model.name, categories: labels, inverse: true })

  return (
    <Figure
      title="Size and cost of eleven image classifiers"
      state={state}
      defaultSize="L"
      caption="Left: every network's parameter count against its multiply-adds for one image, both on log scales, computed from the layer specifications (224 × 224 RGB input; 227 for AlexNet, whose paper sizes need it; 32 × 32 grey for LeNet-5). The chosen network is drawn in ink. Right: how its parameters or multiply-adds split across its stages. Pick VGG-16 and parameters: the three dense layers hold 124 of its 138 million weights, while the convolutions do almost all of the arithmetic. Pick ResNet-50: global average pooling leaves its classifier with 2 million weights, and the convolutions hold the rest. No accuracies are shown; the counts are exact for the specifications, which match torchvision's published parameter counts."
      readouts={
        <>
          <Readout label="parameters" value={millions(chosen.cost.params)} />
          <Readout label="multiply-adds" value={billions(chosen.cost.multiplies)} />
          <Readout label="weight layers on the longest path" value={chosen.cost.depth} />
          <Readout label="last feature map" value={features.outputShape.join(' × ')} />
          <Readout label="receptive field there" value={side(features.receptiveField.size[0])} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={cx} y={cy} height={320} title="parameters against multiply-adds">
          <Points name="networks" x={scatter.x} y={scatter.y} labels={scatter.labels} muted />
          <Points
            name={chosen.model.name}
            x={[chosen.cost.multiplies / 1e9]}
            y={[chosen.cost.params / 1e6]}
            labels={[chosen.model.name]}
            emphasis
            size={10}
          />
        </Plot>
        <Plot x={sx} y={sy} height={320} title={byParams ? 'parameters by stage' : 'multiply-adds by stage'}>
          <Bars
            name={byParams ? 'parameters' : 'multiply-adds'}
            x={labels.map((_, i) => i)}
            y={values}
            orient="y"
            slot={0}
          />
        </Plot>
      </div>
    </Figure>
  )
}
