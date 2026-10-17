import { useMemo } from 'react'
import { Annotation, Bars, Figure, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { convolutionCost, convolutionStackCost } from 'aifn-compute/nn/functional'

const ROWS = [
  '3 × 3, C → C′ (plain)',
  'bottleneck total',
  '1 × 1 reduce, C → B',
  '3 × 3, B → B',
  '1 × 1 expand, B → C′',
]

const millions = (v: number) => `${formatNumber(v / 1e6)} M`

/**
 * Parameters and multiply-accumulates of one 3 × 3 layer against a 1 × 1, 3 × 3, 1 × 1 bottleneck with the same input
 * and output channels, counted layer by layer.
 */
export function BottleneckCost() {
  const state = useFigureState({
    cin: int(256, { ge: 1, le: 4096, suggestions: [64, 128, 256, 512, 1024, 2048], label: 'input channels C' }),
    width: int(64, { ge: 1, le: 4096, suggestions: [16, 32, 64, 128, 256, 512], label: 'bottleneck width B' }),
    cout: int(256, { ge: 1, le: 4096, suggestions: [64, 128, 256, 512, 1024, 2048], label: 'output channels C′' }),
    size: int(56, { ge: 1, le: 512, suggestions: [7, 14, 28, 56, 112], label: 'feature map side H = W' }),
  })
  const { cin, width, cout, size } = state

  const r = useMemo(() => {
    const shape = [size, size]
    const plain = convolutionCost({
      inChannels: cin,
      outChannels: cout,
      kernel: 3,
      padding: 1,
      bias: false,
      input: shape,
    })
    const neck = convolutionStackCost({ channels: cin, shape }, [
      { kernel: 1, outChannels: width, bias: false },
      { kernel: 3, outChannels: width, padding: 1, bias: false },
      { kernel: 1, outChannels: cout, bias: false },
    ])
    const macs = [plain.multiplies, neck.multiplies, ...neck.layers.map((l) => l.multiplies)]
    const params = [plain.params, neck.params, ...neck.layers.map((l) => l.params)]
    return { plain, neck, macs, params }
  }, [cin, width, cout, size])

  const xAxis = useAxis({
    label: 'multiply-accumulates per image (millions)',
    hold: 'union',
    key: `${cin}-${cout}-${size}`,
  })
  const yAxis = useAxis({ categories: ROWS, inverse: true })
  const ratio = r.plain.params / r.neck.params
  return (
    <Figure
      title="The bottleneck: 1 × 1, 3 × 3, 1 × 1"
      state={state}
      caption="One plain 3 × 3 layer from C to C′ channels against a bottleneck that reduces to B channels with a 1 × 1 convolution, applies the 3 × 3 there, and expands to C′ with a second 1 × 1. Bars are multiply-accumulates for one image at the chosen feature-map size, with parameter counts written beside them (no biases). Type a channel count or pick a suggestion. At ResNet-50's first stage (C = C′ = 256, B = 64, 56 × 56) the bottleneck needs 8.5 times fewer weights and multiply-accumulates. Raise B towards C and the saving disappears."
      readouts={
        <>
          <Readout label="plain 3 × 3 weights" value={r.plain.params.toLocaleString('en-GB')} />
          <Readout label="bottleneck weights" value={r.neck.params.toLocaleString('en-GB')} />
          <Readout label="saving" value={`${formatNumber(ratio)} ×`} />
          <Readout label="plain 3 × 3 multiply-accumulates" value={millions(r.plain.multiplies)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} ariaLabel="Multiply-accumulates of a plain layer and of a bottleneck">
        <Bars name="plain" x={[0]} y={[r.macs[0] / 1e6]} orient="y" slot={0} />
        <Bars name="bottleneck" x={[1, 2, 3, 4]} y={r.macs.slice(1).map((v) => v / 1e6)} orient="y" slot={1} />
        {r.params.map((p, k) => (
          <Annotation key={k} at={[r.macs[k] / 1e6, k]} text={`${p.toLocaleString('en-GB')} weights`} />
        ))}
      </Plot>
    </Figure>
  )
}
