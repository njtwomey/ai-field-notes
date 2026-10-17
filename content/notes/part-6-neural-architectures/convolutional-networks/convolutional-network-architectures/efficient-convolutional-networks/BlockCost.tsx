import { useMemo } from 'react'
import { Bars, choice, Figure, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { layerCost, type LayerSpec } from 'aifn-compute/nn/functional'
import { invertedResidual, resnetBlock } from '../_shared/architectures.ts'

const bn: LayerSpec = { kind: 'norm' }
const act: LayerSpec = { kind: 'activation' }

/** The blocks compared, each mapping C channels to C channels at stride 1. */
function blocks(C: number, t: number, g: number): { label: string; short: string; spec: LayerSpec }[] {
  const narrow = Math.max(g, Math.round(C / 4 / g) * g)
  return [
    {
      label: '3×3 convolution',
      short: '3×3',
      spec: { kind: 'sequence', layers: [{ kind: 'conv', out: C, kernel: 3, padding: 1, bias: false }, bn, act] },
    },
    {
      label: 'ResNet bottleneck (C/4 inside)',
      short: 'bottleneck',
      spec: resnetBlock(C, Math.max(1, Math.round(C / 4)), 1, true),
    },
    {
      label: 'depthwise separable (MobileNet)',
      short: 'separable',
      spec: {
        kind: 'sequence',
        layers: [
          { kind: 'depthwise', kernel: 3, padding: 1, bias: false },
          bn,
          act,
          { kind: 'conv', out: C, kernel: 1, bias: false },
          bn,
          act,
        ],
      },
    },
    {
      label: `inverted residual, t = ${t} (MobileNetV2)`,
      short: 'inverted residual',
      spec: invertedResidual(C, C, { expand: t }),
    },
    {
      label: `ShuffleNet unit, g = ${g}`,
      short: 'ShuffleNet',
      spec: {
        kind: 'add',
        branches: [
          [
            { kind: 'conv', out: narrow, kernel: 1, groups: g, bias: false },
            bn,
            act,
            { kind: 'depthwise', kernel: 3, padding: 1, bias: false },
            bn,
            { kind: 'conv', out: C, kernel: 1, groups: g, bias: false },
            bn,
          ],
          [],
        ],
      },
    },
  ]
}

/** Weights and multiply-adds of five blocks that map C channels to C channels on one feature map. */
export function BlockCost() {
  const state = useFigureState({
    channels: int(256, { min: 16, max: 1024, step: 8, label: 'channels C', suggestions: [32, 64, 256, 512] }),
    size: int(14, { min: 7, max: 112, label: 'feature map side', suggestions: [7, 14, 28, 56] }),
    expand: int(6, { min: 1, max: 8, label: 'expansion t of the inverted residual' }),
    groups: choice([2, 4, 8], 8, { label: 'ShuffleNet groups g' }),
    metric: choice(
      [
        { value: 'multiplies', label: 'multiply-adds' },
        { value: 'params', label: 'parameters' },
      ],
      'multiplies',
      { label: 'show' },
    ),
  })
  const { channels: C, size, expand, groups, metric } = state
  const rows = useMemo(
    () =>
      blocks(C, expand, groups).map(({ label, short, spec }) => ({
        label,
        short,
        ...layerCost(spec, [C, size, size]),
      })),
    [C, size, expand, groups],
  )
  const byParams = metric === 'params'
  const scale = byParams ? 1e3 : 1e6
  const labels = rows.map((r) => r.label)
  const x = useAxis({ label: byParams ? 'parameters (thousands)' : 'multiply-adds (millions)', range: [0, undefined] })
  const y = useAxis({ label: 'block', categories: labels, inverse: true })
  const full = rows[0]

  return (
    <Figure
      title="What a block costs at the same width"
      state={state}
      caption="Five blocks that each map a C-channel feature map to C channels at stride 1, counted with batch-norm parameters and without biases. The plain 3×3 convolution costs 9C² per pixel. The depthwise separable block costs 9C + C², the inverted residual 2tC² + 9tC, the ResNet bottleneck about C²/2 + 9C²/16 and the ShuffleNet unit about C²/(2g) + 9C/4. At C = 256 the separable block costs 8.7 times less than the plain convolution. Lower C to 32: the depthwise term 9C and the 1×1 term C² become comparable, and the expanded inverted residual becomes more expensive than the plain 3×3 layer it replaces."
      readouts={rows.slice(1).map((r) => (
        <Readout
          key={r.short}
          label={`3×3 ÷ ${r.short}, multiply-adds`}
          value={`${formatNumber(full.multiplies / r.multiplies)}×`}
        />
      ))}
    >
      <Plot x={x} y={y} height={280}>
        <Bars
          name={byParams ? 'parameters' : 'multiply-adds'}
          x={rows.map((_, i) => i)}
          y={rows.map((r) => (byParams ? r.params : r.multiplies) / scale)}
          orient="y"
          slot={0}
        />
      </Plot>
    </Figure>
  )
}
