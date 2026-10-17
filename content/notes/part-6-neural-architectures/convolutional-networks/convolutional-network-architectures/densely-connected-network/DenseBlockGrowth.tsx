import { useMemo } from 'react'
import {
  Bars,
  Diagram,
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  setting,
  slider,
  useAxis,
  useFigureState,
  type DiagramSpec,
} from 'aifn-render'
import { networkCost, type LayerSpec } from 'aifn-compute/nn/functional'
import { denseLayer } from '../_shared/architectures.ts'

/** A dense layer without the 1×1 bottleneck: BN, ReLU, 3×3 convolution to k channels, concatenated to its input. */
const plainLayer = (k: number): LayerSpec => ({
  kind: 'concat',
  branches: [
    [],
    [{ kind: 'norm' }, { kind: 'activation' }, { kind: 'conv', out: k, kernel: 3, padding: 1, bias: false }],
  ],
})

function blockDiagram(layers: number, k0: number, k: number): DiagramSpec {
  const gap = 1.9
  const nodes: DiagramSpec['nodes'] = [
    { id: 'n0', x: 0, y: 2, w: 1.2, h: 0.8, label: '$\\xvec_0$', tone: 'neutral', notes: { s: `${k0}` } },
    ...Array.from({ length: layers }, (_, i) => ({
      id: `n${i + 1}`,
      x: (i + 1) * gap,
      y: 2,
      w: 1.1,
      h: 0.8,
      label: `$H_{${i + 1}}$`,
      tone: 0,
      notes: { s: `${k0 + i * k} in` },
    })),
    {
      id: 'out',
      x: (layers + 1) * gap + 0.2,
      y: 2,
      w: 1.3,
      h: 0.8,
      shape: 'pill' as const,
      label: 'concat',
      tone: 'neutral',
      notes: { s: `${k0 + layers * k}` },
    },
  ]
  const ids = nodes.map((n) => n.id)
  const edges: DiagramSpec['edges'] = []
  for (let a = 0; a < ids.length; a++)
    for (let b = a + 1; b < ids.length; b++)
      edges.push(
        b === a + 1
          ? { from: ids[a], to: ids[b] }
          : { from: `${ids[a]}:n`, to: `${ids[b]}:n`, route: 'curve', bend: 0.32 * (b - a), arrow: 'end' },
      )
  return { nodes, edges, unit: 34 }
}

/** A dense block: every layer reads the concatenation of the block input and all earlier layers' outputs. */
export function DenseBlockGrowth() {
  const state = useFigureState({
    layers: int(4, { min: 1, max: 8, label: 'layers in the block L' }),
    growth: slider(4, 64, 32, { step: 4, label: 'growth rate k' }),
    k0: slider(16, 256, 64, { step: 16, label: 'block input channels k₀' }),
    size: int(56, { min: 7, max: 112, label: 'feature map side', suggestions: [7, 14, 28, 56] }),
    bottleneck: setting(true, '1×1 bottleneck to 4k channels (DenseNet-B)'),
  })
  const { layers, growth: k, k0, size, bottleneck } = state

  const cost = useMemo(() => {
    const specs = Array.from({ length: layers }, () => (bottleneck ? denseLayer(k) : plainLayer(k)))
    return networkCost(specs, [k0, size, size])
  }, [layers, k, k0, size, bottleneck])
  const spec = useMemo(() => blockDiagram(layers, k0, k), [layers, k0, k])

  const at = Array.from({ length: layers }, (_, i) => i + 1)
  const x = useAxis({ label: 'layer ℓ', range: [0.4, layers + 0.6], integer: true })
  const y = useAxis({ label: 'multiply-adds of layer ℓ (millions)', range: [0, undefined] })
  return (
    <Figure
      title="Channels and cost inside a dense block"
      state={state}
      defaultSize="L"
      caption="Top: a dense block. Each layer H_ℓ reads the concatenation of the block input x₀ and the outputs of all earlier layers (every arc), so layer ℓ receives k₀ + (ℓ − 1)k channels (written under it) and adds k more; the block outputs k₀ + Lk. Bottom: the multiply-adds of each layer for one image, which grow linearly along the block because each layer reads more channels. Switch the bottleneck off to see where it pays: with k₀ = 64 and k = 32 a bare 3×3 layer is cheaper in the first layers, and the 1×1 bottleneck wins once a layer reads more than about 230 channels."
      readouts={
        <>
          <Readout label="block output channels" value={cost.outputShape[0]} />
          <Readout label="after a transition (θ = 0.5)" value={Math.floor(cost.outputShape[0] / 2)} />
          <Readout label="connections into the layers, L(L + 1)/2" value={(layers * (layers + 1)) / 2} />
          <Readout label="parameters" value={formatNumber(cost.params)} />
          <Readout label="multiply-adds" value={`${formatNumber(cost.multiplies / 1e6)} M`} />
        </>
      }
    >
      <Diagram spec={spec} ariaLabel={`Dense block with ${layers} layers; every layer connects to every later layer`} />
      <Plot x={x} y={y} height={220}>
        <Bars name="multiply-adds" x={at} y={cost.layers.map((l) => l.multiplies / 1e6)} width={0.6} slot={0} />
      </Plot>
    </Figure>
  )
}
