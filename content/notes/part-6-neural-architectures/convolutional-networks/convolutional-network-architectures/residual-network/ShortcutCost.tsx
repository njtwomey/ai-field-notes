import { useMemo } from 'react'
import { Bars, choice, Figure, formatNumber, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { layerCost, networkCost, type LayerSpec } from 'aifn-compute/nn/functional'
import { resnet } from '../_shared/architectures.ts'

type Depth = 18 | 34 | 50 | 101 | 152
type Option = 'A' | 'B' | 'C'

const STAGES = ['stage 1', 'stage 2', 'stage 3', 'stage 4']
const AT = [0, 1, 2, 3]

/** A projection shortcut: 1×1 convolution with the block's stride, then batch normalisation. */
const projection = (out: number, stride: number): LayerSpec => ({
  kind: 'sequence',
  layers: [{ kind: 'conv', out, kernel: 1, stride, padding: 0, bias: false }, { kind: 'norm' }],
})

type Cost = { params: number; multiplies: number }
const ZERO: Cost = { params: 0, multiplies: 0 }
const add = (a: Cost, b: Cost, times = 1): Cost => ({
  params: a.params + times * b.params,
  multiplies: a.multiplies + times * b.multiplies,
})

/**
 * Per-stage cost of a ResNet with shortcut option A (identity, zero-padded channels), B (projection only where the
 * shape changes, the networks of the paper) or C (projection on every block). `networkCost` counts the network as
 * built, with option B; the shortcuts are counted on their own with `layerCost` at each block's input shape, so the
 * network under A or C is the B count minus its projections plus the option's.
 */
function shortcutCost(depth: Depth, option: Option) {
  const model = resnet(depth)
  const cost = networkCost(model.stages, model.input)
  let builtShortcuts = ZERO
  let shortcuts = ZERO
  let projections = 0
  let blocks = 0
  const stages = AT.map((i) => {
    const row = cost.layers[i + 1]
    const [inC, h, w] = row.inputShape
    const [out, h2, w2] = row.outputShape
    const n = model.stages[i + 1].layers.length
    const stride = h / h2
    const first = layerCost(projection(out, stride), [inC, h, w])
    const rest = layerCost(projection(out, 1), [out, h2, w2])
    const needed = stride !== 1 || inC !== out
    const built = needed ? first : ZERO
    const chosen = option === 'C' ? add(first, rest, n - 1) : option === 'B' ? built : ZERO
    builtShortcuts = add(builtShortcuts, built)
    shortcuts = add(shortcuts, chosen)
    projections += option === 'C' ? n : option === 'B' && needed ? 1 : 0
    blocks += n
    return { branchParams: row.params - built.params, shortcutParams: chosen.params }
  })
  return {
    stages,
    shortcuts,
    projections,
    blocks,
    params: cost.params - builtShortcuts.params + shortcuts.params,
    multiplies: cost.multiplies - builtShortcuts.multiplies + shortcuts.multiplies,
  }
}

/** Where the parameters of a ResNet sit, for identity, needed-only and all-projection shortcuts. */
export function ShortcutCost() {
  const state = useFigureState({
    depth: choice<Depth>([18, 34, 50, 101, 152], 50, { label: 'ResNet depth' }),
    option: choice<Option>(
      [
        { value: 'A', label: 'A: identity, zero-padded channels' },
        { value: 'B', label: 'B: projection where the shape changes' },
        { value: 'C', label: 'C: projection on every block' },
      ],
      'B',
      { label: 'shortcuts' },
    ),
  })
  const { depth, option } = state
  const c = useMemo(() => shortcutCost(depth, option), [depth, option])
  const identity = useMemo(() => shortcutCost(depth, 'A'), [depth])
  const extra = (100 * (c.multiplies - identity.multiplies)) / identity.multiplies

  const x = useAxis({ label: 'stage', categories: STAGES })
  const y = useAxis({ label: 'parameters (millions)', range: [0, undefined] })
  return (
    <Figure
      title="What shortcuts cost"
      state={state}
      caption="Parameters of each stage of a ResNet, split into the residual branches and the shortcuts, counted from the network's layer specification with batch normalisation included. Option A keeps every shortcut an identity and pads the extra channels with zeros, so its shortcuts cost nothing. Option B, used for the ImageNet ResNets of He et al. (2016), puts a 1 × 1 projection on each block whose shape changes: three blocks in ResNet-18 and -34, four in the bottleneck networks, whose first block widens 64 channels to 256. Option C puts a projection on every block. In a bottleneck network a projection has almost as many weights as the branch beside it, so in ResNet-50 option C adds 75% to 88% to the parameters of each stage."
      readouts={
        <>
          <Readout label="projection shortcuts" value={`${c.projections} of ${c.blocks} blocks`} />
          <Readout label="shortcut parameters" value={formatNumber(c.shortcuts.params)} />
          <Readout label="share of the network" value={`${formatNumber((100 * c.shortcuts.params) / c.params)}%`} />
          <Readout label="network parameters" value={`${formatNumber(c.params / 1e6)} M`} />
          <Readout label="multiply-adds per 224 × 224 image" value={`${formatNumber(c.multiplies / 1e9)} G`} />
          <Readout label="multiply-adds above option A" value={`${formatNumber(extra)}%`} />
        </>
      }
    >
      <Plot x={x} y={y} height={300}>
        <Bars
          name="residual branches"
          x={AT.map((i) => i - 0.2)}
          y={c.stages.map((s) => s.branchParams / 1e6)}
          width={0.38}
          slot={0}
        />
        <Bars
          name="shortcuts"
          x={AT.map((i) => i + 0.2)}
          y={c.stages.map((s) => s.shortcutParams / 1e6)}
          width={0.38}
          slot={1}
        />
      </Plot>
    </Figure>
  )
}
