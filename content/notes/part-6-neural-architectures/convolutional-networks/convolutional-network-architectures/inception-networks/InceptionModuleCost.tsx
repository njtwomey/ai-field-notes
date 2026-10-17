import { useMemo } from 'react'
import { Bars, Figure, formatNumber, int, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { layerCost, type LayerSpec } from 'aifn-compute/nn/functional'
import { inceptionModule, type InceptionWidths } from '../_shared/architectures.ts'

const BRANCHES = ['1×1 branch', '3×3 branch', '5×5 branch', 'pool branch']

type Costs = { branches: { params: number; multiplies: number; channels: number }[]; total: number; out: number }

function costs(w: InceptionWidths, channels: number, size: number, reduce: boolean): Costs {
  const input = [channels, size, size]
  const module = inceptionModule(w, { reduce }) as { branches: readonly (readonly LayerSpec[])[] }
  const branches = module.branches.map((b) => {
    const r = layerCost({ kind: 'sequence', layers: b }, input)
    return { params: r.params, multiplies: r.multiplies, channels: r.outputShape[0] }
  })
  return {
    branches,
    total: branches.reduce((s, b) => s + b.multiplies, 0),
    out: branches.reduce((s, b) => s + b.channels, 0),
  }
}

const millions = (v: number) => `${formatNumber(v / 1e6)} M`

/** One Inception module with and without its 1×1 reductions: multiply-adds per branch and the output width. */
export function InceptionModuleCost() {
  const state = useFigureState({
    channels: int(192, { min: 16, max: 1024, step: 16, label: 'input channels C', suggestions: [192, 256, 480, 832] }),
    size: int(28, { min: 7, max: 56, label: 'input height and width', suggestions: [7, 14, 28] }),
    w1: slider(8, 256, 64, { step: 8, label: '1×1 filters' }),
    r3: slider(8, 256, 96, { step: 8, label: '3×3 reduce' }),
    w3: slider(8, 384, 128, { step: 8, label: '3×3 filters' }),
    r5: slider(4, 64, 16, { step: 4, label: '5×5 reduce' }),
    w5: slider(4, 128, 32, { step: 4, label: '5×5 filters' }),
    wp: slider(8, 128, 32, { step: 8, label: 'pool projection' }),
  })
  const w: InceptionWidths = [state.w1, state.r3, state.w3, state.r5, state.w5, state.wp]
  const reduced = useMemo(() => costs(w, state.channels, state.size, true), [state.channels, state.size, ...w])
  const naive = useMemo(() => costs(w, state.channels, state.size, false), [state.channels, state.size, ...w])
  const at = BRANCHES.map((_, i) => i)

  const x = useAxis({ label: 'multiply-adds per image (millions)', range: [0, undefined] })
  const y = useAxis({ label: 'branch', categories: BRANCHES, inverse: true })
  return (
    <Figure
      title="What the 1×1 reductions save in an Inception module"
      state={state}
      caption="Multiply-adds of each branch of one Inception module for one image, with the 1×1 reductions (GoogLeNet's design) and without them (the naive module, where the 3×3 and 5×5 filters read all C input channels and the pool branch passes all C channels on). The defaults are GoogLeNet's module 3a on a 28 × 28 × 192 input. Lower the 3×3 or 5×5 reduce widths to cut the cost of those branches; raise C to see the naive cost grow in proportion while the reduced branches grow only through their 1×1 layers."
      readouts={{
        'with reductions': (
          <>
            <Readout label="multiply-adds" value={millions(reduced.total)} />
            <Readout label="output channels" value={reduced.out} />
          </>
        ),
        naive: (
          <>
            <Readout label="multiply-adds" value={millions(naive.total)} />
            <Readout label="output channels" value={naive.out} />
            <Readout label="cost ratio" value={`${formatNumber(naive.total / reduced.total)}×`} />
          </>
        ),
      }}
    >
      <Plot x={x} y={y} height={280}>
        <Bars
          name="naive"
          x={at.map((i) => i - 0.2)}
          y={naive.branches.map((b) => b.multiplies / 1e6)}
          width={0.38}
          orient="y"
          slot={1}
        />
        <Bars
          name="with 1×1 reductions"
          x={at.map((i) => i + 0.2)}
          y={reduced.branches.map((b) => b.multiplies / 1e6)}
          width={0.38}
          orient="y"
          slot={0}
        />
      </Plot>
    </Figure>
  )
}
