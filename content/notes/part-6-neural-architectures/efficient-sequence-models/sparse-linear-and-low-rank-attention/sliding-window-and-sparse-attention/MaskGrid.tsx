import { useMemo } from 'react'
import { Figure, formatNumber, int, Plot, Plots, Raster, Readout, setting, useAxis, useFigureState } from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { toRows } from 'aifn-compute/foundation/tensor'
import { attentionReach, maskDensity, type SparsePattern, sparseAttentionMask } from 'aifn-compute/nn/attention'

const PANELS: { pattern: SparsePattern; title: string }[] = [
  { pattern: 'full', title: 'full attention' },
  { pattern: 'sliding', title: 'sliding window' },
  { pattern: 'dilated', title: 'dilated window' },
  { pattern: 'strided', title: 'strided (Sparse Transformer)' },
  { pattern: 'global', title: 'window + global tokens' },
  { pattern: 'bigbird', title: 'BigBird: window + global + random' },
]

/** Six attention patterns side by side, each with the inputs that a stack of L such layers lets every output read. */
export function MaskGrid() {
  const state = useFigureState({
    n: int(32, { min: 8, max: 64, step: 4, suggestions: [16, 32, 48, 64], label: 'sequence length n' }),
    w: int(2, { min: 1, max: 8, label: 'half-width w' }),
    dilation: int(2, { min: 2, max: 4, label: 'dilation δ' }),
    stride: int(6, { min: 2, max: 16, label: 'stride l' }),
    g: int(1, { min: 0, max: 4, label: 'global tokens g' }),
    r: int(2, { min: 0, max: 4, label: 'random keys r' }),
    layers: int(1, { min: 1, max: 8, suggestions: [1, 2, 3, 4], label: 'layers L' }),
    causal: setting(false, 'causal'),
  })
  const { n, w, dilation, stride, g, r, layers, causal } = state

  const panels = useMemo(
    () =>
      PANELS.map(({ pattern, title }) => {
        const maskTensor = sparseAttentionMask(n, pattern, {
          window: w,
          dilation,
          stride,
          globals: g,
          random: r,
          causal,
          stream: stream(11),
        })
        const mask = toRows(maskTensor)
        const reach = toRows(attentionReach(maskTensor, layers))
        // Category 0: scored by one layer. Category 1: reached only through the stack. Empty: out of reach.
        const z = mask.map((row, i) => row.map((v, j) => (v ? 0 : reach[i][j] ? 1 : Number.NaN)))
        const reached = reach.reduce((a, row) => a + row.reduce((s, v) => s + v, 0), 0)
        const visible = causal ? (n * (n + 1)) / 2 : n * n
        return { title, z, keys: maskDensity(maskTensor).keysPerQuery, reach: reached / visible }
      }),
    [n, w, dilation, stride, g, r, layers, causal],
  )
  const pos = useMemo(() => Array.from({ length: n }, (_, i) => i), [n])

  const keyAxis = useAxis({ label: 'key position j', integer: true, hold: 'union' })
  const queryAxis = useAxis({ label: 'query position i', integer: true, inverse: true, hold: 'union' })
  const list = (f: (p: (typeof panels)[number]) => string) => panels.map(f).join(' · ')

  return (
    <Figure
      title="Sparse patterns and their reach"
      state={state}
      caption="Each panel is the mask of one pattern: row i is a query, column j a key. Cells in the first colour are the scores one layer computes; with L > 1, cells in the second colour are inputs that output i can still read through L stacked layers of the same pattern (with the residual path), and empty cells are out of reach. A sliding window of half-width w reaches Lw positions to each side. Dilation δ spreads the same number of keys δ times as far but leaves gaps that only later layers fill. The strided pattern (the previous l keys and every l-th key) connects every pair in two layers. One global token (the first position) gives every pair a path in two layers. BigBird's random keys add long jumps on top of the window and global tokens. Set the half-width, dilation, stride, number of global tokens and random keys, and the number of layers."
      readouts={
        <>
          <Readout label="keys per query" value={list((p) => formatNumber(p.keys))} />
          <Readout
            label={`pairs within reach after ${layers} layer${layers > 1 ? 's' : ''}`}
            value={list((p) => `${formatNumber(100 * p.reach)}%`)}
          />
          <Readout label="order" value="full · sliding · dilated · strided · global · BigBird" />
        </>
      }
    >
      <Plots cols={2} rows={3}>
        {panels.map((p) => (
          <Plot key={p.title} x={keyAxis} y={queryAxis} title={p.title}>
            <Raster
              x={pos}
              y={pos}
              z={p.z}
              scale="categorical"
              categoryNames={['scored by one layer', `reached through ${layers} layers`]}
            />
          </Plot>
        ))}
      </Plots>
    </Figure>
  )
}
