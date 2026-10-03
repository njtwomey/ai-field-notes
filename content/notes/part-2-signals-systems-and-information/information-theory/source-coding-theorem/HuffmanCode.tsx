import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'

const log2 = (x: number) => Math.log(x) / Math.LN2

const PRESETS = {
  dyadic: [0.5, 0.25, 0.125, 0.125],
  textbook: [0.4, 0.2, 0.2, 0.1, 0.1],
  skewed: [0.7, 0.1, 0.1, 0.05, 0.05],
  uniform: [0.2, 0.2, 0.2, 0.2, 0.2],
} as const
type Preset = keyof typeof PRESETS
const SYMBOLS = ['a', 'b', 'c', 'd', 'e']

type Node = { p: number; order: number; symbol?: number; children?: [Node, Node] }

/**
 * Huffman's algorithm: repeatedly merge the two least probable nodes. Ties are broken by creation order, so the
 * result is deterministic. Returns the codeword for each symbol, reading 0 for the first child and 1 for the second.
 */
function huffman(probs: readonly number[]): string[] {
  let order = 0
  let nodes: Node[] = probs.map((p, symbol) => ({ p, order: order++, symbol }))
  while (nodes.length > 1) {
    nodes = [...nodes].sort((x, y) => x.p - y.p || x.order - y.order)
    const [lo, hi, ...rest] = nodes
    nodes = [...rest, { p: lo.p + hi.p, order: order++, children: [lo, hi] }]
  }
  const codes = probs.map(() => '')
  const walk = (node: Node, prefix: string) => {
    if (node.symbol !== undefined) codes[node.symbol] = prefix || '0'
    else if (node.children) {
      walk(node.children[0], `${prefix}0`)
      walk(node.children[1], `${prefix}1`)
    }
  }
  walk(nodes[0], '')
  return codes
}

/** Code lengths from Huffman's algorithm against the ideal −log₂ p and the Shannon lengths ⌈−log₂ p⌉. */
export function HuffmanCode() {
  const [preset, setPreset] = useState<Preset>('textbook')
  const probs = PRESETS[preset]

  const r = useMemo(() => {
    const codes = huffman(probs)
    const ideal = probs.map((p) => -log2(p))
    const shannon = ideal.map((l) => Math.ceil(l - 1e-12))
    const huff = codes.map((c) => c.length)
    const expected = (lengths: number[]) => probs.reduce((s, p, i) => s + p * lengths[i], 0)
    return {
      codes,
      ideal,
      shannon,
      huff,
      entropy: probs.reduce((s, p) => s - p * log2(p), 0),
      huffLength: expected(huff),
      shannonLength: expected(shannon),
      kraft: huff.reduce((s, l) => s + 2 ** -l, 0),
    }
  }, [probs])

  const index = probs.map((_, i) => i + 1)
  const series: XYSeries[] = [
    { name: 'Huffman length', type: 'bar', x: index, y: r.huff, slot: 0 },
    { name: 'Shannon length ⌈−log₂ p⌉', type: 'scatter', x: index, y: r.shannon, slot: 1 },
    { name: 'ideal −log₂ p', type: 'scatter', x: index, y: r.ideal, emphasis: true },
  ]

  return (
    <Interactive
      title="Huffman coding"
      caption="Each symbol gets a binary codeword; no codeword is a prefix of another, so a stream of codewords decodes without separators. Bars: Huffman's code lengths. Diamonds: the ideal lengths −log₂ p, which only an entropy-achieving code would use. Circles: Shannon's rounded-up lengths ⌈−log₂ p⌉. For the dyadic distribution every probability is a power of 1/2 and Huffman meets the entropy exactly."
      controls={
        <ParamChoice
          label="distribution"
          value={preset}
          onChange={setPreset}
          options={[
            { value: 'dyadic', label: '½, ¼, ⅛, ⅛' },
            { value: 'textbook', label: '0.4, 0.2, 0.2, 0.1, 0.1' },
            { value: 'skewed', label: '0.7, 0.1, 0.1, 0.05, 0.05' },
            { value: 'uniform', label: 'uniform over 5' },
          ]}
        />
      }
      readout={
        <>
          <Readout label="code" value={r.codes.map((c, i) => `${SYMBOLS[i]}:${c}`).join('  ')} />
          <Readout label="entropy H" value={`${formatNumber(r.entropy)} bits`} />
          <Readout label="Huffman E[L]" value={`${formatNumber(r.huffLength)} bits`} />
          <Readout label="Shannon E[L]" value={`${formatNumber(r.shannonLength)} bits`} />
          <Readout label="Kraft sum Σ 2^−ℓ" value={formatNumber(r.kraft)} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="symbol (a, b, c, …)"
        yLabel="bits"
        xRange={[0.5, 5.5]}
        yRange={[0, 5]}
        height={300}
      />
    </Interactive>
  )
}
