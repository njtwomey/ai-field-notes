import { useMemo } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'

type KChoice = '4' | '8' | '16' | '32' | '64'
const K_OPTIONS: KChoice[] = ['4', '8', '16', '32', '64']

const S_MAX = 3
const S_STEP = 0.05
const S_GRID = Array.from({ length: Math.round(S_MAX / S_STEP) + 1 }, (_, i) => Number((i * S_STEP).toFixed(2)))
const S_RANGE: [number, number] = [0, S_MAX]
const DEPTH_RANGE: [number | undefined, number | undefined] = [0, undefined]

/** Zipf class frequencies: pi_k proportional to k^(-s), for k = 1..K. */
function zipf(K: number, s: number): number[] {
  const w = Array.from({ length: K }, (_, k) => Math.pow(k + 1, -s))
  const total = w.reduce((a, v) => a + v, 0)
  return w.map((v) => v / total)
}

/** Leaf depths of a Huffman tree: repeatedly merge the two lightest groups; every leaf in a merge gets one deeper. */
function huffmanDepths(p: number[]): number[] {
  const depth = p.map(() => 0)
  let groups = p.map((w, k) => ({ w, leaves: [k] }))
  while (groups.length > 1) {
    groups.sort((a, b) => a.w - b.w)
    const [a, b, ...rest] = groups
    for (const k of [...a.leaves, ...b.leaves]) depth[k] += 1
    groups = [...rest, { w: a.w + b.w, leaves: [...a.leaves, ...b.leaves] }]
  }
  return depth
}

const entropyBits = (p: number[]) => -p.reduce((a, v) => (v > 0 ? a + v * Math.log2(v) : a), 0)

function evaluate(K: number, s: number) {
  const p = zipf(K, s)
  const depths = huffmanDepths(p)
  return {
    p,
    depths,
    huffman: p.reduce((a, v, k) => a + v * depths[k], 0),
    entropy: entropyBits(p),
    balanced: Math.log2(K),
  }
}

/**
 * Expected classifier evaluations per prediction for a balanced tree and a Huffman tree over K classes with Zipf
 * frequencies, against the entropy lower bound and the entropy + 1 upper bound on the Huffman tree.
 */
export function TreeDepthExplorer() {
  const state = useFigureState({
    kChoice: choice<KChoice>(
      K_OPTIONS.map((k) => ({ value: k, label: k })),
      '16',
      { label: 'number of classes K' },
    ),
    s: float(1, { min: 0, max: S_MAX, step: S_STEP, label: 'Zipf exponent s' }),
  })
  const K = Number(state.kChoice)

  const curves = useMemo(() => {
    const rows = S_GRID.map((v) => evaluate(K, v))
    return [
      { name: 'balanced tree', x: S_GRID, y: rows.map((r) => r.balanced), slot: 0 },
      { name: 'Huffman tree', x: S_GRID, y: rows.map((r) => r.huffman), slot: 1 },
      { name: 'entropy H', x: S_GRID, y: rows.map((r) => r.entropy), slot: 2 },
      { name: 'H + 1', x: S_GRID, y: rows.map((r) => r.entropy + 1), slot: 2, dashed: true },
    ] as const
  }, [K])

  const now = useMemo(() => evaluate(K, state.s), [K, state.s])

  const ranks = useMemo(() => now.depths.map((_, k) => k + 1), [now])
  const balancedLine = useMemo(() => ({ x: [0.5, K + 0.5], y: [now.balanced, now.balanced] }), [now, K])

  const barRange = useMemo((): [number, number] => [0.5, K + 0.5], [K])

  const xAxis = useAxis({ label: 'Zipf exponent s', range: S_RANGE })
  const yAxis = useAxis({ label: 'evaluations per prediction', range: DEPTH_RANGE })
  const rankAxis = useAxis({ label: 'class k, most frequent first', range: barRange, integer: true })
  const depthAxis = useAxis({ label: 'depth', range: DEPTH_RANGE })
  return (
    <Figure
      title="Expected depth of balanced and Huffman trees"
      state={state}
      caption="Class k of K occurs with frequency proportional to k^(−s). The top chart shows the expected number of node classifiers evaluated per prediction, when predictions follow the class frequencies, as the skew s grows; drag the vertical line (or use the slider) to set s. The balanced tree always costs log2 K. The Huffman tree stays between the entropy H and H + 1 and falls with H as the frequencies become skewed. The bottom chart shows each class's Huffman depth at the chosen s: frequent classes get short paths, rare ones long paths."
      readouts={
        <>
          <Readout label="entropy H (bits)" value={formatNumber(now.entropy)} />
          <Readout label="Huffman expected depth" value={formatNumber(now.huffman)} />
          <Readout label="balanced depth" value={formatNumber(now.balanced)} />
          <Readout label="most frequent class" value={formatNumber(now.p[0])} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={280}>
        <Curve {...curves[0]} />
        <Curve {...curves[1]} />
        <Curve {...curves[2]} />
        <Curve {...curves[3]} />
        <Handle {...state.handle('s', { label: 'Zipf exponent s' })} />
      </Plot>
      <Plot x={rankAxis} y={depthAxis} height={200}>
        <Bars name="Huffman depth of class k" x={ranks} y={now.depths} slot={1} />
        <Curve name="balanced depth" x={balancedLine.x} y={balancedLine.y} slot={0} dashed />
      </Plot>
    </Figure>
  )
}
