import * as Coding from 'aifn-applied/information/coding'
import { toFlat } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { mapTree, pathToRoot } from 'aifn/graph'
import { useContext, useMemo, useState, type ReactNode } from 'react'
import { Player, Select, Slider } from '@lab/controls'
import type { ElementState } from '@lab/diagram'
import { Figure } from '@lab/layout'
import { DEFAULT_HEIGHT, FrameContext, Readout, XYChart, type XYSeries } from '@lab/viz'
import { formatValue, TreeView } from '@lab/views'

// ── Huffman ──────────────────────────────────────────────────────────────────────────────────────────────────────────

const SYMBOLS = 'abcdefghijklmnopqrstuvwx'

/** Zipf probabilities pₖ ∝ 1/(k + 1)ˢ over K symbols. */
function zipf(k: number, s: number): number[] {
  const w = Array.from({ length: k }, (_, i) => 1 / (i + 1) ** s)
  const total = w.reduce((a, b) => a + b, 0)
  return w.map((v) => v / total)
}

const p2 = (v: number) => v.toFixed(2).replace(/^0\./, '.')

type HuffmanLayout = 'code length' | 'merge order'

/** The Huffman tree with its merges stepped through and a codeword path highlighted. */
function HuffmanTreeFigure() {
  const [k, setK] = useState(8)
  const [s, setS] = useState(1)
  const [layout, setLayout] = useState<HuffmanLayout>('code length')
  const [chosen, setChosen] = useState('c')
  const [hovered, setHovered] = useState<number | null>(null)
  const probs = useMemo(() => zipf(k, s), [k, s])
  const run = useMemo(() => trace(Coding.huffmanSteps(probs), undefined, k), [probs, k])
  const final = run.steps[run.steps.length - 1]
  const tree = useMemo(() => Coding.huffmanTree(final), [final])
  const code = useMemo(() => Coding.huffmanCode(probs), [probs])
  // Merge order as a height: leaves at 0, the node made by merge m at m, so the tree grows upwards step by step.
  const byMerge = useMemo(() => mapTree(tree, (n) => ({ ...n, height: n.id < k ? 0 : n.id - k + 1 })), [tree, k])
  const [step, setStep] = useState<number | null>(null)
  const last = run.steps.length - 1
  const at = Math.min(step ?? last, last)
  const state = run.steps[at]
  const symbol = hovered ?? Math.max(0, Math.min(SYMBOLS.indexOf(chosen), k - 1))
  const path = useMemo(() => pathToRoot(tree, symbol), [tree, symbol])
  const merged = new Set(state.merged ?? [])
  const created = state.created

  const nodeLabels = (v: number) => (v < k ? SYMBOLS[v] : p2(tree.nodes[v].probability))
  const nodeNotes = (v: number) => (v < k ? p2(tree.nodes[v].probability) : undefined)
  const hidden = (v: number) => v >= k + at
  const nodeState = (v: number): ElementState | undefined =>
    at === 0 ? undefined : v === created || merged.has(v) ? 'active' : 'done'
  const edgeState = (c: number): ElementState | undefined => (at === 0 ? undefined : merged.has(c) ? 'active' : 'done')
  const describe =
    at === 0
      ? `${k} symbols in the queue`
      : `merge ${state.merged!.map((v) => nodeLabels(v)).join(' + ')} → ${p2(tree.nodes[created].probability)}`

  return (
    <Figure
      title="Huffman's algorithm builds the code tree"
      description="Each step takes the two least probable nodes off the queue and joins them under a new node whose probability is their sum; the first gets bit 0, the second bit 1. A symbol's codeword is the path from the root to its leaf."
      defaultSize="L"
      hoverReadout={false}
      controls={
        <>
          <Slider label="symbols K" value={k} min={2} max={SYMBOLS.length} step={1} onChange={setK} />
          <Slider label="Zipf exponent s" value={s} min={0} max={3} onChange={setS} />
          <Select
            label="layout"
            value={layout}
            onChange={setLayout}
            options={['code length', 'merge order'] as HuffmanLayout[]}
          />
          <Select
            label="codeword of"
            value={SYMBOLS[Math.min(SYMBOLS.indexOf(chosen), k - 1)]}
            onChange={setChosen}
            options={SYMBOLS.slice(0, k).split('')}
          />
          <div className="col-span-full">
            <Player label="merges" value={at} onChange={setStep} count={run.steps.length} defaultSpeed={1.5} />
          </div>
        </>
      }
      readouts={
        <>
          <Readout label="step" value={describe} />
          <Readout label="queue" value={state.queue.map((v) => nodeLabels(v)).join(' ')} />
          <Readout label={`codeword of ${SYMBOLS[symbol]}`} value={state.codewords[symbol] || '—'} />
          <Readout
            label="E[ℓ] against H (bits)"
            value={`${formatValue(code.expectedLength)} ≥ ${formatValue(code.entropy)}`}
          />
          <Readout label="Kraft sum Σ 2^−ℓ" value={formatValue(Coding.kraftSum(code.lengths))} />
        </>
      }
      caption="pₖ ∝ 1/(k + 1)ˢ. Leaves show symbol and probability, internal nodes the merged probability; the two nodes merged by the current step and their new parent are orange, the chosen symbol's codeword path is blue, and nodes not yet made are hidden. Codewords grow from their last bit to their first: each merge prepends a bit to every symbol under the two merged nodes. Hover or pick a symbol to trace its codeword; 'merge order' draws each merge one level above the last, so the tree grows upwards as the queue shrinks."
    >
      <HuffmanBody
        footer={
          <div className="flex flex-wrap gap-1 font-mono text-xs tabular-nums">
            {state.codewords.map((c, v) => (
              <span
                key={v}
                className={`rounded border px-1.5 py-0.5 ${v === symbol ? 'border-foreground bg-muted' : 'text-muted-foreground'}`}
              >
                {SYMBOLS[v]} {c || '·'}
              </span>
            ))}
          </div>
        }
      >
        <TreeView
          tree={layout === 'code length' ? tree : byMerge}
          heightAxis={layout === 'merge order'}
          height="fill"
          ariaLabel="Huffman tree"
          nodeLabels={nodeLabels}
          nodeNotes={nodeNotes}
          shape={(v) => (v < k ? 'circle' : 'pill')}
          hidden={hidden}
          nodeState={nodeState}
          edgeState={edgeState}
          nodeTone={(v) => (at > 0 && (v === created || merged.has(v)) ? 1 : undefined)}
          edgeTone={(c) => (at > 0 && merged.has(c) ? 1 : undefined)}
          highlight={path}
          onNodeHover={(v) => setHovered(v !== null && v < k ? v : null)}
          onNodeClick={(v) => v < k && setChosen(SYMBOLS[v])}
        />
      </HuffmanBody>
    </Figure>
  )
}

/** A body filling the frame's height, with a footer below it. */
function HuffmanBody({ children, footer }: { children: ReactNode; footer: ReactNode }) {
  const { height } = useContext(FrameContext)
  return (
    <div className="flex flex-col gap-2" style={{ height: height ?? DEFAULT_HEIGHT }}>
      <div className="min-h-0 flex-1">{children}</div>
      <div className="shrink-0">{footer}</div>
    </div>
  )
}

/** Codeword lengths of the Huffman and Shannon–Fano codes against the ideal −log₂ p. */
function HuffmanLengthsFigure() {
  const [k, setK] = useState(12)
  const [s, setS] = useState(1.2)
  const probs = useMemo(() => zipf(k, s), [k, s])
  const huffman = useMemo(() => Coding.huffmanCode(probs), [probs])
  const fano = useMemo(() => Coding.shannonFanoCode(probs), [probs])
  const shannon = useMemo(() => Coding.shannonCode(probs), [probs])
  const lengths = useMemo(
    (): XYSeries[] => [
      {
        name: '−log₂ pₖ',
        type: 'line',
        x: probs.map((_, i) => i),
        y: probs.map((p) => -Math.log2(p)),
        slot: 2,
        dashed: true,
      },
      { name: 'Huffman length', type: 'bar', x: probs.map((_, i) => i), y: toFlat(huffman.lengths), slot: 0 },
      { name: 'Shannon–Fano length', type: 'scatter', x: probs.map((_, i) => i), y: toFlat(fano.lengths), slot: 1 },
    ],
    [probs, huffman, fano],
  )
  return (
    <Figure
      title="Codeword lengths against the ideal"
      description="An optimal code gives symbol k about −log₂ pₖ bits; Huffman lengths are whole numbers near it, so H ≤ E[ℓ] < H + 1."
      controls={
        <>
          <Slider label="symbols K" value={k} min={2} max={24} step={1} onChange={setK} />
          <Slider label="Zipf exponent s" value={s} min={0} max={3} onChange={setS} />
        </>
      }
      readouts={
        <>
          <Readout label="entropy H (bits)" value={formatValue(huffman.entropy)} />
          <Readout label="Huffman E[ℓ]" value={formatValue(huffman.expectedLength)} />
          <Readout label="Shannon–Fano E[ℓ]" value={formatValue(fano.expectedLength)} />
          <Readout label="Shannon E[ℓ]" value={formatValue(shannon.expectedLength)} />
        </>
      }
      caption="pₖ ∝ 1/(k + 1)ˢ. At s = 0 every symbol is equally likely and the lengths differ by at most one bit."
    >
      <XYChart
        series={lengths}
        xLabel="symbol k"
        yLabel="bits"
        integerX
        rescaleOnChange={false}
        holdFit="union"
        axisKey={k}
      />
    </Figure>
  )
}

export function HuffmanSpecimen() {
  return (
    <>
      <HuffmanTreeFigure />
      <HuffmanLengthsFigure />
    </>
  )
}
