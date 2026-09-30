import * as I from 'aifn/info'
import { normals, stream } from 'aifn/random'
import { binaryEntropy } from 'aifn/special'
import { toFlat, unwrap, type Value } from 'aifn/tensor'
import { trace } from 'aifn/trace'
import { mapTree, pathToRoot } from 'aifn/graph'
import { useContext, useMemo, useState, type ReactNode } from 'react'
import { Player, Select, Slider } from '@lab/controls'
import type { ElementState } from '@lab/diagram'
import { Figure } from '@lab/layout'
import { ChartSize, DEFAULT_HEIGHT, FrameContext, Readout, XYChart, type XYSeries } from '@lab/viz'
import { formatValue, TreeView } from '@lab/views'

const num = (v: Value) => {
  const r = unwrap(v)
  return typeof r === 'number' ? r : toFlat(r)[0]
}
const h2 = (p: number) => binaryEntropy(p, 2)

// ── Blahut–Arimoto capacity ──────────────────────────────────────────────────────────────────────────────────────────

type ChannelName = 'binary symmetric' | 'binary erasure' | 'Z channel' | 'noisy typewriter (5)' | 'asymmetric 3 × 3'

function channelOf(name: ChannelName, e: number): { W: number[][]; exact?: number } {
  switch (name) {
    case 'binary symmetric':
      return {
        W: [
          [1 - e, e],
          [e, 1 - e],
        ],
        exact: 1 - h2(e),
      }
    case 'binary erasure':
      return {
        W: [
          [1 - e, e, 0],
          [0, e, 1 - e],
        ],
        exact: 1 - e,
      }
    case 'Z channel':
      // Capacity log₂(1 + (1 − e) e^{e/(1−e)}) for a 1 → 0 flip with probability e.
      return {
        W: [
          [1, 0],
          [e, 1 - e],
        ],
        exact: e < 1 ? Math.log2(1 + (1 - e) * e ** (e / (1 - e))) : 0,
      }
    case 'noisy typewriter (5)':
      return {
        W: Array.from({ length: 5 }, (_, i) =>
          Array.from({ length: 5 }, (_, j) => (j === i ? 1 - e : j === (i + 1) % 5 ? e : 0)),
        ),
      }
    case 'asymmetric 3 × 3':
      return {
        W: [
          [1 - e, e / 2, e / 2],
          [e / 4, 1 - e / 2, e / 4],
          [0.1, 0.3, 0.6],
        ],
      }
  }
}

export function CapacitySpecimen() {
  const [name, setName] = useState<ChannelName>('Z channel')
  const [e, setE] = useState(0.3)
  const { W, exact } = useMemo(() => channelOf(name, e), [name, e])
  const run = useMemo(
    () =>
      trace(I.blahutArimotoCapacity, { channel: W, tolerance: 1e-12 }, 200, {
        record: {
          lower: (s) => s.lower / Math.LN2,
          upper: (s) => s.upper / Math.LN2,
          information: (s) => s.information / Math.LN2,
        },
      }),
    [W],
  )
  const last = run.steps[run.steps.length - 1]
  const series = useMemo((): XYSeries[] => {
    const x = run.index
    return [
      { name: 'upper bound maxₓ D(W‖q)', type: 'line', x, y: toFlat(run.series.upper), slot: 0 },
      { name: 'lower bound log Σ p e^D', type: 'line', x, y: toFlat(run.series.lower), slot: 1 },
      { name: 'I(p; W)', type: 'line', x, y: toFlat(run.series.information), slot: 2, dashed: true },
    ]
  }, [run])
  const input = useMemo(
    (): XYSeries[] => [
      { name: 'capacity-achieving p(x)', type: 'bar', x: W.map((_, i) => i), y: toFlat(last.input), slot: 0 },
    ],
    [W, last],
  )
  return (
    <Figure
      title="Blahut–Arimoto: channel capacity"
      defaultSize="L"
      controls={
        <>
          <Select
            label="channel"
            value={name}
            onChange={setName}
            options={['binary symmetric', 'binary erasure', 'Z channel', 'noisy typewriter (5)', 'asymmetric 3 × 3']}
          />
          <Slider label="noise e" value={e} min={0} max={0.99} onChange={setE} />
        </>
      }
      readouts={
        <>
          <Readout label="capacity (bits)" value={formatValue(last.lower / Math.LN2)} />
          {exact !== undefined && <Readout label="closed form" value={formatValue(exact)} />}
          <Readout label="iterations" value={last.iteration} />
          <Readout label="stopped" value={run.meta.stopped} />
        </>
      }
      caption="Each step reweights the input by exp D(W(·|x) ‖ q); the bounds squeeze the capacity from both sides and the run stops when they meet within 10⁻¹² nats."
    >
      <div className="flex flex-col gap-4">
        <ChartSize scale={0.6}>
          <XYChart series={series} xLabel="iteration" yLabel="bits" integerX rescaleOnChange={false} holdFit="union" />
        </ChartSize>
        <ChartSize scale={0.4}>
          <XYChart series={input} xLabel="input symbol x" yLabel="p(x)" integerX yRange={[0, 1]} />
        </ChartSize>
      </div>
    </Figure>
  )
}

// ── Rate–distortion ──────────────────────────────────────────────────────────────────────────────────────────────────

export function RateDistortionSpecimen() {
  const [p, setP] = useState(0.3)
  const curve = useMemo(() => {
    const betas = Array.from({ length: 60 }, (_, i) => 0.05 * 1.12 ** i)
    return I.rateDistortionCurve(
      [1 - p, p],
      [
        [0, 1],
        [1, 0],
      ],
      betas,
      { base: 2 },
    )
  }, [p])
  const series = useMemo((): XYSeries[] => {
    const ds = Array.from({ length: 200 }, (_, i) => (Math.min(p, 1 - p) * i) / 199)
    return [
      { name: 'Blahut–Arimoto points', type: 'scatter', x: toFlat(curve.distortion), y: toFlat(curve.rate), slot: 0 },
      { name: 'H(p) − H(D)', type: 'line', x: ds, y: ds.map((d) => h2(p) - h2(d)), slot: 1, dashed: true },
    ]
  }, [curve, p])
  return (
    <Figure
      title="Rate–distortion of a binary source under Hamming distortion"
      controls={<Slider label="source p(1)" value={p} min={0.01} max={0.5} onChange={setP} />}
      readouts={
        <>
          <Readout label="R(0) = H(p) (bits)" value={formatValue(h2(p))} />
          <Readout label="all converged" value={String(curve.converged)} />
        </>
      }
      caption="Each point is one Blahut–Arimoto run at slope −β; together they trace R(D) = H(p) − H(D) for D ≤ min(p, 1 − p)."
    >
      <XYChart
        series={series}
        xLabel="distortion D"
        yLabel="rate R (bits)"
        yRange={[0, undefined]}
        rescaleOnChange={false}
        holdFit="union"
      />
    </Figure>
  )
}

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
  const run = useMemo(() => trace(I.huffmanSteps, probs, k), [probs, k])
  const final = run.steps[run.steps.length - 1]
  const tree = useMemo(() => I.huffmanTree(final), [final])
  const code = useMemo(() => I.huffmanCode(probs), [probs])
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
          <Readout label="Kraft sum Σ 2^−ℓ" value={formatValue(I.kraftSum(code.lengths))} />
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
  const huffman = useMemo(() => I.huffmanCode(probs), [probs])
  const fano = useMemo(() => I.shannonFanoCode(probs), [probs])
  const shannon = useMemo(() => I.shannonCode(probs), [probs])
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

// ── Divergences ──────────────────────────────────────────────────────────────────────────────────────────────────────

export function DivergencesSpecimen() {
  const [p, setP] = useState(0.3)
  const qs = useMemo(() => Array.from({ length: 199 }, (_, i) => (i + 1) / 200), [])
  const series = useMemo((): XYSeries[] => {
    const P = [p, 1 - p]
    const at = (f: (q: number[]) => number) => qs.map((q) => f([q, 1 - q]))
    return [
      { name: 'KL(p ‖ q)', type: 'line', x: qs, y: at((q) => num(I.klDivergence(P, q))), slot: 0 },
      { name: 'KL(q ‖ p)', type: 'line', x: qs, y: at((q) => num(I.klDivergence(q, P))), slot: 1 },
      { name: 'Jensen–Shannon', type: 'line', x: qs, y: at((q) => num(I.jensenShannonDivergence(P, q))), slot: 2 },
      { name: 'total variation', type: 'line', x: qs, y: at((q) => num(I.totalVariation(P, q))), slot: 3 },
      { name: 'Hellinger', type: 'line', x: qs, y: at((q) => num(I.hellingerDistance(P, q))), slot: 4 },
      {
        name: 'Pearson χ² (fDivergence)',
        type: 'line',
        x: qs,
        y: at((q) => I.fDivergence(P, q, I.fGenerators.pearsonChiSquare)),
        slot: 5,
        dashed: true,
      },
    ]
  }, [p, qs])
  return (
    <Figure
      title="Divergences between two Bernoulli distributions"
      controls={<Slider label="p" value={p} min={0.01} max={0.99} onChange={setP} />}
      caption="All vanish at q = p. KL is asymmetric and unbounded; Jensen–Shannon is at most log 2; total variation and Hellinger are at most 1."
    >
      <XYChart series={series} xLabel="q" yLabel="nats / distance" yRange={[0, 2]} />
    </Figure>
  )
}

// ── Mutual information from samples ──────────────────────────────────────────────────────────────────────────────────

export function KsgSpecimen() {
  const [rho, setRho] = useState(0.6)
  const [n, setN] = useState(500)
  const [k, setK] = useState(3)
  const { x, y } = useMemo(() => {
    const s = stream('ksg')
    const a = toFlat(normals(s.child('x'), n))
    const e = toFlat(normals(s.child('e'), n))
    return { x: a, y: a.map((v, i) => rho * v + Math.sqrt(1 - rho * rho) * e[i]) }
  }, [rho, n])
  const estimate = useMemo(() => I.ksgMutualInformation(x, y, { k }), [x, y, k])
  return (
    <Figure
      title="Kraskov–Stögbauer–Grassberger estimate of I(X; Y)"
      controls={
        <>
          <Slider label="correlation ρ" value={rho} min={-0.99} max={0.99} onChange={setRho} />
          <Slider label="samples n" value={n} min={50} max={2000} step={50} onChange={setN} />
          <Slider label="neighbours k" value={k} min={1} max={20} step={1} onChange={setK} />
        </>
      }
      readouts={
        <>
          <Readout label="KSG estimate (nats)" value={formatValue(estimate)} />
          <Readout label="Gaussian −½ log(1 − ρ²)" value={formatValue(-0.5 * Math.log(1 - rho * rho))} />
        </>
      }
    >
      <XYChart
        series={[{ name: 'samples', type: 'scatter', x, y, slot: 0 }]}
        xLabel="x"
        yLabel="y"
        equalAspect
        rescaleOnChange={false}
        holdFit="union"
      />
    </Figure>
  )
}
