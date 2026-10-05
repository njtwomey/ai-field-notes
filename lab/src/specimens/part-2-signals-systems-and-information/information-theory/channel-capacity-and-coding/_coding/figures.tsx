/**
 * Showcase: Huffman coding. Everything drawn is read from aifn's `huffmanSteps` trace (each state's queue, the nodes it
 * popped, the node it created and where the queue put it, the ties its rule decided, the symbols under each popped node
 * and the partial codewords), from `huffmanCode`, `canonicalCode`, `sourceExtension`, `prefixEncode` and
 * `prefixDecode`. The lab only places the forest (`forest.ts`) and colours it: symbol k keeps palette slot k on the
 * tree, the table, the bars and the bit string.
 */
import {
  canonicalCode,
  huffmanCode,
  huffmanSteps,
  kraftSum,
  prefixDecode,
  prefixEncode,
  sourceExtension,
  type HuffmanState,
  type HuffmanTies,
} from 'aifn-methods/information/coding'
import { categorical, stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { Pin, PinOff, Shuffle } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Player, StatusText } from 'aifn-render/controls'
import { SLOTS, useTheme, type Mode } from 'aifn-render/design'
import { Diagram, type DiagramEdge, type DiagramNode, type DiagramSpec } from 'aifn-render/diagram'
import { Dashboard, DashboardCell, DashboardRow, Figure } from 'aifn-render/layout'
import { choice, pinField, row, slider, useFigureState, usePinned, when } from 'aifn-render/state'
import { Button } from 'aifn-render/ui/button'
import { Input } from 'aifn-render/ui/input'
import { cn } from 'aifn-render/lib/utils'
import { Annotation, Bars, Curve, Plot, Points, Readout, useAxis } from 'aifn-render/viz'
import { formatValue } from '@lab/views'
import { forestLayout, useTweened, type Point } from './forest'
import { fmtP, makeSource, SOURCE_OPTIONS, symbolColor, type Source, type SourceName } from './sources'

const TIE_OPTIONS: { value: HuffmanTies; label: string }[] = [
  { value: 'minimum-variance', label: 'minimum variance (originals first)' },
  { value: 'merged-first', label: 'merged nodes first' },
]
const TIE_TEXT: Record<HuffmanTies, string> = {
  'minimum-variance': 'minimum variance',
  'merged-first': 'merged first',
}
const OTHER: Record<HuffmanTies, HuffmanTies> = {
  'minimum-variance': 'merged-first',
  'merged-first': 'minimum-variance',
}
const ARITY_OPTIONS = [
  { value: 2, label: '2 (bits)' },
  { value: 3, label: '3' },
  { value: 4, label: '4' },
]
/** Sources whose number of symbols is a parameter. */
const SIZED = new Set<string>(['zipf', 'uniform', 'dyadic', 'skewed', 'english'])
const f3 = (v: number) => formatValue(Number(v.toPrecision(3)))
const unit = (arity: number) => (arity === 2 ? 'bits' : `${arity}-ary digits`)
const logName = (arity: number) => `−log${arity === 2 ? '₂' : arity === 3 ? '₃' : '₄'} p`

/** A source's controls: the distribution, and K, s and q where the distribution takes them. */
const sourceRow = (maxK: number, initial: SourceName, k: number) =>
  row('1 · source', {
    name: choice(SOURCE_OPTIONS, initial, { label: 'distribution' }),
    k: slider(2, maxK, k, { label: 'symbols K', step: 1, when: (v) => SIZED.has(v.name as string) }),
    s: slider(0, 3, 1, { label: 'Zipf exponent s', step: 0.05, when: when('name', 'zipf') }),
    q: slider(0.5, 0.98, 0.85, { label: 'P(a)', step: 0.01, when: when('name', 'skewed') }),
  })

// ── Naming nodes in explanations ─────────────────────────────────────────────────────────────────────────────────

/** A node as text: a leaf by its symbol and probability, a dummy as ∅, a merged node by its symbols and weight. */
function nodeName(state: HuffmanState, names: readonly string[], v: number): string {
  const n = state.nodes[v]
  if (n.dummy) return '∅ (0)'
  if (n.children.length === 0) return `${names[n.symbol]} (${fmtP(n.weight)})`
  const symbols: string[] = []
  const stack = [v]
  while (stack.length) {
    const w = state.nodes[stack.pop()!]
    if (w.children.length) stack.push(...w.children)
    else if (!w.dummy) symbols.push(names[w.symbol])
  }
  symbols.sort()
  return symbols.length <= 3
    ? `{${symbols.join(' ')}} (${fmtP(n.weight)})`
    : `the ${symbols.length}-symbol node (${fmtP(n.weight)})`
}

const list = (items: string[]) =>
  items.length <= 1 ? items.join('') : `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`

/** One step of the trace in words: what was popped, what each popped subtree's symbols gained, where the new node went. */
function explainStep(state: HuffmanState, names: readonly string[]): string {
  if (state.t === 0) {
    const dummies = state.nodes.filter((n) => n.dummy).length
    return (
      `Step 0: the queue holds the ${state.symbols} symbols, least probable first: ${state.queue
        .map((v) => nodeName(state, names, v))
        .join(', ')}.` +
      (dummies
        ? ` ${dummies} zero-weight dumm${dummies > 1 ? 'ies (∅) pad' : 'y (∅) pads'} the queue so that every merge takes ${state.arity} nodes.`
        : '') +
      ` Each step merges the ${state.arity} least probable.`
    )
  }
  const popped = state.popped.map((v) => nodeName(state, names, v))
  const gains = state.popped
    .map((_, i) => {
      const syms = state.under[i].map((k) => names[k])
      return syms.length ? `${syms.join(' ')} ${syms.length > 1 ? 'gain' : 'gains'} ${i.toString(36)}` : null
    })
    .filter((x): x is string => x !== null)
  const created = state.nodes[state.created]
  const tie =
    state.tied.length > 0
      ? ` Tie at ${fmtP(state.nodes[state.popped[state.popped.length - 1]].weight)}: the ${TIE_TEXT[state.ties]} rule popped these before ${list(state.tied.map((v) => nodeName(state, names, v)))}.`
      : ''
  const where = state.done
    ? ' One node is left: the tree is complete.'
    : ` The new node goes back into the queue at place ${state.inserted + 1} of ${state.queue.length}.`
  return `Step ${state.t}: merge ${list(popped)} → ${fmtP(created.weight)}; every symbol under them gains a leading digit (${gains.join('; ')}).${tie}${where}`
}

// ── The forest as a diagram ──────────────────────────────────────────────────────────────────────────────────────

/** The nodes from a symbol's leaf up to its current root. */
function pathUp(state: HuffmanState, leaf: number): number[] {
  const out: number[] = []
  for (let v: number | null = leaf; v !== null; v = state.nodes[v].parent) out.push(v)
  return out
}

function forestSpec(
  state: HuffmanState,
  shown: Map<number, Point>,
  target: Map<number, Point>,
  names: readonly string[],
  focus: number | null,
  frame: DiagramSpec['frame'],
): DiagramSpec {
  const at = (v: number) => shown.get(v) ?? target.get(v)!
  const path = new Set(focus === null ? [] : pathUp(state, focus))
  const active = new Set(state.t > 0 ? [...state.popped, state.created] : [])
  const nodes: DiagramNode[] = state.nodes.map((n) => {
    const p = at(n.id)
    const common = {
      id: `n${n.id}`,
      x: p.x,
      y: p.y,
      // The focused leaf keeps its symbol's colour and takes an ink ring; the merged nodes above it are emphasised.
      highlight: (path.has(n.id) && n.children.length > 0) || undefined,
      selected: n.id === focus || undefined,
      state: active.has(n.id) ? ('active' as const) : undefined,
    }
    if (n.dummy) return { ...common, shape: 'circle', w: 0.5, h: 0.5, dashed: true, tone: 'neutral', notes: { s: '0' } }
    if (n.children.length === 0)
      return {
        ...common,
        shape: 'circle',
        w: 0.8,
        h: 0.8,
        label: names[n.symbol],
        ariaLabel: `symbol ${names[n.symbol]}: click to pin its codeword`,
        tone: n.symbol < SLOTS ? n.symbol : 'neutral',
        notes: { s: fmtP(n.weight) },
      }
    return { ...common, shape: 'pill', w: 1, h: 0.6, label: fmtP(n.weight), tone: 'ink' }
  })
  const edges: DiagramEdge[] = state.nodes
    .filter((n) => n.parent !== null)
    .map((n) => {
      const parent = n.parent!
      const left = at(n.id).x < at(parent).x - 1e-6
      return {
        from: `n${parent}`,
        to: `n${n.id}`,
        route: 'straight',
        arrow: 'none',
        label: n.digit!.toString(36),
        labelRotate: false,
        labelPos: 0.45,
        labelSide: left ? 'right' : 'left',
        state: parent === state.created ? 'active' : undefined,
        highlight: (path.has(n.id) && path.has(parent)) || undefined,
      }
    })
  const roots = state.queue.map(at)
  const x0 = Math.min(...roots.map((p) => p.x)) - 0.75
  const x1 = Math.max(...roots.map((p) => p.x)) + 0.75
  return {
    nodes,
    edges,
    groups: [
      {
        id: 'queue',
        rect: { x: x0, y: -1.05, w: x1 - x0, h: 1.67 },
        label: state.done ? 'the finished tree' : 'priority queue: pops from the left',
        labelAt: 'top-left',
        dashed: true,
        tone: 'neutral',
      },
    ],
    frame,
    accent: 'ink',
    fitLabels: false,
    maxScale: 1.6,
  }
}

// ── The codeword table ───────────────────────────────────────────────────────────────────────────────────────────

function CodewordTable({
  source,
  state,
  final,
  canonical,
  arity,
  focus,
  pinned,
  mode,
  onHover,
  onPin,
}: {
  source: Source
  state: HuffmanState
  final: readonly string[]
  canonical: readonly string[]
  arity: number
  focus: number | null
  pinned: number | null
  mode: Mode
  onHover: (k: number | null) => void
  onPin: (k: number) => void
}) {
  const th = 'px-2 py-1 text-left font-normal text-muted-foreground'
  return (
    <div className="h-full overflow-auto">
      <table className="w-full border-collapse text-xs tabular-nums" onMouseLeave={() => onHover(null)}>
        <thead>
          <tr className="border-b">
            <th className={th}>pin</th>
            <th className={th}>symbol</th>
            <th className={th}>p</th>
            <th className={th}>{logName(arity)}</th>
            <th className={th}>codeword at step {state.t}</th>
            <th className={th}>final ℓ</th>
            <th className={th}>Huffman</th>
            <th className={th}>canonical</th>
          </tr>
        </thead>
        <tbody>
          {source.names.map((name, k) => {
            const now = state.codewords[k]
            return (
              <tr
                key={k}
                className={cn('border-b border-border/50', k === focus && 'bg-muted')}
                onMouseEnter={() => onHover(k)}
              >
                <td className="px-2 py-0.5">
                  <button
                    type="button"
                    aria-label={`pin ${name}`}
                    aria-pressed={pinned === k}
                    className="rounded p-0.5 text-muted-foreground hover:text-foreground"
                    onClick={() => onPin(k)}
                  >
                    {pinned === k ? <PinOff className="size-3.5" /> : <Pin className="size-3.5" />}
                  </button>
                </td>
                <td className="px-2 py-0.5">
                  <span className="inline-flex items-center gap-1.5 font-mono">
                    <span className="size-2.5 rounded-full" style={{ background: symbolColor(mode, k) }} />
                    {name}
                  </span>
                </td>
                <td className="px-2 py-0.5 font-mono">{fmtP(source.p[k])}</td>
                <td className="px-2 py-0.5 font-mono">
                  {source.p[k] > 0 ? f3(-Math.log(source.p[k]) / Math.log(arity)) : '∞'}
                </td>
                <td className="px-2 py-0.5 font-mono">
                  {now ? (
                    <span style={{ color: symbolColor(mode, k) }}>
                      <span className="text-muted-foreground">{'·'.repeat(final[k].length - now.length)}</span>
                      {now}
                    </span>
                  ) : (
                    <span className="text-muted-foreground">in the queue</span>
                  )}
                </td>
                <td className="px-2 py-0.5 font-mono">{final[k].length}</td>
                <td className="px-2 py-0.5 font-mono">{final[k]}</td>
                <td className="px-2 py-0.5 font-mono">{canonical[k]}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/** The pinned (or hovered) symbol's codeword spelled out digit by digit, or why it has none yet. */
function Spelled({
  state,
  names,
  focus,
  finalLength,
}: {
  state: HuffmanState
  names: readonly string[]
  focus: number | null
  finalLength: number
}) {
  if (focus === null)
    return <StatusText>Click a leaf, or a pin in the table, to follow a symbol&apos;s codeword.</StatusText>
  const name = names[focus]
  const word = state.codewords[focus]
  if (word === '')
    return (
      <StatusText tone="attention">
        Not yet assigned: {name} is still in the queue as a leaf. Its first digit comes when it is merged.
      </StatusText>
    )
  const path = pathUp(state, focus).reverse()
  const hops = path
    .slice(1)
    .map((v) => `—${state.nodes[v].digit!.toString(36)}→ ${v === focus ? name : fmtP(state.nodes[v].weight)}`)
    .join(' ')
  if (!state.done)
    return (
      <StatusText>
        {name} so far: {word} ({word.length} of {finalLength} digits), read from its subtree&apos;s root{' '}
        {fmtP(state.nodes[path[0]].weight)} {hops}. That root is still in the queue; each later merge of it puts one
        more digit in front.
      </StatusText>
    )
  return (
    <StatusText tone="attention">
      {name} = {word.split('').join(' ')}: root {hops}. Each edge on the path is one digit, read from the root down.
    </StatusText>
  )
}

// ── Encoding and decoding a message ──────────────────────────────────────────────────────────────────────────────

function Encoder({
  source,
  code,
  mode,
  focus,
  onHover,
}: {
  source: Source
  code: ReturnType<typeof huffmanCode>
  mode: Mode
  focus: number | null
  onHover: (k: number | null) => void
}) {
  const [draws, setDraws] = useState(0)
  const sample = (seed: number) =>
    Array.from(toFlat(categorical(stream(`showcase-huffman/message/${seed}`), source.p, { shape: [24] }) as Tensor))
      .map((k) => source.names[k])
      .join('')
  const [text, setText] = useState<string | null>(null)
  const message = text ?? sample(draws)
  const parsed = useMemo(() => {
    const symbols: number[] = []
    let skipped = 0
    for (const ch of message.toLowerCase()) {
      const k = source.names.indexOf(ch)
      if (k < 0) skipped++
      else symbols.push(k)
    }
    return { symbols, skipped }
  }, [message, source])
  const result = useMemo(() => {
    try {
      const encoded = prefixEncode(code.codewords, parsed.symbols)
      const decoded = prefixDecode(code.tree, encoded.digits)
      return { encoded, decoded, error: null }
    } catch (e) {
      return { encoded: null, decoded: null, error: (e as Error).message }
    }
  }, [code, parsed])
  const fixed = Math.ceil(Math.log(source.names.length) / Math.log(code.arity) - 1e-12)
  const n = parsed.symbols.length
  return (
    <div className="flex h-full flex-col gap-2 text-xs">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-muted-foreground">message</span>
        <Input
          aria-label="message to encode"
          className="h-7 w-72 font-mono text-xs"
          value={message}
          onChange={(e) => setText(e.target.value)}
        />
        <Button
          variant="outline"
          size="sm"
          aria-label="Sample a message"
          onClick={() => {
            setDraws(draws + 1)
            setText(null)
          }}
        >
          <Shuffle /> sample 24 symbols from p
        </Button>
        {parsed.skipped > 0 && (
          <StatusText tone="attention">
            skipped {parsed.skipped} character{parsed.skipped > 1 ? 's' : ''} not in the alphabet{' '}
            {source.names.join('')}
          </StatusText>
        )}
      </div>
      {result.error !== null ? (
        <StatusText tone="error">{result.error}</StatusText>
      ) : (
        <>
          <div className="flex flex-wrap gap-y-1 font-mono" onMouseLeave={() => onHover(null)}>
            {result.encoded!.spans.map((s, i) => (
              <span
                key={i}
                className={cn('inline-flex flex-col items-center rounded px-0.5', s.symbol === focus && 'bg-muted')}
                onMouseEnter={() => onHover(s.symbol)}
              >
                <span style={{ color: symbolColor(mode, s.symbol) }} className="text-sm font-medium">
                  {result.encoded!.digits.slice(s.start, s.end)}
                </span>
                <span className="text-muted-foreground">{source.names[s.symbol]}</span>
              </span>
            ))}
          </div>
          <div className="flex flex-wrap gap-x-5 gap-y-1">
            <StatusText>
              decoded by walking the tree from the root, back to the root at each leaf:{' '}
              <span className="font-mono text-foreground">
                {result.decoded!.symbols.map((k) => source.names[k]).join('')}
              </span>{' '}
              {result.decoded!.symbols.join() === parsed.symbols.join() && result.decoded!.rest === ''
                ? '(matches)'
                : '(differs)'}
            </StatusText>
            <StatusText>
              {result.encoded!.digits.length} {unit(code.arity)} for {n} symbols
              {n > 0 ? ` = ${f3(result.encoded!.digits.length / n)} per symbol` : ''}; E[ℓ] = {f3(code.expectedLength)};
              a fixed-length code needs {fixed}
            </StatusText>
          </div>
        </>
      )}
    </div>
  )
}

// ── Figure 1: the algorithm, step by step ────────────────────────────────────────────────────────────────────────

function HuffmanBuildFigure() {
  const { resolved: mode } = useTheme()
  const figure = useFigureState({
    source: sourceRow(8, 'zipf', 6),
    code: row('2 · code', {
      arity: choice(ARITY_OPTIONS, 2, { label: 'code alphabet D' }),
      ties: choice(TIE_OPTIONS, 'minimum-variance', { label: 'ties' }),
    }),
    pin: pinField(),
  })
  const { name, k, s, q } = figure.source
  const arity = figure.code.arity as number
  const ties = figure.code.ties as HuffmanTies
  const source = useMemo(() => makeSource(name as SourceName, k, s, q), [name, k, s, q])
  const K = source.names.length
  const runKey = `${name}/${K}/${s}/${q}/${arity}/${ties}`
  const steps = useMemo(
    () => trace(huffmanSteps(source.p, { arity, ties }), undefined, 4 * K).steps,
    [source, arity, ties, K],
  )
  const code = useMemo(() => huffmanCode(source.p, { arity, ties }), [source, arity, ties])
  const other = useMemo(() => huffmanCode(source.p, { arity, ties: OTHER[ties] }), [source, arity, ties])
  const canonical = useMemo(() => canonicalCode(code.lengths, arity), [code, arity])
  const lengths = useMemo(() => Array.from(toFlat(code.lengths)), [code])
  const [step, setStep] = useState(0)
  const last = steps.length - 1
  const t = Math.min(step, last)
  const state = steps[t]
  // Hover a symbol to preview its codeword, click to pin it (again or Escape: unpin); the pin is kept in the link.
  const pins = usePinned(figure.pin, (v) => figure.set('pin', v), { valid: (v) => v < K })
  const { pinned, focus } = pins
  const setHover = pins.hover
  const togglePin = pins.toggle

  const layouts = useMemo(() => steps.map(forestLayout), [steps])
  // One frame for the whole run, so the drawing keeps its scale from the first step to the last.
  const frame = useMemo(() => {
    const all = layouts.flatMap((m) => [...m.values()])
    return {
      x0: Math.min(...all.map((p) => p.x)) - 0.8,
      y0: -1.4,
      x1: Math.max(...all.map((p) => p.x)) + 0.8,
      y1: Math.max(...all.map((p) => p.y)) + 0.95,
    }
  }, [layouts])
  const target = layouts[t]
  const shown = useTweened(target, runKey)
  const spec = useMemo(
    () => forestSpec(state, shown, target, source.names, focus, frame),
    [state, shown, target, source.names, focus, frame],
  )
  const kraft = kraftSum(code.lengths, arity)

  return (
    <Figure
      title="Huffman's algorithm, merge by merge"
      purpose="A priority queue of subtrees: each step pops the D least probable, joins them under a new node whose weight is their sum, and pushes it back; every symbol under a popped node gains that node's digit in front of its codeword."
      state={figure}
      defaultSize="XL"
      hoverReadout={false}
      controls={
        <Player label="merge" value={t} onChange={setStep} count={steps.length} format={(p) => `${p} of ${last}`} />
      }
      equation={<div className="font-prose text-[15px] leading-snug">{explainStep(state, source.names)}</div>}
      readouts={{
        [`step ${t}`]: (
          <>
            <Readout label="queue weights" value={state.queue.map((v) => fmtP(state.nodes[v].weight)).join(' ')} />
            <Readout label="roots left" value={state.queue.length} />
          </>
        ),
        'finished code': (
          <>
            <Readout label={`E[ℓ] (${unit(arity)})`} value={f3(code.expectedLength)} />
            <Readout label="entropy H" value={f3(code.entropy)} />
            <Readout label="E[ℓ] − H" value={f3(code.expectedLength - code.entropy)} />
            <Readout label={`length variance (${TIE_TEXT[ties]})`} value={f3(code.lengthVariance)} />
            <Readout label={`(${TIE_TEXT[OTHER[ties]]})`} value={f3(other.lengthVariance)} />
            <Readout
              label="longest codeword"
              value={`${Math.max(...lengths)} (other rule ${Math.max(...Array.from(toFlat(other.lengths)))})`}
            />
            <Readout
              label={`Kraft sum Σ ${arity}^−ℓ`}
              value={f3(kraft) + (kraft < 1 - 1e-9 ? ' (dummies take the rest)' : '')}
            />
          </>
        ),
      }}
      caption={
        <>
          {source.note}. Step 0 is the queue: one leaf per symbol, least probable on the left. Play or step through the
          merges: the popped subtrees drop under their new parent (drawn heavier), edges carry the digits, and the
          parent takes its sorted place in the queue row. Leaves keep their symbol&apos;s colour in the tree, the table,
          the bit string and the next figure. Click a leaf (or a pin in the table) to pin its codeword path (click again
          or press Escape to unpin); hover a table row or a codeword in the message to highlight its leaf. The ties rule
          changes which equal-weight nodes merge first: with .4 .2 .2 .1 .1, minimum variance gives lengths 2 2 2 3 3
          (variance .16) and merged-first 1 2 3 4 4 (variance 1.36), at the same E[ℓ] = 2.2. Canonical codewords keep
          each length and number the codewords in order of (length, symbol), so a decoder needs only the lengths.
        </>
      }
    >
      <Dashboard>
        <DashboardRow fit="width" minHeight={240}>
          <DashboardCell>
            <Diagram
              spec={spec}
              ariaLabel="Huffman's forest at this step"
              onNodeClick={(id) => {
                const v = Number(id.slice(1))
                if (v < K) togglePin(v)
              }}
              onNodeHover={(id) => {
                const v = id === null ? null : Number(id.slice(1))
                setHover(v !== null && v < K ? v : null)
              }}
            />
          </DashboardCell>
        </DashboardRow>
        <DashboardRow fit="width">
          <DashboardCell>
            <div className="flex flex-col gap-2">
              <Spelled
                state={state}
                names={source.names}
                focus={focus}
                finalLength={focus === null ? 0 : lengths[focus]}
              />
              <CodewordTable
                source={source}
                state={state}
                final={code.codewords}
                canonical={canonical}
                arity={arity}
                focus={focus}
                pinned={pinned}
                mode={mode}
                onHover={setHover}
                onPin={togglePin}
              />
            </div>
          </DashboardCell>
          <DashboardCell>
            <Encoder source={source} code={code} mode={mode} focus={focus} onHover={setHover} />
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}

// ── Figure 2: lengths against the ideal, and block coding ───────────────────────────────────────────────────────

/** Most outcomes a block code is built over (K^n). */
const MAX_OUTCOMES = 1024

function HuffmanLengthsFigure() {
  const { resolved: mode } = useTheme()
  const figure = useFigureState({
    source: sourceRow(26, 'english', 26),
    code: row('2 · code', {
      arity: choice(ARITY_OPTIONS, 2, { label: 'code alphabet D' }),
    }),
  })
  const { name, k, s, q } = figure.source
  const arity = figure.code.arity as number
  const source = useMemo(() => makeSource(name as SourceName, k, s, q), [name, k, s, q])
  const K = source.names.length
  const code = useMemo(() => huffmanCode(source.p, { arity }), [source, arity])
  const bars = useMemo(
    () => ({
      x: source.p.map((_, i) => i),
      length: Array.from(toFlat(code.lengths)),
      ideal: source.p.map((p) => (p > 0 ? -Math.log(p) / Math.log(arity) : NaN)),
      colors: source.p.map((_, i) => symbolColor(mode, i)),
    }),
    [source, code, arity, mode],
  )
  const blocks = useMemo(() => {
    const n: number[] = []
    const perSymbol: number[] = []
    const bound: number[] = []
    for (let b = 1; K ** b <= MAX_OUTCOMES && b <= 10; b++) {
      n.push(b)
      perSymbol.push(huffmanCode(sourceExtension(source.p, b), { arity }).expectedLength / b)
      bound.push(code.entropy + 1 / b)
    }
    return { n, perSymbol, bound }
  }, [source, arity, K, code])
  const key = `${name}/${K}/${arity}`
  const ka = useAxis({ label: 'symbol', categories: source.names, key })
  const ba = useAxis({ label: unit(arity), range: [0, undefined], hold: 'union', key })
  const na = useAxis({ label: 'block length n', integer: true, key })
  const pa = useAxis({ label: `E[ℓₙ]/n (${unit(arity)} per symbol)`, key })
  const span = [-0.5, K - 0.5]
  return (
    <Figure
      title="Codeword lengths against −log p, and block coding"
      purpose="Symbol k ideally takes −log_D pₖ digits; Huffman lengths are whole numbers near it, so H ≤ E[ℓ] < H + 1, and coding blocks of n symbols spreads that last digit over n."
      state={figure}
      defaultSize="XL"
      readouts={
        <>
          <Readout label="entropy H" value={f3(code.entropy)} />
          <Readout label="Huffman E[ℓ]" value={f3(code.expectedLength)} />
          <Readout label="redundancy E[ℓ] − H" value={f3(code.expectedLength - code.entropy)} />
          <Readout
            label={`blocks of ${blocks.n[blocks.n.length - 1]}`}
            value={`${f3(blocks.perSymbol[blocks.perSymbol.length - 1])} per symbol`}
          />
        </>
      }
      caption={
        <>
          {source.note}. Bars are Huffman codeword lengths in each symbol&apos;s colour (symbols past the eighth in
          grey: the palette has eight colours); dots are the ideal {logName(arity)}. With dyadic probabilities every
          ideal length is a whole number and Huffman meets it: E[ℓ] = H. With one likely symbol (skewed, P(a) near 1)
          that symbol still needs a whole digit, so E[ℓ] − H approaches 1. Right: coding the n-th extension (blocks of n
          symbols, K^n outcomes, up to {MAX_OUTCOMES}) brings E[ℓₙ]/n under H + 1/n and towards H.
        </>
      }
    >
      <Dashboard>
        <DashboardRow minHeight={280}>
          <DashboardCell ratio={1.6}>
            <Plot x={ka} y={ba}>
              <Bars name="Huffman length ℓ" x={bars.x} y={bars.length} colors={bars.colors} width={0.7} />
              <Points name={logName(arity)} x={bars.x} y={bars.ideal} emphasis />
              <Curve name={`H = ${f3(code.entropy)}`} x={span} y={[code.entropy, code.entropy]} emphasis dashed />
              <Curve name="H + 1" x={span} y={[code.entropy + 1, code.entropy + 1]} muted dashed />
              <Annotation y={code.expectedLength} text={`E[ℓ] = ${f3(code.expectedLength)}`} dashed={false} />
            </Plot>
          </DashboardCell>
          <DashboardCell>
            <Plot x={na} y={pa}>
              <Curve name="E[ℓₙ]/n" x={blocks.n} y={blocks.perSymbol} emphasis />
              <Points name="E[ℓₙ]/n" x={blocks.n} y={blocks.perSymbol} emphasis />
              <Curve name="H + 1/n" x={blocks.n} y={blocks.bound} muted dashed />
              <Curve
                name={`H = ${f3(code.entropy)}`}
                x={[1, blocks.n[blocks.n.length - 1]]}
                y={[code.entropy, code.entropy]}
                emphasis
                dashed
              />
            </Plot>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}

export function HuffmanShowcase() {
  return (
    <>
      <HuffmanBuildFigure />
      <HuffmanLengthsFigure />
    </>
  )
}
