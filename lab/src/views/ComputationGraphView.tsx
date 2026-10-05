import type { Graph, GraphNode } from 'aifn-compute/foundation/autodiff'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { useContext, useMemo, useState, type ReactNode } from 'react'
import { Player } from 'aifn-render/controls'
import { Diagram, layeredLayout, type DiagramEdge, type DiagramSpec, type ElementState } from 'aifn-render/diagram'
import { PanelSlot, Tex } from 'aifn-render/layout'
import { FrameContext, Readout } from 'aifn-render/viz'
import { formatValue } from './format'
import { registerKind, registerView } from './registry'

export type ComputationGraphPanelProps = {
  /** A graph from `traceGraph` in `aifn/foundation/autodiff`. */
  graph: Graph
  /** The function being differentiated, as TeX, set large above the diagram, e.g. `L = \log(1 + e^{wx + b})`. */
  expression?: string
  /** TeX symbol of the output node (default `f`). */
  outputSymbol?: string
  /**
   * TeX symbols by node id, overriding the defaults: inputs are named from their path in x (`x.w` → `w`, `x[1]` →
   * `x_{1}`), intermediates v₁, v₂, … in evaluation order, the output `outputSymbol`.
   */
  symbols?: Record<number, string>
  /** Controlled step (0 … count − 1; see the player); uncontrolled when omitted. */
  step?: number
  onStep?: (step: number) => void
  /**
   * Open at the last step (everything propagated), saying why the finished pass is the point. Without it the player
   * opens at the first step, as every walk-through does.
   */
  startReason?: string
}

// ── Symbols and local rules ──────────────────────────────────────────────────────────────────────────────────────────

/** Significant figures of the numbers set in the diagram and the equations. */
const DIGITS = 4

function num(v: number): string {
  if (!Number.isFinite(v)) return String(v)
  const s = String(Number(v.toPrecision(DIGITS)))
  return s.includes('e') ? formatValue(v) : s
}

/** A value as TeX: a number, or a tensor as its shape and first entries. */
function tex(v: number | Tensor | null): string {
  if (v === null) return '\\cdot'
  if (typeof v === 'number') return num(v)
  const flat = toFlat(v)
  if (v.shape.length === 0) return num(flat[0])
  const head = flat.slice(0, 3).map(num).join(', ')
  return `[${head}${flat.length > 3 ? ', \\dots' : ''}]`
}

/** A value set inside an expression: negatives in parentheses. */
const arg = (v: number | Tensor | null) => {
  const t = tex(v)
  return t.startsWith('-') ? `(${t})` : t
}

/** An argument without its parentheses, e.g. in an exponent. */
const bare = (t: string) => t.replace(/^\(([^()]*)\)$/, '$1')

/** A readable TeX symbol for an input's path in x. */
function inputSymbol(label: string | undefined, i: number): string {
  const path = (label ?? `x[${i}]`).replace(/^x\.?/, '') || 'x'
  const index = /^\[(\d+)\]$/.exec(path)
  if (index) return `x_{${index[1]}}`
  const name =
    path
      .split(/[.[\]]+/)
      .filter(Boolean)
      .at(-1) ?? 'x'
  const digits = /^([a-zA-Z])(\d+)$/.exec(name)
  if (digits) return `${digits[1]}_{${digits[2]}}`
  return name.length === 1 ? name : `\\mathrm{${name}}`
}

/** The adjoint symbol v̄ of a symbol: `v_{3}` → `\bar{v}_{3}`. */
function bar(symbol: string): string {
  const m = /^(\\[a-zA-Z]+|[a-zA-Z])(_.*)?$/.exec(symbol)
  return m ? `\\bar{${m[1]}}${m[2] ?? ''}` : `\\overline{${symbol}}`
}

/**
 * How each primitive is written: its factor label, its expression in its arguments, and its local partial with
 * respect to argument i (in the arguments and the output). Arguments are TeX (a symbol or a parenthesised number).
 */
type Rule = { label: string; expr: (a: string[]) => string; partial?: (i: number, a: string[], out: string) => string }
const RULES: Record<string, Rule> = {
  add: { label: '+', expr: (a) => `${a[0]} + ${a[1]}`, partial: () => '1' },
  sub: { label: '-', expr: (a) => `${a[0]} - ${a[1]}`, partial: (i) => (i === 0 ? '1' : '-1') },
  mul: { label: '\\times', expr: (a) => `${a[0]} \\cdot ${a[1]}`, partial: (i, a) => a[1 - i] },
  div: {
    label: '\\div',
    expr: (a) => `\\frac{${a[0]}}{${a[1]}}`,
    partial: (i, a) => (i === 0 ? `\\frac{1}{${a[1]}}` : `-\\frac{${a[0]}}{${a[1]}^2}`),
  },
  neg: { label: '-', expr: (a) => `-${a[0]}`, partial: () => '-1' },
  exp: { label: '\\exp', expr: (a) => `e^{${bare(a[0])}}`, partial: (_i, a) => `e^{${bare(a[0])}}` },
  log: { label: '\\log', expr: (a) => `\\log ${a[0]}`, partial: (_i, a) => `\\frac{1}{${a[0]}}` },
  log1p: { label: '\\log(1+\\cdot)', expr: (a) => `\\log(1 + ${a[0]})`, partial: (_i, a) => `\\frac{1}{1 + ${a[0]}}` },
  square: { label: '(\\cdot)^2', expr: (a) => `${a[0]}^2`, partial: (_i, a) => `2 \\cdot ${a[0]}` },
  sqrt: { label: '\\sqrt{\\cdot}', expr: (a) => `\\sqrt{${a[0]}}`, partial: (_i, a) => `\\frac{1}{2\\sqrt{${a[0]}}}` },
  sin: { label: '\\sin', expr: (a) => `\\sin ${a[0]}`, partial: (_i, a) => `\\cos ${a[0]}` },
  cos: { label: '\\cos', expr: (a) => `\\cos ${a[0]}`, partial: (_i, a) => `-\\sin ${a[0]}` },
  tanh: { label: '\\tanh', expr: (a) => `\\tanh ${a[0]}`, partial: (_i, _a, out) => `1 - ${out}^2` },
  softplus: {
    label: '\\operatorname{softplus}',
    expr: (a) => `\\operatorname{softplus}(${a[0]})`,
    partial: (_i, a) => `\\sigma(${a[0]})`,
  },
  sigmoid: {
    label: '\\sigma',
    expr: (a) => `\\sigma(${a[0]})`,
    partial: (_i, _a, out) => `${out}(1 - ${out})`,
  },
}

function ruleOf(op: string): Rule {
  return (
    RULES[op] ?? {
      label: `\\operatorname{${op}}`,
      expr: (a) => `\\operatorname{${op}}(${a.join(', ')})`,
    }
  )
}

/** The factor's label: the primitive, with any constant argument written in (`+\,1`, `\times(-1)`). */
function factorLabel(node: GraphNode): string {
  const rule = ruleOf(node.op)
  const constants = node.inputs.map((input) => ('constant' in input ? arg(input.constant) : null))
  if (!constants.some((c) => c !== null)) return rule.label
  if (node.inputs.length === 2)
    return constants[0] !== null ? `${constants[0]}\\,${rule.label}` : `${rule.label}\\,${constants[1]}`
  return rule.label
}

// ── Steps ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The step sequence: 0 sets the inputs; 1 … F compute the F operations forward; F + 1 seeds the output's adjoint with
 * 1; F + 2 … 2F + 1 pull the adjoint back through the operations in reverse order.
 */
type Phase =
  | { kind: 'inputs' }
  | { kind: 'forward'; op: number; index: number }
  | { kind: 'seed' }
  | { kind: 'backward'; op: number; index: number }

function phaseOf(step: number, ops: number[]): Phase {
  const F = ops.length
  if (step <= 0) return { kind: 'inputs' }
  if (step <= F) return { kind: 'forward', op: ops[step - 1], index: step - 1 }
  if (step === F + 1) return { kind: 'seed' }
  const index = step - F - 2
  return { kind: 'backward', op: ops[F - 1 - index], index }
}

/**
 * Backpropagation drawn as a factor graph and stepped node by node. Variables (circles) are the inputs, the
 * intermediate values and the output; factors (small squares) are the primitive operations between them. The forward
 * band, on top, computes values left to right; the backward band below mirrors it column by column and pulls the
 * adjoints v̄ = ∂output/∂v back right to left, each edge carrying its local partial. The function is set large above
 * the diagram and the current step's equation, with numbers substituted, below it.
 */
export function ComputationGraphPanel({
  graph,
  expression,
  outputSymbol = 'f',
  symbols: given,
  step,
  onStep,
  startReason,
}: ComputationGraphPanelProps) {
  const ops = useMemo(() => graph.nodes.filter((n) => n.op !== 'input').map((n) => n.id), [graph])
  const count = 2 * ops.length + 2
  const [own, setOwn] = useState(startReason ? count - 1 : 0)
  const current = Math.min(Math.max(step ?? own, 0), count - 1)
  const setStep = (s: number) => (onStep ? onStep(s) : setOwn(s))

  const symbols = useMemo(() => {
    const out: Record<number, string> = {}
    let k = 0
    for (const node of graph.nodes) {
      if (node.op === 'input') out[node.id] = inputSymbol(node.label, graph.inputs.indexOf(node.id))
      else out[node.id] = node.id === graph.output ? outputSymbol : `v_{${++k}}`
    }
    return { ...out, ...given }
  }, [graph, outputSymbol, given])

  const phase = phaseOf(current, ops)
  const spec = useMemo(() => bands(graph, ops, symbols, current), [graph, ops, symbols, current])
  const equation = stepEquation(graph, symbols, phase, outputSymbol)
  const lines = Math.max(1, ...ops.map((id) => graph.nodes[id].inputs.filter((i) => 'node' in i).length))
  const describe =
    phase.kind === 'inputs'
      ? 'inputs set'
      : phase.kind === 'forward'
        ? `forward ${phase.index + 1} of ${ops.length}: ${graph.nodes[phase.op].op}`
        : phase.kind === 'seed'
          ? 'backward: seed the output adjoint'
          : `backward ${phase.index + 1} of ${ops.length}: through ${graph.nodes[phase.op].op}`

  return (
    <>
      <PanelSlot slot="controls">
        <>
          <div className="col-span-full">
            <Player
              label="step (forward, then backward)"
              value={current}
              onChange={setStep}
              count={count}
              startReason={startReason}
            />
          </div>
        </>
      </PanelSlot>
      <PanelSlot slot="readouts">
        <Readout label="step" value={describe} />
      </PanelSlot>
      <Body top={expression && <Tex display>{expression}</Tex>} bottom={<Tex display>{equation}</Tex>} lines={lines}>
        <Diagram
          spec={spec}
          height="fill"
          ariaLabel="Factor graph of the computation: the forward pass above, the backward pass below"
        />
      </Body>
    </>
  )
}

registerKind(
  'computation-graph',
  (o) =>
    typeof o === 'object' &&
    o !== null &&
    Array.isArray((o as Graph).nodes) &&
    Array.isArray((o as Graph).inputs) &&
    typeof (o as Graph).output === 'number',
)
registerView<Graph>({
  key: 'computation-graph/backprop',
  kind: 'computation-graph',
  description:
    'A traced computation as a factor graph: the forward pass assigns values left to right, the backward pass carries adjoints back, step by step.',
  title: () => 'Computation graph',
  render: (g) => <ComputationGraphPanel graph={g} />,
})

/** The chart area: the function on top, the diagram filling the middle, the step's equation below. */
function Body({
  top,
  bottom,
  lines,
  children,
}: {
  top: ReactNode
  bottom: ReactNode
  lines: number
  children: ReactNode
}) {
  const { height } = useContext(FrameContext)
  return (
    <div className="flex flex-col gap-2" style={{ height }}>
      {top && <div className="shrink-0 overflow-x-auto text-2xl">{top}</div>}
      <div className="min-h-0 flex-1">{children}</div>
      {/* Room for the longest step, so the diagram does not jump as the reader steps. */}
      <div
        className="flex shrink-0 items-center justify-center overflow-x-auto text-xl"
        style={{ minHeight: `${lines * 2.4 + 0.6}em` }}
      >
        {bottom}
      </div>
    </div>
  )
}

/** The current step as TeX, with numbers substituted. */
function stepEquation(graph: Graph, symbols: Record<number, string>, phase: Phase, outputSymbol: string): string {
  const argsOf = (node: GraphNode, numeric: boolean) =>
    node.inputs.map((input) =>
      'constant' in input ? arg(input.constant) : numeric ? arg(graph.nodes[input.node].value) : symbols[input.node],
    )
  if (phase.kind === 'inputs')
    return graph.inputs.map((id) => `${symbols[id]} = ${tex(graph.nodes[id].value)}`).join(',\\quad ')
  if (phase.kind === 'seed') {
    const L = outputSymbol
    return `${bar(L)} = \\frac{\\partial ${L}}{\\partial ${L}} = 1`
  }
  const node = graph.nodes[phase.op]
  const rule = ruleOf(node.op)
  const out = symbols[node.id]
  if (phase.kind === 'forward')
    return `${out} = ${rule.expr(argsOf(node, false))} = ${rule.expr(argsOf(node, true))} = ${tex(node.value)}`
  // Backward: one chain-rule update per input edge.
  const rows = node.inputs.flatMap((input, i) => {
    if (!('node' in input)) return []
    const v = symbols[input.node]
    const local = `\\frac{\\partial ${out}}{\\partial ${v}}`
    const symbolic = rule.partial ? rule.partial(i, argsOf(node, false), out) : null
    const numeric = rule.partial ? rule.partial(i, argsOf(node, true), arg(node.value)) : null
    const chain: string[] = []
    if (symbolic !== null) chain.push(`${bar(out)} \\cdot ${wrap(symbolic)}`)
    if (numeric !== null) chain.push(`${arg(node.adjoint)} \\cdot ${wrap(numeric)}`)
    chain.push(`${arg(node.adjoint)} \\cdot ${arg(input.partial)}`, tex(input.message))
    // Drop steps that repeat the one before (e.g. when the partial is a constant).
    const steps = chain.filter((p, k) => k === 0 || p !== chain[k - 1])
    return [`${bar(v)} \\mathrel{+}= ${bar(out)} \\cdot ${local} &= ${steps.join(' = ')}`]
  })
  return rows.length ? `\\begin{aligned}${rows.join(' \\\\ ')}\\end{aligned}` : ''
}

/** Parenthesises a compound partial so it reads as one factor. */
function wrap(t: string): string {
  const atomic = [/^[\w{}\\^.]+$/, /^\\frac/, /^\\?[a-zA-Z]+\(.*\)$/, /^\([^()]*\)$/, /^e\^\{[^}]*\}$/]
  return atomic.some((r) => r.test(t)) ? t : `(${t})`
}

// ── The two bands ────────────────────────────────────────────────────────────────────────────────────────────────────

const VAR = (id: number) => `v${id}`
const FAC = (id: number) => `f${id}`
const BACK = (id: string) => `b${id}`

/**
 * The diagram spec for one step: the forward graph laid out in layers (variables and factors alternate across the
 * columns), and the same graph again below it for the backward pass, aligned column by column, arrows reversed.
 */
function bands(graph: Graph, ops: number[], symbols: Record<number, string>, step: number): DiagramSpec {
  const phase = phaseOf(step, ops)
  const opIndex = new Map(ops.map((id, i) => [id, i]))
  const F = ops.length

  // Forward states: an operation is done once computed, active while being computed.
  const forwardState = (id: number): ElementState => {
    if (graph.nodes[id].op === 'input') return phase.kind === 'inputs' ? 'active' : 'done'
    const i = opIndex.get(id)!
    const reached = phase.kind === 'inputs' ? -1 : phase.kind === 'forward' ? phase.index : F
    return i < reached ? 'done' : i === reached ? 'active' : 'idle'
  }
  // Backward: operations are processed from the last; `pulled` counts those finished, including the current one.
  const pulled = phase.kind === 'backward' ? phase.index + 1 : 0
  const processed = (id: number) => {
    const i = opIndex.get(id)
    return i !== undefined && F - 1 - i < pulled
  }
  const backwardFactorState = (id: number): ElementState =>
    phase.kind === 'backward' && phase.op === id ? 'active' : processed(id) ? 'done' : 'idle'
  // The adjoint so far: the seed at the output, plus every message from the operations already processed.
  const adjointSoFar = new Map<number, number | Tensor>()
  if (phase.kind === 'seed' || phase.kind === 'backward') {
    const out = graph.nodes[graph.output]
    if (out?.adjoint !== undefined && out.adjoint !== null) adjointSoFar.set(graph.output, out.adjoint)
  }
  const receiving = new Set<number>()
  for (const node of graph.nodes) {
    if (!processed(node.id)) continue
    for (const input of node.inputs) {
      if (!('node' in input) || input.message === null) continue
      const prev = adjointSoFar.get(input.node)
      const m = input.message
      adjointSoFar.set(
        input.node,
        prev === undefined ? m : typeof prev === 'number' && typeof m === 'number' ? prev + m : m,
      )
      if (phase.kind === 'backward' && phase.op === node.id) receiving.add(input.node)
    }
  }
  const backwardVarState = (id: number): ElementState =>
    receiving.has(id) || (phase.kind === 'seed' && id === graph.output)
      ? 'active'
      : adjointSoFar.has(id)
        ? 'done'
        : 'idle'

  // Forward band: laid out once from the graph's structure.
  const forward = layeredLayout({
    layout: 'layered',
    layered: { layerGap: 1.9, nodeGap: 1.8 },
    nodes: [
      ...graph.nodes.map((n) => ({ id: VAR(n.id), shape: 'circle' as const })),
      ...ops.map((id) => ({ id: FAC(id), shape: 'factor' as const })),
    ],
    edges: ops.flatMap((id) => [
      ...graph.nodes[id].inputs.flatMap((input) =>
        'node' in input ? [{ from: VAR(input.node), to: FAC(id), route: 'straight' as const }] : [],
      ),
      { from: FAC(id), to: VAR(id), route: 'straight' as const },
    ]),
  })
  const at = new Map(forward.nodes.map((n) => [n.id, n]))
  const ys = forward.nodes.map((n) => n.y)
  const span = Math.max(...ys) - Math.min(...ys)
  // Room between the bands for both groups' padding, the backward group's label and the circles.
  const offset = span + 3.6
  const via = new Map((forward.edges ?? []).map((e) => [`${e.from}>${e.to}`, e.via]))

  const nodes: DiagramSpec['nodes'] = []
  const edges: DiagramEdge[] = []
  const valueNote = (id: number) => (forwardState(id) === 'idle' ? '$\\cdot$' : `$${tex(graph.nodes[id].value)}$`)
  const adjointNote = (id: number) => {
    const a = adjointSoFar.get(id)
    return a === undefined ? '$\\cdot$' : `$${tex(a)}$`
  }
  for (const node of graph.nodes) {
    const p = at.get(VAR(node.id))!
    const isInput = node.op === 'input'
    const common = { shape: 'circle' as const, tone: isInput ? 1 : 0, filled: node.id === graph.output }
    nodes.push({
      ...common,
      id: VAR(node.id),
      x: p.x,
      y: p.y,
      label: `$${symbols[node.id]}$`,
      state: forwardState(node.id),
      notes: { n: valueNote(node.id) },
    })
    nodes.push({
      ...common,
      id: BACK(VAR(node.id)),
      x: p.x,
      y: p.y + offset,
      label: `$${bar(symbols[node.id])}$`,
      tone: 2,
      state: backwardVarState(node.id),
      notes: { s: adjointNote(node.id) },
    })
  }
  for (const id of ops) {
    const p = at.get(FAC(id))!
    const node = graph.nodes[id]
    const label = `$${factorLabel(node)}$`
    nodes.push({ id: FAC(id), shape: 'factor', x: p.x, y: p.y, label, labelSide: 'n', state: forwardState(id) })
    nodes.push({
      id: BACK(FAC(id)),
      shape: 'factor',
      x: p.x,
      y: p.y + offset,
      label,
      labelSide: 's',
      tone: 2,
      state: backwardFactorState(id),
    })
    const fState = forwardState(id)
    const bState = backwardFactorState(id)
    const shift = (pts: [number, number][] | undefined) => pts?.map(([x, y]): [number, number] => [x, y + offset])
    for (const input of node.inputs) {
      if (!('node' in input)) continue
      const key = `${VAR(input.node)}>${FAC(id)}`
      edges.push({
        from: VAR(input.node),
        to: FAC(id),
        route: 'straight',
        arrow: 'mid',
        via: via.get(key),
        state: fState,
      })
      edges.push({
        from: BACK(VAR(input.node)),
        to: BACK(FAC(id)),
        route: 'straight',
        arrow: 'mid',
        reverse: true,
        tone: 2,
        via: shift(via.get(key)),
        state: bState,
        // The local partial the adjoint is multiplied by on its way to this input.
        note:
          input.partial === null || bState === 'idle'
            ? undefined
            : { text: `$\\times\\,${arg(input.partial)}$`, at: 0.45 },
      })
    }
    edges.push({ from: FAC(id), to: VAR(id), route: 'straight', arrow: 'mid', state: fState })
    edges.push({
      from: BACK(FAC(id)),
      to: BACK(VAR(id)),
      route: 'straight',
      arrow: 'mid',
      reverse: true,
      tone: 2,
      state: bState,
    })
  }
  return {
    nodes,
    edges,
    groups: [
      {
        id: 'forward',
        label: 'forward: values',
        around: forward.nodes.map((n) => n.id),
        pad: 0.8,
        tone: 0,
        dashed: true,
      },
      {
        id: 'backward',
        label: `backward: adjoints $\\bar v = \\partial ${symbols[graph.output] ?? 'f'} / \\partial v$`,
        around: forward.nodes.map((n) => BACK(n.id)),
        pad: 0.8,
        tone: 2,
        dashed: true,
      },
    ],
  }
}
