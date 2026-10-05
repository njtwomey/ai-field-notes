import { useMemo, useState } from 'react'
import {
  beliefPropagationSteps,
  decodeBeliefs,
  type BeliefPropagationState,
} from 'aifn-compute/inference/message-passing'
import { discreteFactor, discreteFactorGraph, type DiscreteFactorGraph } from 'aifn-compute/inference/model'
import { variableElimination } from 'aifn-compute/inference/exact'
import { trace } from 'aifn-compute/foundation/trace'
import { Figure, ControlGroup, Select, NumberSelector, Player, Plot, Bars, Points, Readout, useAxis } from 'aifn-render'

const NAMES = ['a', 'b', 'c', 'd', 'e']
const CARDS = [2, 2, 2, 2, 2]
const agree = (s: number) => (x: Int32Array) => (x[0] === x[1] ? s : 1)

/** A tree of five binary variables: a – b, b – c, b – d, d – e, with evidence-like unary factors on a and e. */
const TREE: DiscreteFactorGraph = discreteFactorGraph(
  CARDS,
  [
    discreteFactor([0], CARDS, [1, 3], 'φ_a'),
    discreteFactor([0, 1], CARDS, agree(3), 'ψ_{ab}'),
    discreteFactor([1, 2], CARDS, agree(2), 'ψ_{bc}'),
    discreteFactor([1, 3], CARDS, [1, 2, 4, 1], 'ψ_{bd}'),
    discreteFactor([3, 4], CARDS, agree(4), 'ψ_{de}'),
    discreteFactor([4], CARDS, [4, 1], 'φ_e'),
  ],
  NAMES,
)

function describe(s: BeliefPropagationState): string {
  if (!s.updated.length) return 'all messages uniform'
  const u = s.updated[0]
  const e = s.edges[u.edge]
  const f = s.graph.factors[e.factor].name ?? `f${e.factor}`
  const v = NAMES[e.variable]
  return u.to === 'variable' ? `${f} → ${v}` : `${v} → ${f}`
}

const MODES = [
  { value: 'sum', label: 'sum-product (marginal probabilities)' },
  { value: 'max', label: 'max-product (MAP configuration)' },
]

export function TreeBpExplorer() {
  const [mode, setMode] = useState<'sum' | 'max'>('sum')
  const run = useMemo(() => trace(beliefPropagationSteps(TREE, { mode, schedule: 'tree' }), undefined, 100), [mode])
  const [step, setStep] = useState(0)

  const s = run.steps[Math.min(step, run.steps.length - 1)]
  const exact = useMemo(() => NAMES.map((_, v) => variableElimination(TREE, [v], { mode }).marginal.data[1]), [mode])
  const x = NAMES.map((_, i) => i)
  const beliefs = s.beliefs.map((b) => b.data[1])

  const variable = useAxis({ label: 'variable', categories: NAMES })
  const p1 = useAxis({ label: mode === 'sum' ? 'p(x = 1)' : 'max-marginal (x = 1)', range: [0, 1] })
  const done = step >= run.steps.length - 1

  return (
    <Figure
      title="Exact belief propagation messages on a tree"
      purpose="The tree schedule passes messages from leaves to a chosen root and back. After one two-sweep pass every variable belief matches the exact marginal (sum-product) or max-marginal (max-product)."
      defaultSize="L"
      controls={
        <ControlGroup>
          <Select label="Semiring" options={MODES} value={mode} onChange={(m) => setMode(m as 'sum' | 'max')} />
          <NumberSelector
            label="Message step"
            value={step}
            onChange={(st) => setStep(Math.max(0, Math.min(run.steps.length - 1, st)))}
            min={0}
            max={run.steps.length - 1}
            step={1}
            suggestions={[0, 4, 8, run.steps.length - 1]}
          />
          <Player label="Step through messages" value={step} onChange={setStep} count={run.steps.length} />
        </ControlGroup>
      }
      readouts={
        <>
          <Readout label="Active message" value={describe(s)} />
          <Readout
            label="Progress"
            value={`${Math.min(step, run.steps.length - 1)} of ${run.steps.length - 1} messages`}
          />
          {mode === 'max' && done && (
            <Readout label="MAP (a b c d e)" value={Array.from(decodeBeliefs(s).data).join(' ')} />
          )}
        </>
      }
      caption="Bars show the exact marginals computed by variable elimination; points track the evolving local beliefs b(x = 1) at each step of message passing. Once incoming messages from all neighbouring factors arrive, the belief reaches the exact bar. In max-product, the beliefs converge to max-marginals whose argmax decodes the global MAP state."
    >
      <Plot x={variable} y={p1}>
        <Bars name={mode === 'sum' ? 'exact p(x = 1)' : 'exact max-marginal'} x={x} y={exact} slot={0} />
        <Points name="current belief b(x = 1)" x={x} y={beliefs} slot={1} />
      </Plot>
    </Figure>
  )
}
