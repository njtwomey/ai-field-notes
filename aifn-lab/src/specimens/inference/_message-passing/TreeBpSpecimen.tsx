import { beliefPropagationSteps, decodeBeliefs, type BeliefPropagationState } from 'aifn/inference/message-passing'
import { discreteFactor, discreteFactorGraph, type DiscreteFactorGraph } from 'aifn/inference/model'
import { variableElimination } from 'aifn/inference/exact'
import { trace } from 'aifn/foundation/trace'
import { useMemo, useState } from 'react'
import { Player, Select } from '@lab/controls'
import { Columns, Figure } from '@lab/layout'
import { FactorGraphView } from '@lab/views'
import { ChartSize, Readout, XYChart, type XYSeries } from '@lab/viz'

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
const AT: Record<number, [number, number]> = { 0: [0, 0], 1: [3, 0], 2: [6, -1.6], 3: [6, 1.6], 4: [9, 1.6] }

function describe(s: BeliefPropagationState): string {
  if (!s.updated.length) return 'all messages uniform'
  const u = s.updated[0]
  const e = s.edges[u.edge]
  const f = s.graph.factors[e.factor].name ?? `f${e.factor}`
  const v = NAMES[e.variable]
  return u.to === 'variable' ? `${f} → ${v}` : `${v} → ${f}`
}

/** Sum-product and max-product on a tree, one message per step, against the exact (max-)marginals. */
export function TreeBpSpecimen() {
  const [mode, setMode] = useState<'sum' | 'max'>('sum')
  const run = useMemo(() => trace(beliefPropagationSteps(TREE, { mode, schedule: 'tree' }), undefined, 100), [mode])
  const [step, setStep] = useState(7)
  const s = run.steps[Math.min(step, run.steps.length - 1)]
  const exact = useMemo(() => NAMES.map((_, v) => variableElimination(TREE, [v], { mode }).marginal.data[1]), [mode])
  const x = NAMES.map((_, i) => i)
  const series: XYSeries[] = [
    { name: mode === 'sum' ? 'exact p(x = 1)' : 'exact max-marginal', type: 'bar', x, y: exact, slot: 0 },
    { name: 'belief', type: 'scatter', x, y: s.beliefs.map((b) => b.data[1]), slot: 1 },
  ]
  const done = step >= run.steps.length - 1
  return (
    <Figure
      title="Messages on a tree, one at a time"
      description="The tree schedule passes messages from the leaves to a root and back; after one pass every belief is exact."
      defaultSize="L"
      hoverReadout={false}
      controls={
        <>
          <Select
            label="semiring"
            value={mode}
            onChange={(m) => (setMode(m), setStep(7))}
            options={[
              { value: 'sum', label: 'sum-product (marginals)' },
              { value: 'max', label: 'max-product (MAP)' },
            ]}
          />
          <div className="col-span-full">
            <Player label="message" value={step} onChange={setStep} count={run.steps.length} defaultSpeed={2} />
          </div>
        </>
      }
      readouts={
        <>
          <Readout label="this step" value={describe(s)} />
          <Readout label="messages sent" value={`${Math.min(step, run.steps.length - 1)} of ${run.steps.length - 1}`} />
          {mode === 'max' && done && (
            <Readout label="MAP (a b c d e)" value={Array.from(decodeBeliefs(s).data).join(' ')} />
          )}
        </>
      }
      caption="Each chip is the message just sent, as μ(x = 1), with its arrow showing the direction; a variable's shade and note are its current belief p(x = 1). Leaves send first. A belief is exact once every message into it has arrived: on the chart the points reach the bars when the pass back from the root is done. Max-product replaces the sums by maxima; its beliefs are max-marginals, and their argmax is the MAP assignment."
    >
      <Columns
        widths={[3, 2]}
        panels={[
          {
            title: 'factor graph: the message just sent, and beliefs',
            body: (
              <FactorGraphView graph={TREE} positions={AT} state={s} height="fill" ariaLabel="A tree factor graph" />
            ),
          },
          {
            title: mode === 'sum' ? 'beliefs against exact marginals' : 'beliefs against exact max-marginals',
            body: (
              <ChartSize scale={0.92}>
                <XYChart
                  series={series}
                  xLabel="variable (0 = a … 4 = e)"
                  yLabel="p(x = 1)"
                  yRange={[0, 1]}
                  integerX
                  formatX={(v) => NAMES[v] ?? ''}
                />
              </ChartSize>
            ),
          },
        ]}
      />
    </Figure>
  )
}
