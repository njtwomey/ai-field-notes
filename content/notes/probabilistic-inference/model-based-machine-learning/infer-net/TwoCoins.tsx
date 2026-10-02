import { Diagram } from '@/components/diagram/Diagram'
import { factor, link, variable } from '@/components/diagram/components'
import type { DiagramSpec } from '@/components/diagram/types'
import { Interactive } from 'aifn-render'

const line = (a: string, b: string, label?: string) =>
  link(a, b, false, label ? { label, labelSide: 'left', labelRotate: false } : {})

const SPEC: DiagramSpec = {
  unit: 46,
  nodes: [
    factor('p1', 0, 0, '$\\Bern(0.5)$', 'w'),
    factor('p2', 4, 0, '$\\Bern(0.5)$', 'e'),
    variable('c1', 0, 1.4, '$c_1$'),
    variable('c2', 4, 1.4, '$c_2$'),
    factor('and', 2, 2.8, 'AND', 'e'),
    variable('b', 2, 4.2, '$b$', { filled: true }),
  ],
  edges: [
    line('p1', 'c1'),
    line('p2', 'c2'),
    line('c1', 'and', '$\\mu_{\\text{AND} \\to c_1} = (0.5, 1)$'),
    line('c2', 'and'),
    line('and', 'b'),
  ],
}

/** Infer.NET's first tutorial as a factor graph, with the message that answers its backward query. */
export function TwoCoins() {
  return (
    <Interactive
      title="The two-coins model"
      caption="Two fair coins c₁ and c₂ and their AND b, 'both heads'. With b observed false (shaded), the AND factor sends c₁ the probability that b is false for each value of c₁: 0.5 if c₁ is heads (c₂ must be tails) and 1 if it is tails. The posterior is P(c₁ = heads) = 0.5 · 0.5 / (0.5 · 0.5 + 0.5 · 1) = 1/3, which Infer.NET prints as Bernoulli(0.3333)."
    >
      <Diagram
        spec={SPEC}
        ariaLabel="Factor graph: two coin variables with Bernoulli priors joined by an AND factor to the observed both-heads variable"
      />
    </Interactive>
  )
}

const PIPELINE: DiagramSpec = {
  unit: 40,
  nodes: [
    { id: 'model', x: 0, y: 0, w: 3, h: 1, label: 'model definition\nand queries', tone: 0 },
    { id: 'compiler', x: 4, y: 0, w: 2.6, h: 1, label: 'model compiler', tone: 2 },
    { id: 'code', x: 8, y: 0, w: 3, h: 1, label: 'inference source code\n(schedule of messages)', tone: 2 },
    { id: 'algo', x: 8, y: 2.2, w: 3, h: 1, label: 'compiled algorithm', tone: 2 },
    { id: 'data', x: 4, y: 2.2, w: 2.6, h: 1, label: 'observed values', tone: 0 },
    { id: 'out', x: 12, y: 2.2, w: 2.4, h: 1, label: 'posterior\nmarginals', tone: 1 },
  ],
  edges: [
    { from: 'model:e', to: 'compiler:w' },
    { from: 'compiler:e', to: 'code:w', label: 'EP, VMP or Gibbs' },
    { from: 'code:s', to: 'algo:n', label: 'C# compiler' },
    { from: 'data:e', to: 'algo:w' },
    { from: 'algo:e', to: 'out:w' },
  ],
}

/** How Infer.NET turns a model into inference code. */
export function InferPipeline() {
  return (
    <Interactive
      title="How Infer.NET runs a model"
      caption="The model definition and the queries go to the model compiler, which chooses a message-passing schedule for the requested algorithm and writes it out as source code. The compiled algorithm then runs on any observed values without recompiling."
    >
      <Diagram
        spec={PIPELINE}
        ariaLabel="Pipeline: model definition to model compiler to generated source code to compiled algorithm, which takes observed values and returns posterior marginals"
      />
    </Interactive>
  )
}
