import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  unit: 40,
  nodes: [
    {
      id: 'req',
      x: 0,
      y: 1.6,
      w: 3.2,
      h: 1.6,
      label: 'request\nstate + questions\n(type, instructions, criteria)',
      tone: 'neutral',
    },
    { id: 'read', x: 4.4, y: 1.6, w: 3, h: 1.3, label: 'read once\ninstructions, state, schema', tone: 0 },
    { id: 'q1', x: 8.6, y: 0, w: 2.8, h: 0.9, label: 'ending: question 1', tone: 1 },
    { id: 'q2', x: 8.6, y: 1.6, w: 2.8, h: 0.9, label: 'ending: question 2', tone: 1 },
    { id: 'q3', x: 8.6, y: 3.2, w: 2.8, h: 0.9, label: 'ending: question Q', tone: 1 },
    { id: 'z', x: 12.6, y: 1.6, w: 2.8, h: 1.3, label: 'code logits\n$\\zvec_{\\Ccal}$ per question', tone: 2 },
    {
      id: 'out',
      x: 16.4,
      y: 1.6,
      w: 2.8,
      h: 1.6,
      label: 'typed answers\n$\\operatorname{softmax}(\\zvec_{\\Ccal}/T)$\nno text written',
      tone: 'neutral',
    },
  ],
  edges: [
    { from: 'req', to: 'read' },
    { from: 'read:e', to: 'q1:w' },
    { from: 'read:e', to: 'q2:w', label: 'cache' },
    { from: 'read:e', to: 'q3:w' },
    { from: 'q1:e', to: 'z:w' },
    { from: 'q2:e', to: 'z:w' },
    { from: 'q3:e', to: 'z:w' },
    { from: 'z', to: 'out' },
  ],
}

/** The path of one request through a decision model. */
export function DecisionFlow() {
  return (
    <Figure
      title="One request through a decision model"
      caption="The model reads the instructions, the state and the schema of every question once and keeps that internal state. Each question's short ending runs against it in one batch; the network's scores for the answer codes at the last position give a probability for every allowed answer. Nothing is generated, so latency depends on the input length alone."
    >
      <Diagram
        spec={spec}
        ariaLabel="A request is read once; per-question endings branch from the cached read; each ending gives code logits; a softmax over the codes gives typed answers"
      />
    </Figure>
  )
}
