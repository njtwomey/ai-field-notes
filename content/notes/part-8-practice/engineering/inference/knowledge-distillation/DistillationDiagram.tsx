import { Interactive } from 'aifn-render'
import { Diagram } from 'aifn-render'
import type { DiagramNode, DiagramSpec } from 'aifn-render'

const soft = (id: string, x: number, y: number, label: string): DiagramNode => ({
  id,
  x,
  y,
  shape: 'pill',
  w: 2.4,
  h: 0.7,
  label,
  tone: 'neutral',
})

const spec: DiagramSpec = {
  unit: 38,
  nodes: [
    { id: 'x', x: 0, y: 1.6, shape: 'text', w: 0.6, label: '$x$' },
    { id: 'fan', x: 0.8, y: 1.6, shape: 'dot' },
    { id: 'teach', x: 2.8, y: 0, w: 2.4, h: 0.9, label: 'teacher, frozen', tone: 'neutral', dashed: true },
    { id: 'stud', x: 2.8, y: 3.2, w: 2.4, h: 0.9, label: 'student', tone: 0 },
    soft('smt', 6.6, 0, 'softmax$(\\zvec / T)$'),
    { id: 'sfan', x: 4.8, y: 3.2, shape: 'dot' },
    soft('sms', 6.6, 2.1, 'softmax$(\\vvec / T)$'),
    soft('sm1', 6.6, 4.3, 'softmax$(\\vvec)$'),
    {
      id: 'lsoft',
      x: 11.2,
      y: 1,
      w: 3.6,
      h: 1.1,
      label: 'soft term, weight $\\lambda T^2$\ncross-entropy from $\\pvec(T)$ to $\\qvec(T)$',
      tone: 2,
    },
    {
      id: 'lhard',
      x: 11.2,
      y: 4.3,
      w: 3.6,
      h: 1.1,
      label: 'hard term, weight $1 - \\lambda$\ncross-entropy from $y$ to $\\qvec(1)$',
      tone: 1,
    },
    { id: 'y', x: 11.2, y: 5.6, shape: 'text', w: 1.4, label: 'label $y$' },
    { id: 'sum', x: 14.8, y: 2.65, shape: 'op', label: '$+$' },
    { id: 'L', x: 15.8, y: 2.65, shape: 'text', w: 0.6, label: '$\\Lcal$' },
  ],
  edges: [
    { from: 'x', to: 'fan', arrow: 'none' },
    { from: 'fan', to: 'teach:w', via: [[0.8, 0]] },
    { from: 'fan', to: 'stud:w', via: [[0.8, 3.2]] },
    { from: 'teach', to: 'smt', label: '$\\zvec$' },
    { from: 'stud', to: 'sfan', arrow: 'none', label: '$\\vvec$' },
    { from: 'sfan', to: 'sms:w', via: [[4.8, 2.1]] },
    { from: 'sfan', to: 'sm1:w', via: [[4.8, 4.3]] },
    { from: 'smt:e', to: 'lsoft:n', via: [[11.2, 0]], label: '$\\pvec(T)$' },
    { from: 'sms:e', to: 'lsoft:s', via: [[11.2, 2.1]], label: '$\\qvec(T)$' },
    { from: 'sm1', to: 'lhard', label: '$\\qvec(1)$' },
    { from: 'y', to: 'lhard' },
    { from: 'lsoft:e', to: 'sum:n', via: [[14.8, 1]] },
    { from: 'lhard:e', to: 'sum:s', via: [[14.8, 4.3]] },
    { from: 'sum', to: 'L' },
    {
      from: 'L:s',
      to: 'stud:s',
      via: [
        [15.8, 6.5],
        [2.8, 6.5],
      ],
      dashed: true,
      tone: 0,
      label: 'gradients update the student only',
    },
  ],
}

/** The distillation loss: soft targets from the teacher, hard targets from the labels. */
export function DistillationDiagram() {
  return (
    <Interactive
      title="The distillation loss"
      caption="The teacher and the student see the same input. Both sets of logits are softened by the temperature T; the soft term compares the teacher's softened distribution p(T) with the student's q(T). The hard term compares the student's ordinary softmax q(1) with the true label. Only the student's weights are trained, and it runs at T = 1 when deployed."
    >
      <Diagram
        spec={spec}
        ariaLabel="An input goes to a frozen teacher and a student; softened softmaxes of both logits feed the soft loss, the student's plain softmax and the label feed the hard loss; their sum trains the student"
      />
    </Interactive>
  )
}
