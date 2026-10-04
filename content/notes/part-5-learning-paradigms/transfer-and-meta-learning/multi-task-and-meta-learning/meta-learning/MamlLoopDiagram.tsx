import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  unit: 36,
  nodes: [
    { id: 'theta', x: 0, y: 3, shape: 'circle', w: 1.1, h: 1.1, label: '$\\thetavec$', tone: 'ink' },
    {
      id: 'a1',
      x: 3.6,
      y: 1,
      w: 3,
      h: 1,
      label: "inner loop, task 1\n$\\thetavec_1' = \\thetavec - \\alpha \\nabla L^{\\text{tr}}_1$",
      tone: 0,
    },
    {
      id: 'a2',
      x: 3.6,
      y: 3,
      w: 3,
      h: 1,
      label: "inner loop, task 2\n$\\thetavec_2' = \\thetavec - \\alpha \\nabla L^{\\text{tr}}_2$",
      tone: 0,
    },
    {
      id: 'a3',
      x: 3.6,
      y: 5,
      w: 3,
      h: 1,
      label: "inner loop, task $B$\n$\\thetavec_B' = \\thetavec - \\alpha \\nabla L^{\\text{tr}}_B$",
      tone: 0,
    },
    { id: 'q1', x: 7.6, y: 1, w: 2.2, h: 0.9, label: "$L^{\\text{val}}_1(\\thetavec_1')$", tone: 1 },
    { id: 'q2', x: 7.6, y: 3, w: 2.2, h: 0.9, label: "$L^{\\text{val}}_2(\\thetavec_2')$", tone: 1 },
    { id: 'q3', x: 7.6, y: 5, w: 2.2, h: 0.9, label: "$L^{\\text{val}}_B(\\thetavec_B')$", tone: 1 },
    { id: 'sum', x: 10.4, y: 3, shape: 'op', label: '$\\Sigma$' },
    {
      id: 'outer',
      x: 5.2,
      y: 7.1,
      w: 5.4,
      h: 1,
      label:
        "outer loop: $\\thetavec \\leftarrow \\thetavec - \\beta \\nabla_{\\thetavec} \\sum_i L^{\\text{val}}_i(\\thetavec_i')$",
      tone: 2,
    },
  ],
  edges: [
    {
      from: 'theta:e',
      to: 'a1:w',
      via: [
        [1.2, 3],
        [1.2, 1],
      ],
    },
    { from: 'theta', to: 'a2' },
    {
      from: 'theta:e',
      to: 'a3:w',
      via: [
        [1.2, 3],
        [1.2, 5],
      ],
    },
    { from: 'a1', to: 'q1' },
    { from: 'a2', to: 'q2' },
    { from: 'a3', to: 'q3' },
    { from: 'q1:e', to: 'sum:n', via: [[10.4, 1]] },
    { from: 'q2', to: 'sum' },
    { from: 'q3:e', to: 'sum:s', via: [[10.4, 5]] },
    {
      from: 'sum:s',
      to: 'outer:e',
      via: [[10.4, 7.1]],
      dashed: true,
      label: 'backpropagate through the inner step',
      labelSide: 'right',
    },
    { from: 'outer:w', to: 'theta:s', via: [[0, 7.1]], dashed: true },
  ],
}

/** The two loops of MAML. */
export function MamlLoopDiagram() {
  return (
    <Figure
      title="The two loops of MAML"
      caption="For each task in a batch, the inner loop takes one or a few gradient steps from the shared initialisation θ on the task's training (support) examples. The adapted weights are scored on the task's validation (query) examples. The outer loop differentiates the sum of those scores with respect to θ, through the inner steps, and updates θ."
    >
      <Diagram
        spec={spec}
        ariaLabel="Shared initialisation theta feeds inner-loop adaptation for several tasks; their validation losses are summed and the outer loop updates theta"
      />
    </Figure>
  )
}
