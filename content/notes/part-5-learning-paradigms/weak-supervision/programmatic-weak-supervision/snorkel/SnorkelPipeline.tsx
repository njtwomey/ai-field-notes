import { Diagram, Figure } from 'aifn-render'
import type { DiagramNode, DiagramSpec } from 'aifn-render'

const lf = (id: string, y: number, label: string): DiagramNode => ({ id, x: 2.8, y, w: 1.9, h: 0.6, label, tone: 0 })

const spec: DiagramSpec = {
  unit: 38,
  spread: [1.35, 1.1],
  nodes: [
    { id: 'data', x: 0, y: 1.5, w: 1.8, h: 1, label: 'unlabelled\ncandidates', tone: 'neutral' },
    { id: 'fan', x: 1.35, y: 1.5, shape: 'dot' },
    lf('l1', 0.2, 'pattern LF $\\lambda_1$'),
    lf('l2', 1.1, 'KB lookup $\\lambda_2$'),
    { id: 'vd', x: 2.8, y: 1.95, shape: 'text', w: 0.6, h: 0.4, label: '$\\vdots$' },
    lf('lm', 2.8, 'weak model $\\lambda_m$'),
    {
      id: 'L',
      x: 6,
      y: 1.5,
      w: 2.2,
      h: 1,
      label: 'label matrix $\\Lambda$\n$n \\times m$, with abstains',
      tone: 'neutral',
    },
    { id: 'gm', x: 9.2, y: 1.5, w: 2.4, h: 1, label: 'label model\n$p_w(\\Lambda, Y)$', tone: 2 },
    { id: 'end', x: 12.6, y: 1.5, w: 2.4, h: 1, label: 'end model\ntrained on $\\tilde Y$', tone: 1 },
    { id: 'out', x: 12.6, y: 3.4, shape: 'text', w: 3.4, label: 'classifier on features of $\\xvec$' },
  ],
  edges: [
    { from: 'data', to: 'fan', arrow: 'none' },
    { from: 'fan', to: 'l1:w', via: [[1.35, 0.2]] },
    { from: 'fan', to: 'l2:w', via: [[1.35, 1.1]] },
    { from: 'fan', to: 'lm:w', via: [[1.35, 2.8]] },
    { from: 'l1:e', to: 'L:n', via: [[6, 0.2]] },
    {
      from: 'l2:e',
      to: 'L:w',
      via: [
        [4.4, 1.1],
        [4.4, 1.5],
      ],
    },
    { from: 'lm:e', to: 'L:s', via: [[6, 2.8]] },
    { from: 'L', to: 'gm' },
    { from: 'gm', to: 'end', label: '$\\tilde Y$' },
    { from: 'end', to: 'out' },
  ],
  groups: [
    { id: 's1', label: '1. write LFs', tone: 0, dashed: true, around: ['l1', 'lm'], pad: 0.3 },
    { id: 's2', label: '2. model the LFs', tone: 2, dashed: true, around: ['gm'], pad: 0.3 },
    { id: 's3', label: '3. train', tone: 1, dashed: true, around: ['end'], pad: 0.3 },
  ],
}

/** The three stages of Snorkel. */
export function SnorkelPipeline() {
  return (
    <Figure
      title="The Snorkel pipeline"
      caption="Labelling functions vote or abstain on every unlabelled candidate, filling the label matrix Λ. The label model learns each function's accuracy and the correlations between functions from their agreements alone, and outputs probabilistic labels Ỹ. A discriminative end model trained on Ỹ uses the full features, so it can label points on which every function abstains."
    >
      <Diagram
        spec={spec}
        ariaLabel="Unlabelled candidates pass through labelling functions to form a label matrix; a label model turns the matrix into probabilistic labels; an end model is trained on those labels"
      />
    </Figure>
  )
}
