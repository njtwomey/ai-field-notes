import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  unit: 40,
  nodes: [
    {
      id: 'u',
      x: 0,
      y: 0,
      w: 3.6,
      h: 1.2,
      label: 'input function $u$\nat sensors $x_1, \\dots, x_m$',
      tone: 'neutral',
    },
    { id: 'y', x: 0, y: 3.4, w: 3.6, h: 1.2, label: 'query point $y$', tone: 'neutral' },
    { id: 'branch', x: 5, y: 0, w: 3.6, h: 1.2, label: 'branch network\n$\\reals^m \\to \\reals^p$', tone: 0 },
    { id: 'trunk', x: 5, y: 3.4, w: 3.6, h: 1.2, label: 'trunk network\n$\\reals^d \\to \\reals^p$', tone: 1 },
    { id: 'b', x: 10, y: 0, w: 3.8, h: 1.2, label: 'coefficients\n$b_1(u), \\dots, b_p(u)$', tone: 0 },
    { id: 't', x: 10, y: 3.4, w: 3.8, h: 1.2, label: 'basis values\n$t_1(y), \\dots, t_p(y)$', tone: 1 },
    {
      id: 'dot',
      x: 15.2,
      y: 1.7,
      w: 4.2,
      h: 1.2,
      shape: 'pill',
      label: '$\\sum_k b_k(u)\\, t_k(y) + b_0$',
      tone: 'neutral',
    },
  ],
  edges: [
    { from: 'u', to: 'branch', label: '$m$ numbers', labelSide: 'left' },
    { from: 'y', to: 'trunk', label: '$d$ numbers', labelSide: 'left' },
    { from: 'branch', to: 'b' },
    { from: 'trunk', to: 't' },
    { from: 'b:e', to: 'dot', via: [[15.2, 0]] },
    { from: 't:e', to: 'dot', via: [[15.2, 3.4]] },
  ],
}

/** The two networks of a DeepONet and the dot product that joins them. */
export function DeepOnetDiagram() {
  return (
    <Figure
      title="Branch and trunk"
      hoverReadout={false}
      caption="A DeepONet has two networks. The branch network reads the input function u through its values at m fixed sensors and returns p coefficients. The trunk network reads one query point y of the output's domain and returns p basis values. The predicted output G(u)(y) is the dot product of the two vectors plus a bias. The branch runs once per input function; the trunk runs once per query point, so a batch of N functions at Q points costs N branch passes, Q trunk passes and one N × p by p × Q matrix product."
    >
      <Diagram
        spec={spec}
        ariaLabel="The input function sampled at m sensors goes through the branch network to p coefficients; the query point goes through the trunk network to p basis values; their dot product plus a bias is the prediction"
      />
    </Figure>
  )
}
