import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramSpec } from '@/components/diagram/types'

const spec: DiagramSpec = {
  nodes: [
    { id: 'K', x: 1.4, y: 1.5, w: 1.1, h: 0.8, label: '$-\\Kmat$', tone: 0 },
    { id: 'ubus', x: 3, y: 1.5, shape: 'dot' },
    {
      id: 'plant',
      x: 6,
      y: -0.9,
      w: 3.6,
      h: 1.1,
      label: 'plant\n$\\dot{\\xvec} = \\Amat\\xvec + \\Bmat\\uvec$',
      tone: 1,
    },
    { id: 'ytap', x: 10.6, y: -0.9, shape: 'dot' },
    { id: 'y', x: 11.8, y: -0.9, shape: 'text', w: 0.6, label: '$\\yvec$' },
    {
      id: 'model',
      x: 6,
      y: 3.2,
      w: 3.6,
      h: 1.1,
      label: 'model copy\n$\\dot{\\hat{\\xvec}} = \\Amat\\hat{\\xvec} + \\Bmat\\uvec + \\Lmat(\\yvec - \\hat{\\yvec})$',
      tone: 2,
    },
    { id: 'xtap', x: 8.9, y: 3.2, shape: 'dot' },
    { id: 'C', x: 10.6, y: 3.2, w: 0.9, h: 0.7, label: '$\\Cmat$', tone: 2 },
    { id: 'sum', x: 10.6, y: 1.6, shape: 'op', label: '$\\Sigma$' },
    { id: 'minus', x: 10.25, y: 2.1, shape: 'text', small: true, w: 0.3, h: 0.3, label: '$-$' },
    { id: 'L', x: 8.4, y: 1.6, w: 0.9, h: 0.7, label: '$\\Lmat$', tone: 2 },
  ],
  edges: [
    { from: 'K', to: 'ubus', arrow: 'none', label: '$\\uvec$' },
    { from: 'ubus', to: 'plant:w', via: [[3, -0.9]] },
    { from: 'ubus', to: 'model:w', via: [[3, 3.2]] },
    { from: 'plant', to: 'ytap', arrow: 'none' },
    { from: 'ytap', to: 'y' },
    { from: 'ytap:s', to: 'sum:n' },
    { from: 'model', to: 'xtap', arrow: 'none' },
    { from: 'xtap', to: 'C' },
    { from: 'C:n', to: 'sum:s', label: '$\\hat{\\yvec}$', labelSide: 'right', labelRotate: false },
    { from: 'sum', to: 'L', label: '$\\yvec - \\hat{\\yvec}$' },
    { from: 'L:w', to: 'model:n', via: [[6, 1.6]] },
    {
      from: 'xtap:s',
      to: 'K:s',
      via: [
        [8.9, 4.6],
        [1.4, 4.6],
      ],
      label: 'estimate $\\hat{\\xvec}$',
    },
  ],
  groups: [{ id: 'obs', label: 'observer', tone: 2, dashed: true, around: ['model', 'C', 'sum', 'L'], pad: 0.3 }],
}

/** An observer-based controller: the plant, the observer that estimates its state, and state feedback. */
export function ObserverDiagram() {
  return (
    <Interactive
      title="Observer-based state feedback"
      caption="The observer runs a copy of the plant model on the same input u. It compares the predicted output ŷ = Cx̂ with the measured output y and feeds the difference back through the observer gain L. The controller applies the state-feedback gain K to the estimate x̂ instead of the unmeasured state x."
    >
      <Diagram
        spec={spec}
        ariaLabel="The input u drives both the plant and a model copy; the output error y minus y-hat is fed back into the model through L, and the estimate x-hat is fed through minus K to form u"
      />
    </Interactive>
  )
}
