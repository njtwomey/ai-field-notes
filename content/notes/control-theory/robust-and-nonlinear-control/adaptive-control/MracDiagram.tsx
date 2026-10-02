import { Interactive } from 'aifn-render'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramSpec } from '@/components/diagram/types'

const spec: DiagramSpec = {
  nodes: [
    { id: 'r', x: -0.4, y: 1.5, shape: 'text', w: 0.6, label: '$r$' },
    { id: 'rbus', x: 1, y: 1.5, shape: 'dot' },
    {
      id: 'model',
      x: 4.6,
      y: 0,
      w: 3.8,
      h: 1.1,
      label: 'reference model\n$\\dot{y}_m = -a_m y_m + b_m r$',
      tone: 'neutral',
    },
    { id: 'ctl', x: 3.4, y: 3.2, w: 3, h: 1.1, label: 'controller\n$u = \\theta_1 r - \\theta_2 y$', tone: 0 },
    { id: 'plant', x: 7.4, y: 3.2, w: 2.6, h: 1.1, label: 'plant\n$\\dot{y} = -ay + bu$', tone: 1 },
    { id: 'ytap', x: 9.6, y: 3.2, shape: 'dot' },
    { id: 'y', x: 10.8, y: 3.2, shape: 'text', w: 0.6, label: '$y$' },
    { id: 'sum', x: 9.6, y: 1.6, shape: 'op', label: '$\\Sigma$' },
    { id: 'minus', x: 9.95, y: 1.1, shape: 'text', small: true, w: 0.3, h: 0.3, label: '$-$' },
    { id: 'adapt', x: 6.4, y: 1.6, w: 2.8, h: 0.8, label: 'adaptation law', tone: 2 },
  ],
  edges: [
    { from: 'r', to: 'rbus', arrow: 'none' },
    { from: 'rbus', to: 'model:w', via: [[1, 0]] },
    { from: 'rbus', to: 'ctl:w', via: [[1, 3.2]] },
    { from: 'ctl', to: 'plant', label: '$u$' },
    { from: 'plant', to: 'ytap', arrow: 'none' },
    { from: 'ytap', to: 'y' },
    { from: 'ytap:n', to: 'sum:s' },
    { from: 'model:e', to: 'sum:n', via: [[9.6, 0]], label: '$y_m$' },
    { from: 'sum', to: 'adapt', label: '$e = y - y_m$' },
    {
      from: 'adapt:w',
      to: 'ctl:n',
      via: [[3.4, 1.6]],
      dashed: true,
      tone: 2,
      label: 'adjusts $\\thetavec$',
      labelSide: 'right',
      labelPos: 0.3,
    },
    {
      from: 'ytap:s',
      to: 'ctl:s',
      via: [
        [9.6, 4.5],
        [3.4, 4.5],
      ],
    },
  ],
}

/** Model reference adaptive control: an outer adaptation loop around an ordinary feedback loop. */
export function MracDiagram() {
  return (
    <Interactive
      title="Model reference adaptive control"
      caption="The inner loop is ordinary feedback: the controller with parameters θ drives the plant. The reference model states the response the closed loop should have to the command r. The outer loop compares the plant output y with the model output y_m and the adaptation law adjusts θ from the error e, by the MIT rule or a Lyapunov design."
    >
      <Diagram
        spec={spec}
        ariaLabel="The command r drives a reference model and a controller; the controller drives the plant; the difference between plant output y and model output y_m drives an adaptation law that adjusts the controller parameters"
      />
    </Interactive>
  )
}
