import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramSpec } from '@/components/diagram/types'

const spec: DiagramSpec = {
  nodes: [
    {
      id: 'plant',
      x: 5,
      y: 0,
      w: 4.4,
      h: 1.1,
      label: 'plant\n$\\xvec_{k+1} = \\Amat\\xvec_k + \\Bmat\\uvec_k + \\wvec_k$',
      tone: 1,
    },
    { id: 'w', x: 5, y: -1.5, shape: 'text', w: 2.6, label: 'process noise $\\wvec_k$' },
    { id: 'C', x: 8.6, y: 0, w: 0.9, h: 0.7, label: '$\\Cmat$', tone: 1 },
    { id: 'vsum', x: 10.2, y: 0, shape: 'op', label: '$\\Sigma$' },
    { id: 'v', x: 10.2, y: -1.5, shape: 'text', w: 2.6, label: 'sensor noise $\\vvec_k$' },
    { id: 'ytap', x: 11.4, y: 0, shape: 'dot' },
    {
      id: 'kf',
      x: 7.6,
      y: 3,
      w: 3.4,
      h: 1.1,
      label: 'Kalman filter\n$\\hat{\\xvec}_{k \\mid k} = \\expect[\\xvec_k \\mid \\yvec_{0:k}, \\uvec_{0:k-1}]$',
      tone: 2,
    },
    {
      id: 'K',
      x: 2.6,
      y: 3,
      w: 2.4,
      h: 1.1,
      label: 'LQR gain\n$\\uvec_k = -\\Kmat_k\\hat{\\xvec}_{k \\mid k}$',
      tone: 0,
    },
    { id: 'ubus', x: 0.6, y: 1.2, shape: 'dot' },
  ],
  edges: [
    { from: 'w', to: 'plant' },
    { from: 'plant', to: 'C', label: '$\\xvec_k$' },
    { from: 'C', to: 'vsum' },
    { from: 'v', to: 'vsum' },
    { from: 'vsum', to: 'ytap', arrow: 'none' },
    { from: 'ytap:s', to: 'kf:e', via: [[11.4, 3]], label: 'measurement $\\yvec_k$', labelSide: 'right' },
    { from: 'kf', to: 'K', label: '$\\hat{\\xvec}_{k \\mid k}$' },
    { from: 'K:w', to: 'ubus:s', via: [[0.6, 3]], arrow: 'none' },
    { from: 'ubus:n', to: 'plant:w', via: [[0.6, 0]], label: '$\\uvec_k$', labelSide: 'right' },
    { from: 'ubus:e', to: 'kf:n', via: [[7.6, 1.2]], dashed: true },
  ],
  groups: [
    { id: 'est', label: 'estimation', labelAt: 'bottom-right', tone: 2, dashed: true, around: ['kf'], pad: 0.3 },
    { id: 'ctl', label: 'control', labelAt: 'bottom-left', tone: 0, dashed: true, around: ['K'], pad: 0.3 },
  ],
}

/** The LQG controller as a Kalman filter followed by an LQR gain. */
export function LqgDiagram() {
  return (
    <Interactive
      title="The LQG controller"
      caption="The Kalman filter turns the noisy measurements and the past inputs into the state estimate. The LQR gain acts on the estimate as if it were the true state. The filter is designed from the noise covariances W and V alone, and the gain from the cost weights Q and R alone: this is the separation theorem."
    >
      <Diagram
        spec={spec}
        ariaLabel="A plant with process noise produces a state, measured through C with sensor noise; a Kalman filter estimates the state from the measurements and past inputs, and an LQR gain maps the estimate to the next input"
      />
    </Interactive>
  )
}
