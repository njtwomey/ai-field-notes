import { Figure } from 'aifn-render'
import { Diagram } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  unit: 40,
  nodes: [
    { id: 'data', x: 0, y: 0, w: 2, h: 0.9, label: '$N$ examples', tone: 'neutral' },
    {
      id: 'lot',
      x: 3.2,
      y: 0,
      w: 2.8,
      h: 0.9,
      label: 'lot $\\Lcal_t$: keep each\nwith probability $q$',
      tone: 'neutral',
    },
    { id: 'grad', x: 6.8, y: 0, shape: 'stack', w: 2.8, h: 0.9, label: 'per-example $\\gvec_i$', tone: 0 },
    { id: 'clip', x: 10.2, y: 0, w: 2.6, h: 0.9, label: 'clip: $\\norm{\\bar\\gvec_i}_2 \\le C$', tone: 1 },
    { id: 'sum', x: 1, y: 2.6, shape: 'op', label: '$\\Sigma$' },
    { id: 'add', x: 3.2, y: 2.6, shape: 'op', label: '$+$' },
    {
      id: 'noise',
      x: 3.2,
      y: 4.1,
      shape: 'noise',
      w: 1,
      h: 1,
      label: '$\\Gauss(\\zeros, \\sigma^2 C^2 \\Imat)$',
      labelSide: 'e',
      tone: 1,
    },
    { id: 'div', x: 5.6, y: 2.6, shape: 'pill', w: 1, h: 0.6, label: '$\\div L$' },
    {
      id: 'step',
      x: 8.8,
      y: 2.6,
      w: 3.4,
      h: 0.9,
      label: '$\\thetavec_{t+1} = \\thetavec_t - \\eta_t \\tilde\\gvec_t$',
      tone: 0,
    },
    {
      id: 'acct',
      x: 8.8,
      y: 4.4,
      w: 3.4,
      h: 1,
      label: 'privacy accountant\n$(q, \\sigma, T) \\to (\\varepsilon, \\delta)$',
      tone: 2,
      dashed: true,
    },
  ],
  edges: [
    { from: 'data', to: 'lot' },
    { from: 'lot', to: 'grad' },
    { from: 'grad', to: 'clip' },
    {
      from: 'clip:s',
      to: 'sum:n',
      via: [
        [10.2, 1.3],
        [1, 1.3],
      ],
    },
    { from: 'sum', to: 'add' },
    { from: 'noise', to: 'add' },
    { from: 'add', to: 'div' },
    { from: 'div', to: 'step', label: '$\\tilde\\gvec_t$' },
    { from: 'step', to: 'acct', dashed: true, label: 'after $T$ steps', labelSide: 'right' },
  ],
}

/** One DP-SGD step: sample, clip per example, add noise, step. */
export function DpSgdStep() {
  return (
    <Figure
      title="One step of DP-SGD"
      purpose="Follow one DP-SGD step: sampling a lot, clipping each gradient, adding Gaussian noise and averaging."
      caption="Each example joins the lot independently with probability q. Its gradient is computed on its own and clipped to norm at most C, which bounds how much any one example can move the sum. Gaussian noise with standard deviation σC is added to the sum, which is divided by the expected lot size L and used for an ordinary gradient step. The accountant turns q, σ and the number of steps T into the privacy guarantee (ε, δ)."
    >
      <Diagram
        spec={spec}
        ariaLabel="Examples are Poisson-sampled into a lot; per-example gradients are clipped to norm C, summed, noised with Gaussian noise, divided by L and used for a gradient step; a privacy accountant tracks epsilon and delta"
      />
    </Figure>
  )
}
