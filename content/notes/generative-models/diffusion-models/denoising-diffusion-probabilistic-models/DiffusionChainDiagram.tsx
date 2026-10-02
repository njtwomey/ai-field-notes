import { Interactive } from 'aifn-render'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramSpec } from '@/components/diagram/types'

const spec: DiagramSpec = {
  nodes: [
    { id: 'xT', x: 0, y: 1, shape: 'noise', label: '$\\xvec_T$', tone: 'ink' },
    { id: 'd1', x: 1.8, y: 1, shape: 'text', w: 0.6, label: '$\\cdots$' },
    { id: 'xt', x: 3.6, y: 1, shape: 'circle', label: '$\\xvec_t$', tone: 'ink' },
    { id: 'xs', x: 7, y: 1, shape: 'circle', label: '$\\xvec_{t-1}$', tone: 'ink' },
    { id: 'd2', x: 8.8, y: 1, shape: 'text', w: 0.6, label: '$\\cdots$' },
    { id: 'x0', x: 10.6, y: 1, shape: 'circle', filled: true, label: '$\\xvec_0$', tone: 'ink' },
    { id: 'nl', x: 0, y: 1.9, shape: 'text', small: true, label: '$\\Gauss(\\zeros, \\Imat)$' },
    // Curved edges are not counted in the diagram's bounds; this empty node keeps the top arc inside the view.
    { id: 'pad', x: 7, y: -1.1, shape: 'text', w: 0.1, h: 0.1, label: '' },
    { id: 'dl', x: 10.6, y: 1.9, shape: 'text', small: true, label: 'data' },
  ],
  edges: [
    { from: 'xT', to: 'd1', route: 'straight' },
    { from: 'd1', to: 'xt', route: 'straight' },
    { from: 'xt', to: 'xs', route: 'straight', label: '$p_{\\thetavec}(\\xvec_{t-1} \\mid \\xvec_t)$' },
    { from: 'xs', to: 'd2', route: 'straight' },
    { from: 'd2', to: 'x0', route: 'straight' },
    {
      from: 'xs',
      to: 'xt',
      route: 'curve',
      bend: -0.9,
      dashed: true,
      label: '$q(\\xvec_t \\mid \\xvec_{t-1})$',
      labelRotate: false,
    },
    {
      from: 'x0',
      to: 'xt',
      route: 'curve',
      bend: 2.2,
      dashed: true,
      label: '$q(\\xvec_t \\mid \\xvec_0)$ in closed form',
      labelRotate: false,
    },
  ],
}

/** The DDPM chain: a fixed noising process one way, a learned denoising process the other. */
export function DiffusionChainDiagram() {
  return (
    <Interactive
      title="Forward and reverse processes"
      caption="Solid arrows are the learned reverse process, which generates by running from pure noise to data. Dashed arrows are the fixed forward process, which adds a little Gaussian noise per step and is used only in training; its marginal at any step t has a closed form, so training jumps straight from the data to a random step."
    >
      <Diagram
        spec={spec}
        ariaLabel="Diffusion Markov chain from x_T to x_0: learned reverse transitions p_theta left to right, fixed forward transitions q dashed right to left, and the closed-form jump q(x_t | x_0)"
      />
    </Interactive>
  )
}
