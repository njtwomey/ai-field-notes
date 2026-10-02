import { Interactive } from 'aifn-render'
import { Diagram } from '@/components/diagram/Diagram'
import { projector } from '@/components/diagram/components'
import type { DiagramSpec } from '@/components/diagram/types'

const spec: DiagramSpec = {
  nodes: [
    { id: 'x', x: 0, y: 2, w: 0.9, h: 0.9, label: '$\\xvec$' },
    projector('encoder', 'enc', 2.1, 2, 'encoder'),
    { id: 'ze', x: 4.3, y: 2, w: 1, h: 0.7, label: '$\\zvec_e$', tone: 0 },
    {
      id: 'q',
      x: 7,
      y: 2,
      w: 2.6,
      h: 1,
      label: 'nearest code\n$k^\\star = \\argmin_j \\norm{\\zvec_e - \\evec_j}$',
      tone: 2,
    },
    { id: 'zq', x: 10, y: 2, w: 1.8, h: 0.7, label: '$\\zvec_q = \\evec_{k^\\star}$', tone: 2 },
    projector('decoder', 'dec', 12.3, 2, 'decoder', { tone: 1 }),
    { id: 'xh', x: 14.4, y: 2, w: 0.9, h: 0.9, label: '$\\hat\\xvec$' },
    { id: 'cb', x: 7, y: 4.3, shape: 'stack', w: 2.6, h: 0.9, label: 'codebook $\\evec_1, \\dots, \\evec_K$', tone: 2 },
  ],
  edges: [
    { from: 'x', to: 'enc' },
    { from: 'enc', to: 'ze' },
    { from: 'ze', to: 'q' },
    { from: 'q', to: 'zq' },
    { from: 'zq', to: 'dec' },
    { from: 'dec', to: 'xh' },
    { from: 'cb', to: 'q' },
    {
      from: 'zq:n',
      to: 'ze:n',
      via: [
        [10, 0.4],
        [4.3, 0.4],
      ],
      dashed: true,
      label: 'straight-through: copy the gradient past the argmin',
      labelSide: 'right',
    },
  ],
}

/** Encode, snap to the nearest codebook vector, decode; the gradient skips the snap. */
export function VqVaeDiagram() {
  return (
    <Interactive
      title="The VQ-VAE forward and backward pass"
      caption="Each encoder output is replaced by its nearest codebook vector before decoding. The argmin has no useful derivative, so the backward pass copies the decoder's input gradient straight to the encoder output (dashed). The codebook is trained only by the codebook loss, which pulls the chosen vector towards the encoder output."
    >
      <Diagram
        spec={spec}
        ariaLabel="VQ-VAE: x to encoder to z_e, nearest codebook vector gives z_q, decoder reconstructs x; codebook feeds the lookup; straight-through gradient from z_q back to z_e"
      />
    </Interactive>
  )
}
