import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  unit: 40,
  nodes: [
    { id: 'init', x: 0, y: 0, w: 2.4, h: 0.9, label: 'start at $\\thetab^{(0)}$', tone: 'neutral' },
    {
      id: 'e',
      x: 4.4,
      y: 0,
      w: 4.4,
      h: 1.2,
      label: 'E-step: posterior of the latents\n$q^{(t)}(\\zvec) = p(\\zvec \\mid \\xvec, \\thetab^{(t)})$',
      tone: 0,
    },
    {
      id: 'm',
      x: 4.4,
      y: 2.6,
      w: 4.4,
      h: 1.2,
      label: 'M-step: maximise the bound\n$\\thetab^{(t+1)} = \\argmax_{\\thetab} Q(\\thetab, \\thetab^{(t)})$',
      tone: 1,
    },
    {
      id: 'conv',
      x: 9.4,
      y: 2.6,
      shape: 'pill',
      w: 3,
      h: 0.9,
      label: 'change in $\\ell$ and $\\thetab$ small?',
      tone: 2,
    },
    { id: 'stop', x: 9.4, y: 4.3, shape: 'text', w: 3, label: 'stationary point of $\\ell$' },
  ],
  edges: [
    { from: 'init', to: 'e' },
    { from: 'e', to: 'm', label: 'bound touches $\\ell$ at $\\thetab^{(t)}$', labelSide: 'right' },
    { from: 'm', to: 'conv' },
    { from: 'conv:n', to: 'e:e', via: [[9.4, 0]], label: 'no: $t \\leftarrow t + 1$', labelSide: 'right' },
    { from: 'conv', to: 'stop', label: 'yes', labelSide: 'right' },
  ],
}

/** The EM iteration as a loop. */
export function EmLoop() {
  return (
    <Figure
      title="The EM iteration"
      caption="The E-step computes the posterior of the latent variables at the current parameters, which makes the lower bound touch the log-likelihood ℓ there. The M-step maximises the bound over the parameters, so ℓ cannot fall. The loop stops when neither ℓ nor the parameters change much."
    >
      <Diagram
        spec={spec}
        ariaLabel="From a starting value, the E-step computes the latent posterior, the M-step maximises the expected complete-data log-likelihood, and the loop repeats until convergence"
      />
    </Figure>
  )
}
