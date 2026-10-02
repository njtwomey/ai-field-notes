import { Interactive } from 'aifn-render'
import { Diagram } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

// The hierarchical "community" multi-class Bayes point machine of Diethe, Twomey and Flach (2015, 2016), drawn as a
// directed graphical model with plates. Hyperparameters are shown as small nodes.
const spec: DiagramSpec = {
  unit: 38,
  nodes: [
    {
      id: 'mmu',
      x: 0,
      y: 0,
      shape: 'circle',
      w: 0.8,
      h: 0.8,
      label: '$\\mu_\\mu, \\sigma_\\mu$',
      tone: 'ink',
      small: true,
    },
    {
      id: 'kt',
      x: 3.4,
      y: 0,
      shape: 'circle',
      w: 0.8,
      h: 0.8,
      label: '$k_\\tau, \\vartheta_\\tau$',
      tone: 'ink',
      small: true,
    },
    { id: 'mu', x: 0, y: 1.9, shape: 'circle', label: '$\\mu_{cd}$', tone: 'ink' },
    { id: 'tau', x: 3.4, y: 1.9, shape: 'circle', label: '$\\tau_{cd}$', tone: 'ink' },
    { id: 'w', x: 1.7, y: 3.8, shape: 'circle', label: '$w_{rcd}$', tone: 'ink' },
    { id: 'x', x: 4.6, y: 3.8, shape: 'circle', filled: true, label: '$\\xvec_{rn}$', tone: 'ink' },
    { id: 's', x: 3.15, y: 5.5, shape: 'circle', label: '$s_{rnc}$', tone: 'ink' },
    { id: 'st', x: 3.15, y: 7.1, shape: 'circle', label: '$\\tilde s_{rnc}$', tone: 'ink' },
    { id: 'y', x: 3.15, y: 8.9, shape: 'circle', filled: true, label: '$y_{rn}$', tone: 'ink' },
  ],
  edges: [
    { from: 'mmu', to: 'mu', route: 'straight' },
    { from: 'kt', to: 'tau', route: 'straight' },
    { from: 'mu', to: 'w', route: 'straight', label: '$\\Gauss$' },
    { from: 'tau', to: 'w', route: 'straight' },
    { from: 'w', to: 's', route: 'straight', label: '$\\wvec_{rc}\\transpose \\xvec_{rn}$' },
    { from: 'x', to: 's', route: 'straight' },
    { from: 's', to: 'st', route: 'straight', label: '$+\\Gauss(0,1)$', labelSide: 'right' },
    { from: 'st', to: 'y', route: 'straight', label: 'arg max over $c$', labelSide: 'right' },
  ],
  groups: [
    {
      id: 'D',
      label: 'features $D$, classes $C$',
      tone: 'ink',
      around: ['mu', 'tau', 'w'],
      pad: 0.55,
      labelAt: 'top-left',
    },
    {
      id: 'N',
      label: 'examples $N_r$',
      tone: 'ink',
      around: ['x', 's', 'st', 'y'],
      pad: 0.45,
      labelAt: 'bottom-right',
    },
    { id: 'R', label: 'individuals $R$', tone: 0, rect: { x: 0.8, y: 3, w: 5.2, h: 6.9 }, labelAt: 'bottom-left' },
  ],
}

export function CommunityBpmDiagram() {
  return (
    <Interactive
      title="The community Bayes point machine"
      caption="Each individual r in the community has their own weights w, drawn from a shared Gaussian whose mean μ and precision τ are learnt from everyone. A score s = wᵀx gets Gaussian noise, and the label is the class with the largest noisy score. To personalise, the posterior over μ (and optionally τ) from the community becomes the prior for a new individual, whose weights are then updated online as labels arrive."
    >
      <Diagram
        spec={spec}
        ariaLabel="Hierarchical Bayes point machine: community mean and precision generate per-individual weights, which with features give noisy scores and labels"
      />
    </Interactive>
  )
}
