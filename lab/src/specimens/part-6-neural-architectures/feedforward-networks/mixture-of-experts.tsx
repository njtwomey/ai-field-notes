import type { Specimen } from '@lab/specimen'
import { MixtureOfExpertsShowcase } from './_mixture-of-experts/showcase'

export const specimens: Specimen[] = [
  {
    module: 'part-6-neural-architectures/feedforward-networks',
    title: 'Showcase: mixture of experts',
    description:
      'Linear or MLP experts under a softmax, top-k, noisy top-k, Switch, expert-choice or hierarchical gate, trained in the browser by EM or Adam with the load-balancing and router z-losses, on regime data whose true regimes score the gate; presets for EM on piecewise-linear data, balanced top-2 routing, and expert collapse.',
    tags: [
      'showcase',
      'mixture of experts',
      'hierarchical mixture of experts',
      'EM',
      'router',
      'top-k',
      'Switch',
      'load balancing',
      'expert collapse',
      'mixtureOfExpertsRun',
      'route',
    ],
    render: () => <MixtureOfExpertsShowcase />,
  },
]
