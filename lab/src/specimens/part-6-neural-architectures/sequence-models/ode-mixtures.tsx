import type { Specimen } from '@lab/specimen'
import { SvfmBehaviourShowcase } from './_ode-mixtures/behaviour'
import { SvfmEfficiencyShowcase } from './_ode-mixtures/efficiency'
import { SvfmToyShowcase } from './_ode-mixtures/toy'

const TAGS = ['showcase', 'neural ODE', 'mixture', 'stochastic vector field', 'forward filtering', 'NFE', 'training']

export const specimens: Specimen[] = [
  {
    module: 'part-6-neural-architectures/sequence-models',
    title: 'Neural ODEs with stochastic vector field mixtures: the toy tasks',
    description:
      'Twomey, Kozłowski & Santos-Rodríguez (2020): the crossing, splitting and scaling problems a neural ODE cannot solve, and moons, nested circles and XOR, trained in the worker with VF, SVF, VFM or SVFM, TLoss/VLoss, K components and pick-and-stick or forward-filtering selection; the learned transform over time, each component’s field and the component posterior along a path.',
    tags: TAGS,
    render: () => <SvfmToyShowcase />,
  },
  {
    module: 'part-6-neural-architectures/sequence-models',
    title: 'Neural ODEs with stochastic vector field mixtures: efficiency',
    description:
      'The forward-evaluation analysis of Twomey, Kozłowski & Santos-Rodríguez (2020), computed in the worker: VF, VF with TVLoss, SVFM and SVFM with TVLoss trained on moons, nested circles or XOR, then every instance solved alone by Dormand–Prince over tolerances; the per-instance NFE histograms against the single NFE usually reported, NFE against tolerance and against the VF variance along the path, and model against model instance by instance.',
    tags: TAGS,
    render: () => <SvfmEfficiencyShowcase />,
  },
  {
    module: 'part-6-neural-architectures/sequence-models',
    title: 'Neural ODEs with stochastic vector field mixtures: behaviour',
    description:
      'Forecasting walks in a house from the sofa to the front door, kitchen, landing or study (synthetic stand-in for the paper’s behavioural data), with VF, SVF, VFM and SVFM trained by FLoss: a single VF heads for the mean of the destinations, mixture components take the branches at the doorways, stochastic VFs spread along them; the component posterior along a forecast, and day against night with the time of day as an input.',
    tags: TAGS,
    render: () => <SvfmBehaviourShowcase />,
  },
]
