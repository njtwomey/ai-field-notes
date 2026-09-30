import type { Specimen } from '../../specimen'
import {
  Diagnostics,
  GibbsChains,
  GibbsZigZag,
  HmcTrajectories,
  LangevinBias,
  MetropolisBanana,
  ParticleFilter,
} from './_stochastic/figures'

export const specimens: Specimen[] = [
  {
    module: 'inference/stochastic',
    title: 'Metropolis–Hastings on a banana',
    description:
      'Random-walk Metropolis on a twisted Gaussian: the chain and its rejected proposals, and the acceptance rate and ESS per step against the proposal scale.',
    tags: ['Metropolis–Hastings', 'random walk', 'acceptance rate', 'ESS', 'banana'],
    render: () => <MetropolisBanana />,
  },
  {
    module: 'inference/stochastic',
    title: 'Gibbs sampling on a correlated Gaussian',
    description:
      'The axis-parallel zig-zag of systematic-scan Gibbs, and four chains with their traces, ACF and diagnostics as the correlation grows.',
    tags: ['Gibbs', 'full conditionals', 'ChainView', 'autocorrelation'],
    render: () => (
      <>
        <GibbsZigZag />
        <GibbsChains />
      </>
    ),
  },
  {
    module: 'inference/stochastic',
    title: 'Hamiltonian Monte Carlo trajectories',
    description:
      'Leapfrog trajectories of HMC and NUTS on a banana, a correlated Gaussian and Neal’s funnel, with the energy along each trajectory and divergences.',
    tags: ['HMC', 'NUTS', 'leapfrog', 'energy', 'divergence', 'funnel'],
    render: () => <HmcTrajectories />,
  },
  {
    module: 'inference/stochastic',
    title: 'Langevin: MALA against ULA',
    description:
      'The O(h) bias of the unadjusted Langevin algorithm and its removal by the Metropolis-adjusted version.',
    tags: ['Langevin', 'ULA', 'MALA', 'bias', 'step size'],
    render: () => <LangevinBias />,
  },
  {
    module: 'inference/stochastic',
    title: 'Convergence diagnostics: ESS and R̂',
    description:
      'Trace plots, histogram, ACF and running means of dispersed chains, with the bulk and tail ESS, rank-normalised split R̂ and MCSE.',
    tags: ['diagnostics', 'ESS', 'R-hat', 'MCSE', 'ChainView'],
    render: () => <Diagnostics />,
  },
  {
    module: 'inference/stochastic',
    title: 'Bootstrap particle filter',
    description:
      'A particle filter on the nonlinear growth model: the particle cloud, filter mean and band, the ESS and adaptive resampling with four schemes.',
    tags: ['particle filter', 'SMC', 'resampling', 'ESS', 'state-space model'],
    render: () => <ParticleFilter />,
  },
]
