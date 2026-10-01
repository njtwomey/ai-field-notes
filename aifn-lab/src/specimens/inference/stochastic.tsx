import type { Specimen } from '../../specimen'
import {
  BlockGibbs,
  Diagnostics,
  GibbsChains,
  GibbsZigZag,
  HmcTrajectories,
  LangevinBias,
  MetropolisBanana,
  ParticleFilter,
  WarmupAdaptation,
} from './_stochastic/figures'
import { HmcFunnelTrajectories, HmcSamplerComparison } from './_stochastic/showcase-hmc'

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
      'The axis-parallel zig-zag of systematic-scan Gibbs, four chains with their traces, ACF and diagnostics as the correlation grows, and block Gibbs with Rao–Blackwellised estimates.',
    tags: ['Gibbs', 'full conditionals', 'ChainView', 'autocorrelation', 'block Gibbs', 'Rao–Blackwell'],
    render: () => (
      <>
        <GibbsZigZag />
        <GibbsChains />
        <BlockGibbs />
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
    title: 'HMC and NUTS warmup: step-size adaptation',
    description:
      'Dual averaging tunes the leapfrog step size during warmup so that the mean acceptance statistic reaches its target δ, then freezes it at the averaged ε̄.',
    tags: ['HMC', 'NUTS', 'warmup', 'dual averaging', 'step size', 'acceptance statistic'],
    render: () => <WarmupAdaptation />,
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
  {
    module: 'inference/stochastic',
    title: 'Showcase: HMC on the funnel and a hard banana',
    description:
      'Leapfrog trajectories played step by step with their energy error and divergences on Neal’s funnel and a hard banana; the non-centred funnel removes the divergences; ESS and R̂ of four chains of random-walk MH, MALA and HMC.',
    tags: ['showcase', 'HMC', 'funnel', 'divergence', 'non-centred', 'ESS', 'R-hat', 'MALA'],
    render: () => (
      <>
        <HmcFunnelTrajectories />
        <HmcSamplerComparison />
      </>
    ),
  },
]
