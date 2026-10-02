import type { Specimen } from '../../specimen'
import { ChainSpecimen, GamblersRuinSpecimen, MixingSpecimen } from './_markov/figures'

export const specimens: Specimen[] = [
  {
    module: 'probability/markov',
    title: 'Markov chains: stationarity, absorption and mixing',
    description:
      'A transition graph you can rearrange, with a walker and the distribution p₀Pᵗ converging to π; gambler’s ruin absorption probabilities and durations against simulated games; mixing times against the spectral gap.',
    tags: [
      'Markov chain',
      'stationary distribution',
      'absorption',
      "gambler's ruin",
      'mixing time',
      'spectral gap',
      'random walk',
    ],
    render: () => (
      <>
        <ChainSpecimen />
        <GamblersRuinSpecimen />
        <MixingSpecimen />
      </>
    ),
  },
]
