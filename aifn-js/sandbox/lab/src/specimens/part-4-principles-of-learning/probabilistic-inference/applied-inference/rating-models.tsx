import type { Specimen } from '@lab/specimen'
import { ChessSitesSpecimen, SettlingSpecimen } from './_rating-models/chess-sites'
import { ItemResponseSpecimen, RatingComparisonSpecimen, RatingTrajectorySpecimen } from './_rating-models/figures'
import { SkillTrackingSpecimen } from './_rating-models/tracking'

export const specimens: Specimen[] = [
  {
    module: 'part-4-principles-of-learning/probabilistic-inference/applied-inference',
    title: 'Rating systems: Elo, Glicko, Bradley–Terry',
    description:
      'Elo, Glicko and Glicko-2 ratings chasing the known skills of a simulated league game by game; the systems and a refitted Bradley–Terry model scored on predictive log loss and rank agreement; item response curves, test information and recovered difficulties.',
    tags: ['Elo', 'Glicko', 'Glicko-2', 'Bradley–Terry', 'paired comparison', 'item response theory', 'skill rating'],
    render: () => (
      <>
        <RatingTrajectorySpecimen />
        <RatingComparisonSpecimen />
        <ItemResponseSpecimen />
      </>
    ),
  },
  {
    module: 'part-4-principles-of-learning/probabilistic-inference/applied-inference',
    title: 'Chess ratings: chess.com against Lichess',
    description:
      'One simulated population with the same true skills and the same games, rated by chess.com’s Glicko-1, Lichess’s Glicko-2 and FIDE’s Elo with their published (or stated, assumed) settings: rating trajectories from start to settled, the three settled distributions, the mapping between the scales, games to settle and the provisional deviation shrinking.',
    tags: ['Glicko', 'Glicko-2', 'Elo', 'FIDE', 'chess.com', 'Lichess', 'skill rating', 'rating deviation'],
    render: () => (
      <>
        <ChessSitesSpecimen />
        <SettlingSpecimen />
      </>
    ),
  },
  {
    module: 'part-4-principles-of-learning/probabilistic-inference/applied-inference',
    title: 'Tracking a change in skill: Elo, Glicko, TrueSkill and smoothing',
    description:
      'A focal player whose true skill steps, drifts or wanders, against opponents of known skill: Elo, Glicko, Glicko-2 and TrueSkill follow it online and TrueSkill Through Time smooths over all games; lag, overshoot, noise, RMSE before and after, and the coverage of each uncertainty band.',
    tags: ['Elo', 'Glicko', 'Glicko-2', 'TrueSkill', 'TrueSkill Through Time', 'tracking', 'smoothing', 'skill rating'],
    render: () => <SkillTrackingSpecimen />,
  },
]
