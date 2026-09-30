import type { Specimen } from '../../specimen'
import { ClutterSpecimen } from './_expectation-propagation/ClutterSpecimen'
import { MatchSetSpecimen, MatchSpecimen } from './_expectation-propagation/TrueSkillSpecimen'

export const specimens: Specimen[] = [
  {
    module: 'inference/expectation-propagation',
    title: 'EP on the clutter problem',
    description:
      'Minka’s clutter problem stepped site by site: cavity, exact factor, tilted moments and the EP posterior against the exact one.',
    tags: ['expectation propagation', 'ADF', 'clutter problem', 'moment matching', 'damping'],
    render: () => <ClutterSpecimen />,
  },
  {
    module: 'inference/expectation-propagation',
    title: 'TrueSkill',
    description: 'One two-player match update from the published formulas, and EP over a small set of matches.',
    tags: ['TrueSkill', 'skill rating', 'truncated Gaussian', 'expectation propagation'],
    render: () => (
      <>
        <MatchSpecimen />
        <MatchSetSpecimen />
      </>
    ),
  },
]
