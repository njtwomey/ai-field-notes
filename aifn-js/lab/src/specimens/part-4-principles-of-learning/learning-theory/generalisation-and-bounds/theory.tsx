import type { Specimen } from '@lab/specimen'
import { BiasVarianceSpecimen } from './_theory/bias-variance'
import { ShatteringSpecimen } from './_theory/capacity'
import { LimitTheoremsSpecimen, McdiarmidSpecimen, TailBoundsSpecimen } from './_theory/concentration'
import { DoubleDescentSpecimen } from './_theory/double-descent'

export const specimens: Specimen[] = [
  {
    module: 'part-4-principles-of-learning/learning-theory/generalisation-and-bounds',
    title: 'Double descent',
    description:
      'Random-features regression across the interpolation threshold: test error peaks where the number of features equals the number of points, with the weight norm, then falls again; ridge removes the peak.',
    tags: ['double descent', 'random features', 'interpolation', 'minimum norm', 'overparameterisation'],
    render: () => <DoubleDescentSpecimen />,
  },
  {
    module: 'part-4-principles-of-learning/learning-theory/generalisation-and-bounds',
    title: 'Bias and variance by resampling',
    description:
      'Fits of polynomial and nearest-neighbour regression to hundreds of training sets: the mean fit against the truth, bias² and variance at each x, and the decomposition across complexity with a draggable model.',
    tags: ['bias–variance decomposition', 'model complexity', 'resampling', 'overfitting'],
    render: () => <BiasVarianceSpecimen />,
  },
  {
    module: 'part-4-principles-of-learning/learning-theory/generalisation-and-bounds',
    title: 'Concentration inequalities against the truth',
    description:
      'Chebyshev, Hoeffding, Bernstein and Chernoff bounds against simulated tails of sample means; McDiarmid on the empty bins of balls into bins; the law of large numbers and the central limit theorem, and its failure with infinite variance.',
    tags: ['Hoeffding', 'Chernoff', 'Bernstein', 'McDiarmid', 'central limit theorem', 'law of large numbers'],
    render: () => (
      <>
        <TailBoundsSpecimen />
        <McdiarmidSpecimen />
        <LimitTheoremsSpecimen />
      </>
    ),
  },
  {
    module: 'part-4-principles-of-learning/learning-theory/generalisation-and-bounds',
    title: 'Shattering and Rademacher complexity',
    description:
      'Drag points and play through all their labellings: which ones half-planes, rectangles or intervals realise, whether the set is shattered, and the empirical Rademacher complexity of the realised labellings.',
    tags: ['VC dimension', 'shattering', 'Rademacher complexity', 'learning theory'],
    render: () => <ShatteringSpecimen />,
  },
]
