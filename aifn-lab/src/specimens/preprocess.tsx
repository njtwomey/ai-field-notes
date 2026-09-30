import type { Specimen } from '../specimen'
import { PowerTransformSpecimen, SplineBasisSpecimen } from './_preprocess/figures'

export const specimens: Specimen[] = [
  {
    module: 'preprocess',
    title: 'Box–Cox and Yeo–Johnson',
    description:
      'powerTransform on skewed draws: histograms before and after, and the profile log-likelihood over λ with the maximum-likelihood λ.',
    tags: ['powerTransform', 'Box–Cox', 'Yeo–Johnson', 'Brent'],
    render: () => <PowerTransformSpecimen />,
  },
  {
    module: 'preprocess',
    title: 'Spline features',
    description:
      'The B-spline basis of splineFeatures for any number of knots and degree, and its two extrapolation modes.',
    tags: ['splineFeatures', 'B-spline', 'de Boor'],
    render: () => <SplineBasisSpecimen />,
  },
]
