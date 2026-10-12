import type { Specimen } from '@lab/specimen'
import { SplineVmClassificationShowcase, SplineVmShowcase } from './_splinevm/showcase'
import { SplineVm2dShowcase } from './_splinevm/showcase-2d'

const TAGS = [
  'prototype',
  'SplineVM',
  'splines',
  'free-knot splines',
  'ARD',
  'sparse Bayesian learning',
  'RVM',
  'evidence',
]

export const specimens: Specimen[] = [
  {
    module: 'part-5-learning-paradigms/supervised-learning/kernel-methods',
    title: 'Prototype: SplineVM, sparse Bayesian spline knots',
    description:
      'A spline in truncated-power form with an ARD prior on each knot’s weight: the evidence removes knots (fast-RVM α updates) and moves the rest (gradient on the knot locations), played iteration by iteration with the knot paths coloured by weight. Regression and classification in 1-D, a knot cost λ, and a 2-D classifier (axis knots and corner terms) shown as its prediction surface. Local numerics, not yet in the engine.',
    tags: TAGS,
    render: () => (
      <>
        <SplineVmShowcase />
        <SplineVmClassificationShowcase />
        <SplineVm2dShowcase />
      </>
    ),
  },
]
