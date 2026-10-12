import type { Specimen } from '@lab/specimen'
import { SplineVbShowcase } from './_splinevm/showcase-vb'
import { VbWriteup } from './_splinevm/writeup-vb'

export const specimens: Specimen[] = [
  {
    module: 'part-5-learning-paradigms/supervised-learning/kernel-methods',
    title: 'Prototype: variational GAM with adaptive smoothing',
    description:
      'A cubic-spline GAM term under any loss read as a likelihood (Gaussian, Student-t, Laplace, Huber, pinball, logistic, Poisson), fitted by variational Bayes. Each knot’s relevance is drawn around the GAM’s shared smoothing level, so the model departs from the GAM only where the evidence pays; one knob ν runs from the plain GAM to ARD-like sparsity. ELBO as fit minus complexity, bits per knot, and a sweep of complexity against n. Proof of concept with local numerics.',
    tags: [
      'prototype',
      'SplineVM',
      'GAM',
      'variational Bayes',
      'ARD',
      'adaptive smoothing',
      'generalised likelihood',
      'quantile regression',
    ],
    render: () => (
      <>
        <SplineVbShowcase />
        <VbWriteup />
      </>
    ),
  },
]
