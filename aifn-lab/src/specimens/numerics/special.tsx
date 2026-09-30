import { softplus } from 'aifn/numerics/special'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { Figure } from '@lab/layout'
import { XYChart } from '@lab/viz'

import type { Specimen } from '../../specimen'
import {
  DerivativeKernelsSpecimen,
  GammaFamilySpecimen,
  IncompleteGammaBetaSpecimen,
  NormalTailSpecimen,
  SoftmaxSpecimen,
  TruncatedMomentsSpecimen,
} from './_special/figures'

const xs = linspace(-10, 10, 201)

export const specimens: Specimen[] = [
  {
    module: 'numerics/special',
    title: 'softplus',
    description: 'log(1 + eˣ), computed without overflow for large |x|.',
    tags: ['stable forms'],
    render: () => (
      <Figure title="softplus(x)">
        <XYChart
          xLabel="x"
          yLabel="softplus(x)"
          series={[{ name: 'softplus', type: 'line', x: toFlat(xs), y: toFlat(softplus(xs)) }]}
        />
      </Figure>
    ),
  },
  {
    module: 'numerics/special',
    title: 'Normal cdf in the far tail',
    description:
      'log₁₀ Φ(z) to z = −40: normalLogCdf never underflows and normalCdf keeps relative accuracy to its underflow near −37.5, while the old A&S 7.1.26 form is noise below about −5.',
    tags: ['normal', 'erfc', 'tail accuracy', 'log cdf'],
    render: () => <NormalTailSpecimen />,
  },
  {
    module: 'numerics/special',
    title: 'Truncated-normal moments v and w',
    description:
      'v(t) + t and w(t) from the Mills-ratio continued fraction stay smooth to t = −30; the naive φ/Φ ratio with the old Φ breaks down.',
    tags: ['truncated normal', 'expectation propagation', 'trueskill', 'probit'],
    render: () => <TruncatedMomentsSpecimen />,
  },
  {
    module: 'numerics/special',
    title: 'log Γ, digamma and trigamma',
    description: 'log |Γ(x)|, ψ(x) and ψ₁(x) across the poles at 0, −1, −2, … (reflection for x < 0).',
    tags: ['gamma', 'digamma', 'trigamma', 'polygamma'],
    render: () => <GammaFamilySpecimen />,
  },
  {
    module: 'numerics/special',
    title: 'Incomplete gamma and beta, Student t quantiles',
    description:
      'The regularised incomplete gamma P(a, x), the regularised incomplete beta I_x(a, b), and t quantiles.',
    tags: ['incomplete gamma', 'incomplete beta', 'student t', 'quantile'],
    render: () => <IncompleteGammaBetaSpecimen />,
  },
  {
    module: 'numerics/special',
    title: 'Softmax with temperature',
    description:
      'softmax(x/T) of fixed logits: T → 0 approaches one-hot at the largest logit, T → ∞ approaches uniform.',
    tags: ['softmax', 'temperature', 'logsumexp', 'stable forms'],
    render: () => <SoftmaxSpecimen />,
  },
  {
    module: 'numerics/special',
    title: 'Derivatives through the tape',
    description:
      'Each one-argument primitive f evaluated on a whole grid tensor, with f′ from grad (aifn/autodiff) of Σ f(x), one backward sweep over its recorded derivative rules, against central differences.',
    tags: ['derivative', 'primitive', 'autodiff', 'tensor'],
    render: () => <DerivativeKernelsSpecimen />,
  },
]
