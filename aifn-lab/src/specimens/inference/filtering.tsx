import type { Specimen } from '../../specimen'
import { KalmanTracking } from './_filtering/showcase-kalman'
import { BocpdShowcase } from './_filtering/showcase-bocpd'

export const specimens: Specimen[] = [
  {
    module: 'inference/filtering',
    title: 'Showcase: Kalman filter and RTS smoother tracking a target',
    description:
      'A nearly-constant-velocity target in the plane: predict and update ellipses played step by step, the RTS smoother pass played backwards, toggled dropouts, the filter’s noise on sliders, and NIS and NEES against their χ² bands.',
    tags: ['showcase', 'Kalman filter', 'RTS smoother', 'tracking', 'missing data', 'NIS', 'NEES'],
    render: () => <KalmanTracking />,
  },
  {
    module: 'inference/filtering',
    title: 'Showcase: Bayesian online changepoint detection',
    description:
      'Observations fed one at a time to the BOCPD recursion: the series with true and detected changepoints, the growing run-length posterior with its MAP path, and the one-step predictive, on mean, variance, Poisson and AR shifts and the coal-mining disasters.',
    tags: ['showcase', 'BOCPD', 'changepoint', 'run length', 'hazard', 'conjugate', 'coal mining'],
    render: () => <BocpdShowcase />,
  },
]
