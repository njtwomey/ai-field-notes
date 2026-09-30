import type { Specimen } from '../../specimen'
import {
  ArmaSpecimen,
  EmSpecimen,
  GarchSpecimen,
  HoltWintersSpecimen,
  KalmanTrackingSpecimen,
  StlSpecimen,
} from './_timeseries/figures'

export const specimens: Specimen[] = [
  {
    module: 'applied/timeseries',
    title: 'Kalman filter and RTS smoother',
    description: 'Tracking a target in the plane: filtered and smoothed estimates with their 2σ ellipses.',
    tags: ['Kalman filter', 'RTS smoother', 'state space', 'tracking'],
    render: () => <KalmanTrackingSpecimen />,
  },
  {
    module: 'applied/timeseries',
    title: 'ARMA processes',
    description: 'The AR(2) stationarity triangle, the lag-polynomial roots, a simulated series, and its ACF and PACF.',
    tags: ['ARMA', 'ACF', 'PACF', 'stationarity', 'roots'],
    render: () => <ArmaSpecimen />,
  },
  {
    module: 'applied/timeseries',
    title: 'Holt–Winters forecasting',
    description:
      'Additive and multiplicative seasonal exponential smoothing with forecasts, intervals and a least-squares fit.',
    tags: ['exponential smoothing', 'Holt–Winters', 'forecasting'],
    render: () => <HoltWintersSpecimen />,
  },
  {
    module: 'applied/timeseries',
    title: 'GARCH volatility',
    description: 'Volatility clustering and heavy tails of a GARCH(1,1), and the ACFs of returns and squared returns.',
    tags: ['GARCH', 'volatility', 'heavy tails'],
    render: () => <GarchSpecimen />,
  },
  {
    module: 'applied/timeseries',
    title: 'EM for a state-space model',
    description: 'Expectation–maximisation of a linear-Gaussian state-space model, step by step.',
    tags: ['EM', 'trace', 'TraceView', 'Kalman smoother'],
    render: () => <EmSpecimen />,
  },
  {
    module: 'applied/timeseries',
    title: 'STL decomposition',
    description: 'Trend, seasonal and remainder components by loess.',
    tags: ['STL', 'loess', 'seasonal decomposition'],
    render: () => <StlSpecimen />,
  },
]
