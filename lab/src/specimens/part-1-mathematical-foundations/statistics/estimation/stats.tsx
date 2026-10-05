import type { Specimen } from '@lab/specimen'
import {
  HistogramSpecimen,
  QuantileSpecimen,
  AutocorrelationSpecimen,
  RunningMomentsSpecimen,
  BootstrapSpecimen,
  RankCorrelationSpecimen,
  ImportanceSpecimen,
} from './_stats/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-1-mathematical-foundations/statistics/estimation',
    title: 'Histogram bin rules and KDE',
    description:
      'histogram with an explicit rule (count, width, Sturges, Freedman–Diaconis) as a density, under a Gaussian KDE with Scott or Silverman bandwidth.',
    tags: ['histogram', 'kde', 'bandwidth'],
    render: () => <HistogramSpecimen />,
  },
  {
    module: 'part-1-mathematical-foundations/statistics/estimation',
    title: 'The thirteen quantile methods',
    description:
      'Each numpy.quantile method as a function of q on a small sample, against linear (the default) and the order statistics at k/n.',
    tags: ['quantile', 'ecdf', 'Hyndman–Fan'],
    render: () => <QuantileSpecimen />,
  },
  {
    module: 'part-1-mathematical-foundations/statistics/estimation',
    title: 'Autocorrelation: direct and FFT',
    description:
      'The sample ACF of an AR(1) series against φ^k; the FFT method gives the same values in O(n log n) instead of O(n²).',
    tags: ['acf', 'autocovariance', 'fft'],
    render: () => <AutocorrelationSpecimen />,
  },
  {
    module: 'part-1-mathematical-foundations/statistics/estimation',
    title: 'Running moments (Welford)',
    description:
      'runningMean and runningVariance converge on the true values, and stay accurate when a large offset is added to every value.',
    tags: ['welford', 'running mean', 'variance'],
    render: () => <RunningMomentsSpecimen />,
  },
  {
    module: 'part-1-mathematical-foundations/statistics/estimation',
    title: 'Bootstrap',
    description:
      'The bootstrap distribution of a median or mean, its standard error, and percentile and basic 95% intervals (lines).',
    tags: ['bootstrap', 'confidence interval', 'resampling'],
    render: () => <BootstrapSpecimen />,
  },
  {
    module: 'part-1-mathematical-foundations/statistics/estimation',
    title: 'Pearson, Spearman and Kendall',
    description:
      'A monotone nonlinear relation: the rank correlations see it fully, and one outlier moves Pearson far more than Spearman or Kendall.',
    tags: ['spearman', 'kendall', 'ranks', 'correlation'],
    render: () => <RankCorrelationSpecimen />,
  },
  {
    module: 'part-1-mathematical-foundations/statistics/estimation',
    title: 'Importance-sampling ESS',
    description:
      "Kish's effective sample size of importance weights for a shifted Gaussian proposal, from log-weights, against its theoretical value.",
    tags: ['ess', 'importance sampling'],
    render: () => <ImportanceSpecimen />,
  },
]
