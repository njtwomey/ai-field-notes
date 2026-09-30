import type { Specimen } from '../../../specimen'
import { RealDataSpecimen } from './_shared/figures'
import { PairPlotSpecimen, ParallelCoordinatesSpecimen } from './_shared/views'

export const specimens: Specimen[] = [
  {
    module: 'applied/data/real',
    title: 'Pair plot',
    description:
      'Every pair of features in one grid: per-class histograms or KDE curves on the diagonal, scatters below, correlations above, with linked brushing and hover across panels.',
    tags: ['pair plot', 'scatter matrix', 'brushing', 'correlation', 'KDE', 'PairPlotView'],
    render: () => <PairPlotSpecimen />,
  },
  {
    module: 'applied/data/real',
    title: 'Parallel coordinates',
    description:
      'One axis per feature and one line per row, coloured by class: brush axes to filter, drag to reorder, flip, and compare with Andrews curves.',
    tags: ['parallel coordinates', 'Andrews curves', 'brushing', 'high-dimensional', 'ParallelCoordinatesView'],
    render: () => <ParallelCoordinatesSpecimen />,
  },
  {
    module: 'applied/data/real',
    title: 'Real datasets',
    description: "Iris, Old Faithful and Anscombe's quartet, embedded.",
    tags: ['Iris', 'Old Faithful', 'Anscombe'],
    render: () => <RealDataSpecimen />,
  },
]
