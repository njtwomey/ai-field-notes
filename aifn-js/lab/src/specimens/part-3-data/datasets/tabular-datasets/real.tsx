import type { Specimen } from '@lab/specimen'
import { RealDataSpecimen } from './_shared/figures'
import { PairPlotSpecimen, ParallelCoordinatesSpecimen } from './_shared/views'

export const specimens: Specimen[] = [
  {
    module: 'part-3-data/datasets/tabular-datasets',
    title: 'Pair plot',
    description:
      'Every pair of features in one grid: per-class histograms or KDE curves on the diagonal, scatters below, correlations above, with linked brushing and hover across panels.',
    tags: ['pair plot', 'scatter matrix', 'brushing', 'correlation', 'KDE', 'PairPlotView'],
    render: () => <PairPlotSpecimen />,
  },
  {
    module: 'part-3-data/datasets/tabular-datasets',
    title: 'Parallel coordinates',
    description:
      'One axis per feature and one line per row, coloured by class: brush axes to filter, drag to reorder, flip, and compare with Andrews curves.',
    tags: ['parallel coordinates', 'Andrews curves', 'brushing', 'high-dimensional', 'ParallelCoordinatesView'],
    render: () => <ParallelCoordinatesSpecimen />,
  },
  {
    module: 'part-3-data/datasets/tabular-datasets',
    title: 'Real datasets',
    description: "Iris, Old Faithful and Anscombe's quartet, embedded.",
    tags: ['Iris', 'Old Faithful', 'Anscombe'],
    render: () => <RealDataSpecimen />,
  },
]
