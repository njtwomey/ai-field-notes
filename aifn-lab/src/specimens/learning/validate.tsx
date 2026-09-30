import type { Specimen } from '../../specimen'
import { NestedOptimismSpecimen, SplitterSpecimen } from './_validate/figures'

export const specimens: Specimen[] = [
  {
    module: 'learning/validate',
    title: 'Fold assignments and per-fold scores',
    description:
      'CrossValidationView of crossValidate with any splitter: the assignment matrix (train, test, unused per split and row) and each fold’s test accuracy and log loss.',
    tags: ['CrossValidationView', 'k-fold', 'stratified', 'grouped', 'time series', 'crossValidate'],
    render: () => <SplitterSpecimen />,
  },
  {
    module: 'learning/validate',
    title: 'Nested cross-validation: the optimism of tuning',
    description:
      'nested(outer, inner, gridSearch) on replicate datasets: the unnested best inner score against the nested outer score.',
    tags: ['nested', 'gridSearch', 'optimism', 'ridge'],
    render: () => <NestedOptimismSpecimen />,
  },
]
