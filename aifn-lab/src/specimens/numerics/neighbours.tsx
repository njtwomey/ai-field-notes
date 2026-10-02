import type { Specimen } from '../../specimen'
import { BenchmarkFigure, HnswFigure, LshFigure, TreeSearchFigure } from './_neighbours/figures'

export const specimens: Specimen[] = [
  {
    module: 'numerics/neighbours',
    title: 'Nearest-neighbour search: trees, hashing and graphs',
    description:
      'Exact search by branch and bound in a k-d tree or ball tree, stepped node by node; locality-sensitive hashing with its buckets and candidates; an HNSW graph searched layer by layer with its greedy walk; and recall against speed for trees, LSH, IVF, PQ and HNSW measured in the browser. The query is draggable throughout.',
    tags: [
      'nearest neighbours',
      'k-d tree',
      'ball tree',
      'LSH',
      'HNSW',
      'IVF',
      'product quantisation',
      'recall',
      'treeQuery',
      'lshQuery',
      'hnswQuery',
      'annBenchmark',
    ],
    render: () => (
      <>
        <TreeSearchFigure />
        <LshFigure />
        <HnswFigure />
        <BenchmarkFigure />
      </>
    ),
  },
]
