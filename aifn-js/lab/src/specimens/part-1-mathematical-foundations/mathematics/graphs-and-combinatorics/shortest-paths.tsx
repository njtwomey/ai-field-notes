import type { Specimen } from '@lab/specimen'
import { MazeSpecimen } from './_shortest-paths/paths'

export const specimens: Specimen[] = [
  {
    module: 'part-1-mathematical-foundations/mathematics/graphs-and-combinatorics',
    title: 'Dijkstra against A* on a maze',
    description:
      'Shortest paths on a grid maze from a binary-heap priority queue: A* with a Manhattan heuristic expands fewer cells than Dijkstra and finds the same length.',
    tags: ['Dijkstra', 'A*', 'shortest path', 'priority queue', 'heuristic', 'maze'],
    render: () => <MazeSpecimen />,
  },
]
