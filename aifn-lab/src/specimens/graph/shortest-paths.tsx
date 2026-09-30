import type { Specimen } from '../../specimen'
import { MazeSpecimen } from './_shortest-paths/paths'

export const specimens: Specimen[] = [
  {
    module: 'graph/shortest-paths',
    title: 'Dijkstra against A* on a maze',
    description:
      'Shortest paths on a grid maze from a binary-heap priority queue: A* with a Manhattan heuristic expands fewer cells than Dijkstra and finds the same length.',
    tags: ['Dijkstra', 'A*', 'shortest path', 'priority queue', 'heuristic', 'maze'],
    render: () => <MazeSpecimen />,
  },
]
