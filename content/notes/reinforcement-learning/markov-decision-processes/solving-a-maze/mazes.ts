/** The preset mazes of the maze explorer, as rows of text, top row first. See parseMaze in _shared/maze.ts. */
export const MAZES = {
  room: {
    label: 'room',
    rows: ['.......G', '........', '........', '........', '........', '........', '........', 'S.......'],
  },
  corridors: {
    label: 'corridors',
    rows: [
      '..........G',
      '##########.',
      '.....#.....',
      '.###.#.###.',
      '.#.....#...',
      '.#######.##',
      '.#...#...#.',
      '.#.#.#.###.',
      '...#.#.#...',
      '####.#.#.#.',
      'S....#...#.',
    ],
  },
  cliff: {
    label: 'cliff',
    rows: ['..........', '..........', '..........', 'STTTTTTTTG'],
  },
  routes: {
    label: 'routes',
    rows: ['.........', '.#######.', '.#######.', '.#TTTTT#.', 'S.......G', '##TTTTT##'],
  },
} as const

export type MazeName = keyof typeof MAZES
