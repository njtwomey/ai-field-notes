/**
 * One drawing of a grid environment (gridworld, maze, FrozenLake, the cliff) from its `render` spec, used by every grid
 * page. Layers, bottom to top: the cells by kind (or a value field in their place), policy arrows, a path drawn as move
 * arrows (`pathMoves`: one arrow per direction of a cell edge, counted when walked more than once, the two directions
 * side by side, the latest move in ink), and the agent, in the success or destructive tone at an episode's ending.
 * Cells are square and the axes hold their first fit.
 */
import { useMemo } from 'react'
import type { EpisodeEnd, GridRender } from 'aifn/foundation/contracts'
import { Plot, Points, Raster, useAxis, Vectors, type AxisModel, type Range } from '@lab/viz'
import { cellXY, GRID_KINDS, kindRows, pathMoves, policyArrows, valueRows } from './grid'

export type GridValueField = {
  /** A value per cell index (V(s), or max_a Q(s, a)); walls and cliffs are left blank. */
  values: ArrayLike<number>
  /** The value's name in the tooltip and over the colour bar. */
  label: string
  /** The colour scale's ends (default symmetric about zero from the values). */
  range?: Range
  /** The colour bar beside the grid (default true). */
  colorBar?: boolean
  /** `diverging` (default) for signed values, `sequential` for values of one sign. */
  scale?: 'diverging' | 'sequential'
  /** Cell colour strength below 1, so a path in a colour near the scale's still reads (default 1). */
  fillOpacity?: number
}

export type GridViewProps<S> = {
  render: GridRender<S>
  title?: string
  /** A value field drawn as a raster in place of the cell kinds. */
  value?: GridValueField | null
  /** An action per cell index, drawn as arrows (a negative action draws nothing). */
  policy?: ArrayLike<number> | null
  /** A path of states, drawn as move arrows up to `step`. */
  path?: readonly S[] | null
  /** The path's last drawn index (default its end); the agent stands at this state. */
  step?: number
  /** The palette slot of the path's moves (default 1), for pages with more than one path. */
  pathSlot?: number
  /** Draw the path's latest move in ink (default true; false for a whole route such as a greedy path). */
  inkLatest?: boolean
  /** Mark the agent at the path's state at `step` (default true when there is a path). */
  agent?: boolean
  /** At an episode's last step, how it ended: the agent takes the success or destructive tone. */
  end?: EpisodeEnd | null
  /** Shared axes, when several grids zoom together; by default the view makes its own with equal units. */
  x?: AxisModel
  y?: AxisModel
  /** The plot's size in a `Plots` group. */
  scale?: number
}

const axisOf = (n: number) => Array.from({ length: n }, (_, i) => i)

export function GridView<S>({
  render: r,
  title,
  value,
  policy,
  path,
  step,
  pathSlot = 1,
  inkLatest = true,
  agent = true,
  end,
  x,
  y,
  scale,
}: GridViewProps<S>) {
  const { width, height } = r
  const xs = useMemo(() => axisOf(width), [width])
  const ys = useMemo(() => axisOf(height), [height])
  const kinds = useMemo(() => kindRows(r as GridRender<unknown>), [r])
  const values = value?.values
  const field = useMemo(() => (values ? valueRows(r as GridRender<unknown>, values) : null), [r, values])
  const arrows = useMemo(() => (policy ? policyArrows(r as GridRender<unknown>, policy) : null), [r, policy])
  const cells = useMemo(() => (path ? path.map((s) => cellXY(width, r.cell(s))) : null), [path, r, width])
  const at = cells ? Math.min(step ?? cells.length - 1, cells.length - 1) : -1
  const moves = useMemo(
    () => (cells ? pathMoves(cells, at, { slot: pathSlot, inkLatest }) : null),
    [cells, at, pathSlot, inkLatest],
  )
  const ownX = useAxis({ label: 'x', hold: 'initial', key: r })
  const ownY = useAxis({ label: 'y', equal: ownX, hold: 'initial', key: r })
  const tone = end ? (end.success ? 'success' : 'destructive') : undefined
  return (
    <Plot x={x ?? ownX} y={y ?? ownY} title={title} scale={scale}>
      {field ? (
        <Raster
          x={xs}
          y={ys}
          z={field}
          scale={value?.scale ?? 'diverging'}
          range={value?.range}
          valueLabel={value?.label}
          colorBar={value?.colorBar}
          fillOpacity={value?.fillOpacity}
        />
      ) : (
        <Raster x={xs} y={ys} z={kinds} scale="categorical" categoryNames={GRID_KINDS} />
      )}
      {arrows && <Vectors vectors={arrows} />}
      {moves && <Vectors vectors={moves} />}
      {cells && agent && at >= 0 && (
        <Points
          name="agent"
          x={[cells[at][0]]}
          y={[cells[at][1]]}
          emphasis={!end}
          tone={tone}
          size={end ? 16 : undefined}
        />
      )}
    </Plot>
  )
}
