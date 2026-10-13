import { useContext, useMemo, type ReactNode } from 'react'
import { FrameContext, Plot, Raster, Segments, useAxis } from 'aifn-render'
import { toRows } from 'aifn-compute/foundation/tensor'
import { assignment, type Split } from 'aifn-compute/learning/validate'

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
export function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** A row of categories under the grid, one per data row: class labels, for example. */
export type FoldStrip = {
  /** One category per data row, numbered from 0. */
  values: readonly number[]
  /** A name per category. */
  names: readonly string[]
  /** The strip's title. */
  title: string
}

type FoldGridProps = {
  /** The splits, as a splitter returns them. */
  splits: readonly Split[]
  /** The number of data rows. */
  n: number
  /** The plot's title. */
  title: string
  /** The y-axis label (default "split"). */
  rowLabel?: string
  /** Positions between data rows at which to draw a thin boundary, e.g. where one group ends and the next begins. */
  boundaries?: readonly number[]
  /** A categorical strip under the grid. */
  strip?: FoldStrip
  /** Pixels per split row (default 22). */
  rowHeight?: number
}

/**
 * The fold assignment of a splitter as a grid: one row per split, one column per data row; a cell is a training row
 * (slot 0), a test row (slot 1) or empty when the split leaves the row out. Drawn from `assignment` of
 * `aifn-compute/learning/validate`.
 */
export function FoldGrid({ splits, n, title, rowLabel = 'split', boundaries, strip, rowHeight = 22 }: FoldGridProps) {
  const view = useMemo(() => {
    const rows = toRows(assignment(splits, n)) as number[][]
    const z = rows.map((row) => row.map((v) => (v < 0 ? NaN : v)))
    const ys = rows.map((_, f) => f + 1)
    const lines = (boundaries ?? []).map((b) => ({
      from: [b, 0.5] as [number, number],
      to: [b, rows.length + 0.5] as [number, number],
    }))
    return { z, ys, lines }
  }, [splits, n, boundaries])
  const columns = useMemo(() => Array.from({ length: n }, (_, i) => i), [n])
  const x = useAxis({ label: 'data row', range: [-0.5, n - 0.5], nice: false })
  const y = useAxis({ label: rowLabel, range: [0.5, view.ys.length + 0.5], nice: false, integer: true })
  const sx = useAxis({ label: 'data row', range: [-0.5, n - 0.5], nice: false })
  const sy = useAxis({ categories: [strip?.title ?? ''] })
  const stripZ = useMemo(() => (strip ? [strip.values.map((v) => v + 2)] : []), [strip])
  return (
    <>
      <FixedHeight height={70 + rowHeight * view.ys.length}>
        <Plot x={x} y={y} title={title}>
          <Raster
            name="assignment"
            x={columns}
            y={view.ys}
            z={view.z}
            scale="categorical"
            categoryNames={['train', 'test']}
          />
          {view.lines.length > 0 && <Segments name="group boundary" segments={view.lines} />}
        </Plot>
      </FixedHeight>
      {strip && (
        <FixedHeight height={96}>
          <Plot x={sx} y={sy} title={strip.title}>
            <Raster
              name={strip.title}
              x={columns}
              y={[0]}
              z={stripZ}
              scale="categorical"
              categoryNames={['', '', ...strip.names]}
            />
          </Plot>
        </FixedHeight>
      )}
    </>
  )
}
