import type { ReactNode } from 'react'
import { Plot, Points, Raster, useAxis } from 'aifn-render'
import type { Grid, Problem } from './consistency-runs'

/** Class names of the chapter's data sets. */
export const MOON_NAMES = ['outer moon', 'inner moon']
export const BLOB_NAMES = ['class A', 'class B', 'class C']

/**
 * The input plane of a few-label problem: the classifier's field (P(class 1) − ½ for two classes, the predicted class
 * for three), the unlabelled points (small, muted, unless `hideUnlabelled`), the labelled points (large, by class),
 * and any further layers.
 */
export function BoundaryMap({
  problem,
  grid,
  box,
  values,
  stale,
  title,
  hideUnlabelled,
  children,
}: {
  problem: Problem
  grid: Grid
  box: { x: readonly [number, number]; y: readonly [number, number] }
  values: number[][] | null
  stale?: boolean
  title?: string
  hideUnlabelled?: boolean
  children?: ReactNode
}) {
  const xa = useAxis({ label: 'x₁', range: [box.x[0], box.x[1]] })
  const ya = useAxis({ label: 'x₂', range: [box.y[0], box.y[1]], equal: xa })
  const names = problem.classes === 2 ? MOON_NAMES : BLOB_NAMES
  return (
    <Plot x={xa} y={ya} title={title}>
      {values &&
        (problem.classes === 2 ? (
          <Raster
            x={grid.xs}
            y={grid.ys}
            z={values}
            scale="diverging"
            range={[-0.5, 0.5]}
            valueLabel="P(inner moon | x) − ½"
            boundary
            stale={stale}
            fillOpacity={0.55}
          />
        ) : (
          <Raster
            x={grid.xs}
            y={grid.ys}
            z={values}
            scale="categorical"
            categoryNames={names}
            valueLabel="predicted class"
            boundary
            stale={stale}
            fillOpacity={0.3}
          />
        ))}
      {!hideUnlabelled && (
        <Points
          name="unlabelled"
          x={problem.unlabelled.map((p) => p[0])}
          y={problem.unlabelled.map((p) => p[1])}
          muted
          thin
        />
      )}
      {children}
      <Points
        name="labelled"
        x={problem.labelled.map((p) => p[0])}
        y={problem.labelled.map((p) => p[1])}
        group={problem.labelledY}
        groupNames={names}
        size={11}
      />
    </Plot>
  )
}
