import { expectedClass, type Fitted, type Point } from './ordinal'

/** The part of every 2-D ordinal caption that explains the shared view. */
export const MAP_CAPTION =
  'The muted fill is the expected class E[y | x] on the class colour scale, or with the decision fill the class the model predicts; the black lines are contours of E[y | x] at the half-way values between classes. Points are coloured by their true class; the marker shape shows the error on each (circle exact, square off by one, triangle off by two or more). Drag the query point to read the class probabilities there.'

/** Share of the grid where a model's class probabilities go negative: the rank inconsistency of crossing splits. */
export function negativeShare(fitted: Fitted, grid: number[]): number {
  const cells = grid.flatMap((y) => grid.map((x) => fitted.probs([x, y]).some((p) => p < -1e-9)))
  return cells.filter(Boolean).length / cells.length
}

export const expectedAt = (fitted: Fitted, x: Point) => expectedClass(fitted.probs(x))
