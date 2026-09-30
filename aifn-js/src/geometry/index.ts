/**
 * `aifn/geometry`: the geometry behind figures: covariance and precision ellipses, convex hulls and polygons, contour
 * lines by marching squares, grids for evaluating fields, decimation of long lines for drawing, and the probability
 * simplex in barycentric coordinates.
 *
 * - Ellipses: `covarianceEllipse`, `precisionEllipse` (at k standard deviations or a probability mass), `massToRadius`.
 * - Polygons: `convexHull` (Andrew's monotone chain), `polygonArea` (signed), `polygonCentroid`, `pointInPolygon`.
 * - Contours: `contourSegments`, `contourLines` (joined polylines), `contourLevels`.
 * - Grids: `meshgrid`, `grid2d`, `evaluateGrid` (z[i][j] = f(x[j], y[i])), `logspace`; `linspace` is in `aifn/tensor`.
 * - Decimation: `lttb` (largest triangle three buckets), `minMaxDecimate`.
 * - Simplex: `simplexVertices`, `barycentricToCartesian`, `cartesianToBarycentric`, `simplexGrid`.
 */

export {
  covarianceEllipse,
  massToRadius,
  precisionEllipse,
  type Ellipse,
  type EllipseOptions,
  type Matrix2Input,
  type Point2Input,
} from './ellipse'
export { convexHull, pointInPolygon, polygonArea, polygonCentroid, type Hull, type Points2Input } from './planar'
export { contourLevels, contourLines, contourSegments, type GridInput } from './contours'
export { evaluateGrid, grid2d, logspace, lttb, meshgrid, minMaxDecimate, type Decimated, type Grid2d } from './grids'
export {
  barycentricToCartesian,
  cartesianToBarycentric,
  simplexGrid,
  simplexVertices,
  type SimplexGrid,
} from './simplex'
