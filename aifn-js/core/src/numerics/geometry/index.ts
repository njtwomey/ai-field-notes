/**
 * `aifn/numerics/geometry`: the geometry behind figures: covariance and precision ellipses, convex hulls and polygons, contour
 * lines by marching squares, grids for evaluating fields, and the probability
 * simplex in barycentric coordinates.
 *
 * - Ellipses: `covarianceEllipse`, `precisionEllipse` (at k standard deviations or a probability mass), `massToRadius`.
 * - Polygons: `convexHull` (Andrew's monotone chain), `polygonArea` (signed), `polygonCentroid`, `pointInPolygon`.
 * - Contours: `contourSegments`, `contourLines` (joined polylines), `contourLevels`.
 * - Grids: `meshgrid`, `grid2d`, `evaluateGrid` (z[i][j] = f(x[j], y[i])), `logspace`; `linspace` is in `aifn/foundation/tensor`.
 * - Simplex: `simplexVertices`, `barycentricToCartesian`, `cartesianToBarycentric`, `simplexGrid`.
 */

export { covarianceEllipse, massToRadius, precisionEllipse, type Ellipse, type EllipseOptions } from './ellipse'
export { convexHull, pointInPolygon, polygonArea, polygonCentroid, type Hull } from './planar'
export { contourLevels, contourLines, contourSegments } from './contours'
export {
  barycentricToCartesian,
  cartesianToBarycentric,
  simplexGrid,
  simplexVertices,
  type SimplexGrid,
} from './simplex'
export { grid2d, evaluateGrid, type Grid2d } from './grids'
export { geometryFunctions } from './registry'
