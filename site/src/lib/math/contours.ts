/** Contour lines of a field sampled on a rectangular grid, by marching squares. */

export type ContourSegment = [[number, number], [number, number]]

/**
 * The segments where a field crosses `level`. `z[i][j]` is the value at (x[j], y[i]), the layout Heatmap uses. Each
 * grid square whose corners straddle the level contributes one or two segments, with end points placed by linear
 * interpolation along the square's edges. Saddle squares are resolved by the value at the square's centre.
 */
export function contourSegments(x: number[], y: number[], z: number[][], level: number): ContourSegment[] {
  const out: ContourSegment[] = []
  const cross = (xa: number, ya: number, za: number, xb: number, yb: number, zb: number): [number, number] => {
    const f = za === zb ? 0.5 : (level - za) / (zb - za)
    return [xa + f * (xb - xa), ya + f * (yb - ya)]
  }
  for (let i = 0; i + 1 < y.length; i++)
    for (let j = 0; j + 1 < x.length; j++) {
      // Corners anticlockwise from bottom-left: (j, i), (j+1, i), (j+1, i+1), (j, i+1).
      const [x0, x1, y0, y1] = [x[j], x[j + 1], y[i], y[i + 1]]
      const [a, b, c, d] = [z[i][j], z[i][j + 1], z[i + 1][j + 1], z[i + 1][j]]
      const index = (a > level ? 1 : 0) | (b > level ? 2 : 0) | (c > level ? 4 : 0) | (d > level ? 8 : 0)
      if (index === 0 || index === 15) continue
      const bottom = () => cross(x0, y0, a, x1, y0, b)
      const right = () => cross(x1, y0, b, x1, y1, c)
      const top = () => cross(x0, y1, d, x1, y1, c)
      const left = () => cross(x0, y0, a, x0, y1, d)
      const centreAbove = (a + b + c + d) / 4 > level
      switch (index) {
        case 1:
        case 14:
          out.push([left(), bottom()])
          break
        case 2:
        case 13:
          out.push([bottom(), right()])
          break
        case 3:
        case 12:
          out.push([left(), right()])
          break
        case 4:
        case 11:
          out.push([right(), top()])
          break
        case 6:
        case 9:
          out.push([bottom(), top()])
          break
        case 7:
        case 8:
          out.push([left(), top()])
          break
        case 5:
          // a and c above: joined through the centre when it is above too.
          if (centreAbove) out.push([left(), top()], [bottom(), right()])
          else out.push([left(), bottom()], [right(), top()])
          break
        case 10:
          if (centreAbove) out.push([left(), bottom()], [right(), top()])
          else out.push([left(), top()], [bottom(), right()])
          break
      }
    }
  return out
}
