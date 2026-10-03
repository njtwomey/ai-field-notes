/** Axis-aligned boxes as [x₀, y₀, x₁, y₁] with x₀ < x₁ and y₀ < y₁. */
export type Box = [number, number, number, number]

const area = (b: Box) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1])

export type BoxOverlap = {
  intersection: number
  union: number
  iou: number
  /** Smallest box enclosing both. */
  enclosing: Box
  giou: number
  diou: number
  ciou: number
  /** CIoU's aspect-ratio consistency term v and its weight α. */
  v: number
  alpha: number
}

/**
 * IoU and its regression-friendly variants for a prediction b against ground truth g: GIoU (Rezatofighi et al. 2019),
 * and DIoU and CIoU (Zheng et al. 2020).
 */
export function overlap(b: Box, g: Box): BoxOverlap {
  const ix = Math.max(0, Math.min(b[2], g[2]) - Math.max(b[0], g[0]))
  const iy = Math.max(0, Math.min(b[3], g[3]) - Math.max(b[1], g[1]))
  const intersection = ix * iy
  const union = area(b) + area(g) - intersection
  const iou = union > 0 ? intersection / union : 0
  const enclosing: Box = [Math.min(b[0], g[0]), Math.min(b[1], g[1]), Math.max(b[2], g[2]), Math.max(b[3], g[3])]
  const c = area(enclosing)
  const giou = iou - (c - union) / c
  const centre = (r: Box) => [(r[0] + r[2]) / 2, (r[1] + r[3]) / 2]
  const [bx, by] = centre(b)
  const [gx, gy] = centre(g)
  const rho2 = (bx - gx) ** 2 + (by - gy) ** 2
  const diag2 = (enclosing[2] - enclosing[0]) ** 2 + (enclosing[3] - enclosing[1]) ** 2
  const diou = iou - rho2 / diag2
  const v =
    (4 / Math.PI ** 2) * (Math.atan((g[2] - g[0]) / (g[3] - g[1])) - Math.atan((b[2] - b[0]) / (b[3] - b[1]))) ** 2
  const alpha = v > 0 ? v / (1 - iou + v) : 0
  return { intersection, union, iou, enclosing, giou, diou, ciou: diou - alpha * v, v, alpha }
}
