/**
 * Sizing a column of Subplots panels when one panel has equal units. Pure, so it can be tested: the equal-aspect panel's
 * plot height is its plot width × (y span / x span), with both spans the panel's fitted data ranges; the other panels
 * share what remains of the frame's height by their ratios (at least `minHeight` each). If the total would exceed
 * `cap`, the column's plot width narrows (the caller centres it) until it fits. Ranges never change and units stay
 * equal.
 */

export type ColumnInput = {
  /** The column's full width in pixels. */
  width: number
  /** The height the frame offers (the size preset or the dragged size). */
  frameHeight: number
  /** The most the column may take; at least `frameHeight` is always allowed. */
  cap: number
  /** Relative heights of the rows. */
  ratios: readonly number[]
  /** The row with equal units, and its data spans. */
  equal: { row: number; xSpan: number; ySpan: number }
  /** Plot margins: left and right shared by the column, top and bottom per row. */
  margins: { left: number; right: number; top: readonly number[]; bottom: readonly number[] }
  /** Pixels of each panel outside its ECharts box (the toolbar row), per row. */
  chrome: readonly number[]
  gap: number
  minHeight: number
  /**
   * `frame` (default): the other rows share what remains of the frame by their ratios. `equal`: each other row's plot
   * height is its ratio over the equal row's ratio times the equal panel's plot height, e.g. a strip of bin counts a
   * fifth as tall as the square reliability diagram above it.
   */
  ratiosOf?: 'frame' | 'equal'
}

export type ColumnLayout = {
  /** Each row's panel height in pixels. */
  heights: number[]
  /** The column's width in pixels (the full width, or narrower to fit the cap). */
  width: number
  /** The equal-aspect panel's plot area in pixels. */
  plot: { width: number; height: number }
}

const MIN_PLOT_WIDTH = 120

export function layoutColumn(p: ColumnInput): ColumnLayout {
  const rows = p.ratios.length
  const e = p.equal.row
  const aspect = p.equal.ySpan / p.equal.xSpan
  const outside = (row: number) => p.margins.top[row] + p.margins.bottom[row] + p.chrome[row]
  const gaps = p.gap * (rows - 1)
  const others = p.ratios.map((_, i) => i).filter((i) => i !== e)
  const otherRatio = others.reduce((sum, i) => sum + p.ratios[i], 0) || 1

  const layoutAt = (plotWidth: number): ColumnLayout => {
    const plotHeight = Math.round(plotWidth * aspect)
    const equalHeight = plotHeight + outside(e)
    // The other rows share what remains of the frame's height, by their ratios, but never below the minimum.
    const remaining = Math.max(p.frameHeight - gaps - equalHeight, 0)
    const heights = p.ratios.map((r, i) =>
      i === e
        ? equalHeight
        : p.ratiosOf === 'equal'
          ? Math.max(p.minHeight, Math.round((plotHeight * r) / p.ratios[e]) + outside(i))
          : Math.max(p.minHeight, Math.round((remaining * r) / otherRatio)),
    )
    return {
      heights,
      width: Math.round(plotWidth + p.margins.left + p.margins.right),
      plot: { width: plotWidth, height: plotHeight },
    }
  }
  const total = (l: ColumnLayout) => l.heights.reduce((a, b) => a + b, 0) + gaps
  const limit = Math.max(p.cap, p.frameHeight)

  const full = layoutAt(Math.max(p.width - p.margins.left - p.margins.right, MIN_PLOT_WIDTH))
  if (total(full) <= limit) return full
  if (p.ratiosOf === 'equal') {
    // Every row scales with the equal panel's plot: shrink its width until the column fits the cap.
    const per = others.reduce((sum, i) => sum + p.ratios[i] / p.ratios[e], 0)
    const fixed = gaps + outside(e) + others.reduce((sum, i) => sum + outside(i), 0)
    const plotWidth = Math.max(
      MIN_PLOT_WIDTH,
      Math.min(full.plot.width, Math.floor((limit - fixed) / (aspect * (1 + per)))),
    )
    return layoutAt(plotWidth)
  }
  // Too tall: the others sit at their minimum, and the equal panel gets the rest of the cap.
  const room = limit - gaps - others.length * p.minHeight - outside(e)
  const plotWidth = Math.max(MIN_PLOT_WIDTH, Math.min(full.plot.width, Math.floor(room / aspect)))
  return layoutAt(plotWidth)
}
