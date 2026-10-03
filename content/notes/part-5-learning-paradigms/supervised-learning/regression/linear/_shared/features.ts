/**
 * The diabetes features that get a colour in coefficient plots. bmi and s5 enter every path first; s1 and s2 are
 * correlated at 0.9 and show how each penalty treats a correlated pair; s3 is the coefficient the LARS–lasso path
 * drops when it crosses zero. The other five are drawn muted.
 */
export const HIGHLIGHT: Record<string, number> = { bmi: 0, s5: 1, s1: 2, s2: 3, s3: 4 }

/**
 * Plot style for a feature: its own name and a fixed palette slot if highlighted; otherwise muted and named "other
 * features", so the legend shows one entry for all of them and no series ever needs a ninth palette slot.
 */
export const featureStyle = (name: string) =>
  name in HIGHLIGHT ? { name, slot: HIGHLIGHT[name] } : { name: 'other features', muted: true }

/** Index of the grid value closest to x. */
export const nearest = (grid: number[], x: number) =>
  grid.reduce((best, g, k) => (Math.abs(g - x) < Math.abs(grid[best] - x) ? k : best), 0)
