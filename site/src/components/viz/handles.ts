/**
 * Draggable handles: bind a parameter to a place on a chart.
 *
 * A handle has a position in data coordinates and an `onDrag` setter. Every chart (EChart, XYChart, Heatmap) accepts
 * `handles`. Pointer events only: charts never emit when their option changes, so a handle and a slider bound to the
 * same parameter cannot trigger each other.
 */
export type Vec2 = [number, number]

type Placement =
  /** A point to drag in two dimensions, e.g. a start position or a pair of parameters. */
  | { kind: 'point'; at: Vec2; onDrag: (p: Vec2) => void }
  /** A vertical guide line; dragging sets its x value, e.g. a rank, a threshold or an iteration. */
  | { kind: 'x'; at: number; onDrag: (x: number) => void }
  /** A horizontal guide line; dragging sets its y value, e.g. a probability on a cdf. */
  | { kind: 'y'; at: number; onDrag: (y: number) => void }

export type Handle = Placement & {
  label?: string
  /** Called once when the pointer lets go, e.g. to start work too slow to redo on every move. */
  onRelease?: () => void
}

/** How near, in pixels, the pointer must be to grab a handle when a chart has several. */
export const GRAB_RADIUS = 28
