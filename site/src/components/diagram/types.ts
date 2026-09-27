/**
 * A diagram is specified, not laid out: every node is placed by hand on a grid, groups are drawn around nodes or at
 * given rectangles, and edges are routed through ports and optional waypoints. Coordinates are grid units (one unit is
 * `unit` pixels, 40 by default); y grows downwards. Node positions are centres.
 */

/** A palette slot (0–7, the categorical data colours), or a neutral tone. */
export type Tone = number | 'neutral' | 'ink'

export type Side = 'n' | 's' | 'e' | 'w'

export type Shape =
  | 'box' // rounded rectangle, tinted with its tone
  | 'pill' // fully rounded rectangle
  | 'circle' // a variable or state
  | 'latent' // a latent variable: circle with a double border
  | 'noise' // a random input: dashed circle
  | 'op' // a small operator circle holding a symbol (⊕, ⊗, σ, tanh)
  | 'factor' // a small filled square: a factor in a factor graph
  | 'encoder' // trapezoid narrowing along `dir`: a projection into a smaller space
  | 'decoder' // trapezoid widening along `dir`: a projection back out
  | 'stack' // a box with layered copies behind it: a block repeated N times
  | 'dot' // a junction point where lines meet or split
  | 'text' // a label with no outline

export type Direction = 'right' | 'left' | 'up' | 'down'

export type DiagramNode = {
  id: string
  x: number
  y: number
  shape?: Shape
  /** Width and height in grid units; each shape has a default. */
  w?: number
  h?: number
  /** Text with `$…$` maths, rendered with KaTeX and the site macros. `\n` breaks lines. */
  label?: string
  tone?: Tone
  /** Flow direction for encoder and decoder trapezoids. */
  dir?: Direction
  dashed?: boolean
  /** Shade the node: an observed variable in a graphical model. */
  filled?: boolean
  /** Shade the node in proportion to a value in [0, 1], e.g. a probability; overrides `filled`. */
  shade?: number
  /** Draw in the accent colour with a heavier outline, e.g. the variables a sentence or a control is about. */
  highlight?: boolean
  /** Put the label outside the shape instead of inside (defaults to `n` for factors and dots). */
  labelSide?: Side
  /** Smaller label text, e.g. for annotations. */
  small?: boolean
}

export type DiagramGroup = {
  id: string
  label?: string
  tone?: Tone
  /** Draw around these nodes (with `pad` grid units of margin), or at an explicit rectangle. */
  around?: string[]
  pad?: number
  rect?: { x: number; y: number; w: number; h: number }
  /** Where the label sits. */
  labelAt?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right'
  dashed?: boolean
}

export type DiagramEdge = {
  /**
   * A node id, optionally with a side: `mha:s`. Without a side the side facing the other end is used. An edge from a
   * node to itself is a loop drawn outside the side given on `from` (default `n`), e.g. a self-transition.
   */
  from: string
  to: string
  /** Waypoints in grid units; the edge passes through them in order. */
  via?: [number, number][]
  /**
   * `ortho` (default) routes with right angles: through the waypoints if given, else with one or two elbows chosen
   * from the port sides. `straight` draws a direct line (the usual choice in graphical models). `curve` bows the line
   * sideways by `bend` grid units, e.g. for two edges between the same pair of nodes.
   */
  route?: 'ortho' | 'straight' | 'curve'
  bend?: number
  label?: string
  /** Where the label sits along the edge, as a fraction of its length (default: middle of the longest straight run). */
  labelPos?: number
  /** Which side of the edge the label sits on, relative to the direction of travel (default left: above a rightward edge). */
  labelSide?: 'left' | 'right'
  /** Gap between the edge and its label, in grid units. */
  labelOffset?: number
  /** Turn the label to run along the edge, kept upright (default true). */
  labelRotate?: boolean
  dashed?: boolean
  arrow?: 'end' | 'start' | 'both' | 'none'
  tone?: Tone
  highlight?: boolean
}

export type DiagramSpec = {
  nodes: DiagramNode[]
  edges?: DiagramEdge[]
  groups?: DiagramGroup[]
  /** Pixels per grid unit. */
  unit?: number
  /** How far the diagram may scale up beyond its natural size to fill its container. */
  maxScale?: number
  /**
   * Multiplies every position (nodes, waypoints, group rectangles) to space the layout out without resizing the shapes;
   * a pair scales x and y separately.
   */
  spread?: number | [number, number]
  /** Grow nodes whose label does not fit (default true). The label is measured after the first render. */
  fitLabels?: boolean
}
