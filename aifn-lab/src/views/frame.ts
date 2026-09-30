import type { FigureProps } from '@lab/layout'

/**
 * The parts of a `Figure` that a view drawn as a whole figure passes through: a specimen gives the title, its own
 * controls, readouts and caption, and the view adds its controls and readouts after them.
 */
export type FrameProps = Pick<
  FigureProps,
  'id' | 'description' | 'controls' | 'readouts' | 'caption' | 'defaultSize'
> & {
  title?: string
}
