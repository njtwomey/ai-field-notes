import { createElement } from 'react'
import { QuickFigure, type QuickFigureProps } from './Show'

/** `QuickFigure` as a function: `quickFigure(dist, { title })`. */
export const quickFigure = (value: unknown, props: Omit<QuickFigureProps, 'value'> = {}) =>
  createElement(QuickFigure, { value, ...props })
