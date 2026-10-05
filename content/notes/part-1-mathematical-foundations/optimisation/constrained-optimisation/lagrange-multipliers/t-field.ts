import { float, formatNumber } from 'aifn-render'

/** The position t of the point along the constraint curve, as a state field. */
export const tField = (
  range: [number, number],
  initial: number,
  step: number | undefined,
  label: string,
  format: (t: number) => string = formatNumber,
) => float(initial, { min: range[0], max: range[1], step, label, format })
