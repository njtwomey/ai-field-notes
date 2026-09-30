import { chrome, seriesColor, type Mode } from './palette'

/** An arrow from `from` to `to`. Ink by default; `slot` colours it as that palette slot, `label` names it at its tip. */
export type Vector = { from: [number, number]; to: [number, number]; slot?: number; label?: string }

/** ECharts markLine data for arrows. Style set on the first point of a pair applies to the whole line. */
export function vectorLines(vectors: Vector[], mode: Mode) {
  return vectors.map((v) => {
    const color = v.slot === undefined ? chrome(mode).ink : seriesColor(mode, v.slot)
    const label = v.label ? { show: true, formatter: v.label, position: 'end', color, fontSize: 11 } : { show: false }
    return [{ coord: v.from, lineStyle: { color }, label }, { coord: v.to }]
  })
}
