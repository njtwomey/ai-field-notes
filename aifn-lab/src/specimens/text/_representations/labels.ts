import { chrome } from '@lab/design/palette'
import { defineLayer, type CommonProps } from '@lab/viz'

export type WordLabelsProps = CommonProps & {
  x: ArrayLike<number>
  y: ArrayLike<number>
  labels: readonly string[]
  /** Bold labels in ink, always shown (the pinned word and its neighbours); otherwise secondary, hidden on overlap. */
  strong?: boolean
}

/**
 * Text labels beside points (a word map): one silent scatter series of invisible marks, each labelled to its right.
 * Overlapping secondary labels are hidden; a strong layer drawn above them (the pinned word) always shows.
 */
export const WordLabels = defineLayer<WordLabelsProps>({
  kind: 'WordLabels',
  legend: () => [],
  slotted: () => false,
  extent: () => undefined,
  build: (p, ctx) => {
    const c = chrome(ctx.mode)
    const data = Array.from({ length: p.x.length }, (_, i) => ({ value: [p.x[i], p.y[i]], name: p.labels[i] }))
    return {
      series: [
        {
          id: ctx.id,
          name: '__word-labels',
          type: 'scatter',
          data,
          symbolSize: 0,
          silent: true,
          clip: true,
          tooltip: { show: false },
          label: {
            show: true,
            position: 'right',
            distance: 4,
            formatter: (q: { name: string }) => q.name,
            color: p.strong ? c.ink : c.inkSecondary,
            fontWeight: p.strong ? 'bold' : 'normal',
            fontSize: p.strong ? 12 : 10,
          },
          labelLayout: { hideOverlap: !p.strong },
          z: p.strong ? 6 : 4,
        },
      ],
    }
  },
})
