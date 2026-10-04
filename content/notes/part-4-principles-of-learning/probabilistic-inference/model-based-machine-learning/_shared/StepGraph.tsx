import { useMemo } from 'react'
import { Diagram, Figure, float, MathText, useFigureState } from 'aifn-render'
import type { DiagramNode, DiagramSpec } from 'aifn-render'

/** One assumption: the node and group ids it adds (or removes), and a sentence saying what it assumes. */
export type GraphStep = { add: string[]; remove?: string[]; text: string }

const endId = (end: string) => end.split(':')[0]

/**
 * A factor graph built one assumption at a time. Each step reveals its nodes (highlighted while new), the edges whose
 * ends are both visible, and its groups. Edge labels, used for messages, appear from step `labelsFrom` on. Two
 * invisible corner anchors keep the drawing the same size at every step.
 */
export function StepGraph({
  title,
  caption,
  spec,
  steps,
  ariaLabel,
  labelsFrom,
}: {
  title: string
  caption: string
  spec: DiagramSpec
  steps: GraphStep[]
  ariaLabel: string
  labelsFrom?: number
}) {
  const state = useFigureState({
    step: float(steps.length - 1, {
      min: 0,
      max: steps.length - 1,
      step: 1,
      label: 'assumption',
      format: (v) => `${v + 1} of ${steps.length}`,
    }),
  })

  const anchors = useMemo((): DiagramNode[] => {
    const xs = spec.nodes.map((n) => n.x ?? 0)
    const ys = spec.nodes.map((n) => n.y ?? 0)
    const corner = (id: string, x: number, y: number): DiagramNode => ({
      id,
      x,
      y,
      shape: 'text',
      label: '',
      w: 0.1,
      h: 0.1,
    })
    return [
      corner('__anchor0', Math.min(...xs) - 1.2, Math.min(...ys) - 1),
      corner('__anchor1', Math.max(...xs) + 1.2, Math.max(...ys) + 1.3),
    ]
  }, [spec])

  const shown = useMemo((): DiagramSpec => {
    const ids = new Set<string>()
    for (const s of steps.slice(0, state.step + 1)) {
      s.add.forEach((id) => ids.add(id))
      s.remove?.forEach((id) => ids.delete(id))
    }
    const fresh = new Set(state.step > 0 ? steps[state.step].add : [])
    const labels = labelsFrom === undefined || state.step >= labelsFrom
    return {
      ...spec,
      nodes: [
        ...spec.nodes.filter((n) => ids.has(n.id)).map((n) => (fresh.has(n.id) ? { ...n, highlight: true } : n)),
        ...anchors,
      ],
      edges: (spec.edges ?? [])
        .filter((e) => ids.has(endId(e.from)) && ids.has(endId(e.to)))
        .map((e) => (labels ? e : { ...e, label: undefined })),
      groups: (spec.groups ?? [])
        .filter((g) => ids.has(g.id))
        .map((g) => (g.around ? { ...g, around: g.around.filter((n) => ids.has(n)) } : g))
        .filter((g) => !g.around || g.around.length > 0),
    }
  }, [spec, steps, state.step, labelsFrom, anchors])

  return (
    <Figure title={title} state={state} caption={caption}>
      <Diagram spec={shown} ariaLabel={ariaLabel} />
      <p className="min-h-10 text-sm text-muted-foreground">
        <MathText text={steps[state.step].text} />
      </p>
    </Figure>
  )
}
