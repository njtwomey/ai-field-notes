import type { DiagramEdge, DiagramNode, DiagramSpec, Tone } from 'aifn-render'

/**
 * The family database of the inductive logic programming notes: seven people over four generations. Ann is the
 * parent of Bob and Cid; Bob of Dee and Eve; Cid of Fay; Eve of Gus. Ann, Cid, Dee, Eve and Fay are female.
 */
export const PARENT: readonly (readonly [string, string])[] = [
  ['ann', 'bob'],
  ['ann', 'cid'],
  ['bob', 'dee'],
  ['bob', 'eve'],
  ['cid', 'fay'],
  ['eve', 'gus'],
]
export const FEMALE: readonly string[] = ['ann', 'cid', 'dee', 'eve', 'fay']
export const PEOPLE: readonly string[] = ['ann', 'bob', 'cid', 'dee', 'eve', 'fay', 'gus']

/**
 * The family extended by two generations, for tasks that need longer chains: Gus is the father of Hal, and Hal of
 * Ivy, who is female. Used by the predicate invention note.
 */
export const EXTENDED_PARENT: readonly (readonly [string, string])[] = [...PARENT, ['gus', 'hal'], ['hal', 'ivy']]
export const EXTENDED_FEMALE: readonly string[] = [...FEMALE, 'ivy']
export const EXTENDED_PEOPLE: readonly string[] = [...PEOPLE, 'hal', 'ivy']

/** The parent facts as Prolog text. */
export const PARENT_TEXT = PARENT.map(([a, b]) => `parent(${a}, ${b}).`).join(' ')
/** The female facts as Prolog text. */
export const FEMALE_TEXT = FEMALE.map((a) => `female(${a}).`).join(' ')
/** Both, the whole background knowledge. */
export const FAMILY_TEXT = `${PARENT_TEXT}\n${FEMALE_TEXT}`

const POSITION: Record<string, { x: number; y: number }> = {
  ann: { x: 2.6, y: 0 },
  bob: { x: 1.3, y: 1.6 },
  cid: { x: 4.2, y: 1.6 },
  dee: { x: 0.2, y: 3.2 },
  eve: { x: 2.4, y: 3.2 },
  fay: { x: 4.2, y: 3.2 },
  gus: { x: 2.4, y: 4.8 },
  hal: { x: 2.4, y: 6.4 },
  ivy: { x: 2.4, y: 8.0 },
}

/** How to draw the family: tones of people and of parent edges, and extra edges (such as a target relation). */
export interface FamilyHighlight {
  readonly nodeTone?: (person: string) => Tone | undefined
  readonly edgeTone?: (parent: string, child: string) => Tone | undefined
  readonly extra?: readonly DiagramEdge[]
}

/**
 * The family tree as a diagram: parent edges point down to the child; female people are shaded. With `extended`, the
 * two extra generations (Hal and Ivy) are drawn too.
 */
export function familyDiagram(h: FamilyHighlight = {}, options: { extended?: boolean } = {}): DiagramSpec {
  const people = options.extended ? EXTENDED_PEOPLE : PEOPLE
  const female = options.extended ? EXTENDED_FEMALE : FEMALE
  const parent = options.extended ? EXTENDED_PARENT : PARENT
  const nodes: DiagramNode[] = people.map((p) => {
    const tone = h.nodeTone?.(p)
    return {
      id: p,
      ...POSITION[p],
      shape: 'pill' as const,
      w: 1.3,
      h: 0.7,
      label: p,
      filled: female.includes(p),
      tone: (tone ?? 'ink') as Tone,
      highlight: tone !== undefined,
    }
  })
  const edges: DiagramEdge[] = parent.map(([a, b]) => {
    const tone = h.edgeTone?.(a, b)
    return {
      from: `${a}:s`,
      to: `${b}:n`,
      route: 'straight' as const,
      arrow: 'end' as const,
      tone: (tone ?? 'neutral') as Tone,
      highlight: tone !== undefined,
    }
  })
  return { unit: 40, nodes, edges: [...edges, ...(h.extra ?? [])] }
}
