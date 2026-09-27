import { MathText } from '@/components/content/MathText'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramEdge, DiagramNode, DiagramSpec } from '@/components/diagram/types'
import { Interactive } from '@/components/viz'

type Column = { id: string; title: string; models: string[]; tone: number }

const COLUMNS: Column[] = [
  { id: 'time', title: 'time', models: ['dynamic topic model', 'topics over time'], tone: 0 },
  {
    id: 'meta',
    title: 'metadata',
    models: ['author-topic model', 'structural topic model', 'supervised LDA', 'labelled LDA'],
    tone: 1,
  },
  { id: 'corr', title: 'topic correlation', models: ['correlated topic model', 'pachinko allocation'], tone: 2 },
  { id: 'np', title: 'number and structure', models: ['HDP topic model', 'hierarchical LDA'], tone: 3 },
  { id: 'short', title: 'text length', models: ['biterm topic model', 'one topic per text'], tone: 4 },
  { id: 'neural', title: 'inference', models: ['neural topic models'], tone: 5 },
]

const ROW = 1.5
const X0 = 6.4
const COL = 3.9
const MID = (ROW * (COLUMNS.length - 1)) / 2

const nodes: DiagramNode[] = [
  { id: 'plsa', x: 0, y: MID - 2.2, w: 2.4, h: 0.8, label: 'pLSA', tone: 'neutral' },
  { id: 'lda', x: 0, y: MID, w: 2.4, h: 0.9, label: 'LDA', tone: 'neutral', highlight: true },
  ...COLUMNS.flatMap((c, i) =>
    c.models.map((m, j) => ({
      id: `${c.id}${j}`,
      x: X0 + j * COL,
      y: i * ROW,
      w: 3.6,
      h: 0.8,
      label: m,
      tone: c.tone,
    })),
  ),
]

const edges: DiagramEdge[] = [
  { from: 'plsa:s', to: 'lda:n', label: 'Dirichlet priors', labelRotate: false, labelOffset: 0.2 },
  ...COLUMNS.map((c, i) => ({
    from: 'lda:e',
    to: `${c.id}0:w`,
    via: [[2.2, MID] as [number, number], [2.2, i * ROW] as [number, number]],
    label: c.title,
    labelRotate: false,
    labelOffset: 0.15,
    labelPos: 0.88,
  })),
]

const spec: DiagramSpec = { nodes, edges }

export function FamilyMap() {
  return (
    <Interactive
      title="The topic model family"
      caption={
        <MathText text="LDA adds Dirichlet priors to pLSA. Each extension below it changes one assumption of LDA, grouped by what it changes: when a document was written, who or what produced it, how topics co-occur, how many topics there are and how they are arranged, how long documents are, and how the posterior is computed." />
      }
    >
      <Diagram spec={spec} ariaLabel="LDA and the families of models that extend it" />
    </Interactive>
  )
}
