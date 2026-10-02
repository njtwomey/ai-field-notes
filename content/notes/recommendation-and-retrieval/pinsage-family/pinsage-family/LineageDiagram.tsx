import { Interactive } from 'aifn-render'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramEdge, DiagramSpec } from '@/components/diagram/types'

const model = (id: string, x: number, y: number, label: string, tone: number | 'neutral') => ({
  id,
  x,
  y,
  w: 2.3,
  h: 0.9,
  label,
  tone,
  small: true,
})

const year = (id: string, x: number, label: string) => ({ id, x, y: 6.3, shape: 'text' as const, small: true, label })

const straight = (from: string, to: string, dashed = false): DiagramEdge => ({ from, to, route: 'straight', dashed })

const spec: DiagramSpec = {
  unit: 34,
  nodes: [
    { id: 'rowi', x: -1.9, y: 1.4, shape: 'text', small: true, label: 'item and\nentity\nembeddings' },
    { id: 'rowu', x: -1.9, y: 4.4, shape: 'text', small: true, label: 'users and\nsequences' },
    model('sage', 0, 1.4, 'GraphSAGE', 'neutral'),
    model('pixie', 2.6, 4.4, 'Pixie', 1),
    model('pinsage', 2.6, 1.4, 'PinSage', 0),
    model('pinnersage', 5.2, 4.4, 'PinnerSage', 1),
    model('multibi', 7.8, 0.2, 'MultiBiSage', 0),
    model('itemsage', 7.8, 2.1, 'ItemSage', 0),
    model('pinnerformer', 7.8, 4.4, 'PinnerFormer', 1),
    model('transact', 10.4, 4.4, 'TransAct', 1),
    model('omnisearch', 13, 2.1, 'OmniSearchSage', 0),
    model('omnisage', 15.6, 1.4, 'OmniSage', 0),
    model('transact2', 15.6, 5.1, 'TransAct V2', 1),
    model('pinfm', 15.6, 3.6, 'PinFM', 1),
    year('y17', 0, '2017'),
    year('y18', 2.6, '2018'),
    year('y20', 5.2, '2020'),
    year('y22', 7.8, '2022'),
    year('y23', 10.4, '2023'),
    year('y24', 13, '2024'),
    year('y25', 15.6, '2025'),
  ],
  edges: [
    straight('sage', 'pinsage'),
    straight('pixie', 'pinsage'),
    straight('pinsage', 'pinnersage'),
    straight('pinsage', 'multibi'),
    straight('pinsage', 'itemsage'),
    straight('pinsage', 'pinnerformer'),
    straight('pinnersage', 'pinnerformer', true),
    straight('pinnerformer', 'transact'),
    straight('itemsage', 'omnisearch'),
    straight('transact', 'transact2'),
    straight('transact', 'pinfm'),
    straight('pinnerformer', 'omnisage'),
  ],
}

/** Who builds on whom in Pinterest's representation models, by year of publication. */
export function LineageDiagram() {
  return (
    <Interactive
      title="Lineage of the PinSage family"
      caption="Top row: models that embed items and other entities. Bottom row: models of users and their action sequences. An arrow means that the later model uses the earlier one's output or extends its method; the dashed arrow marks a replacement (PinnerFormer replaced PinnerSage as the user feature in ranking). PinSage embeddings are an input to almost every later model until OmniSage replaced them."
    >
      <Diagram
        spec={spec}
        ariaLabel="Timeline from GraphSAGE 2017 through Pixie and PinSage 2018, PinnerSage 2020, MultiBiSage, ItemSage and PinnerFormer 2022, TransAct 2023, OmniSearchSage 2024, to OmniSage, TransAct V2 and PinFM 2025"
      />
    </Interactive>
  )
}
