import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  unit: 40,
  nodes: [
    { id: 'raw', x: 0, y: 2, w: 3.2, h: 1.2, label: 'raw data\ntables, event streams', tone: 'neutral' },
    { id: 'defs', x: 4.6, y: 0, w: 3.6, h: 1.2, label: 'feature definitions\none per feature, versioned', tone: 0 },
    { id: 'batch', x: 4.6, y: 2, w: 3.6, h: 1.2, label: 'batch and streaming\ncomputation', tone: 0 },
    {
      id: 'offline',
      x: 9.8,
      y: 0.6,
      w: 3.8,
      h: 1.4,
      label: 'offline store\nevery value with its\nevent time',
      tone: 1,
    },
    { id: 'online', x: 9.8, y: 3.4, w: 3.8, h: 1.4, label: 'online store\nlatest value per\nentity key', tone: 1 },
    {
      id: 'train',
      x: 15.2,
      y: 0.6,
      w: 3.8,
      h: 1.4,
      label: 'training set by\npoint-in-time join\non (entity, $t_0$)',
      tone: 2,
    },
    { id: 'serve', x: 15.2, y: 3.4, w: 3.8, h: 1.4, label: 'prediction request\nlookup by entity key', tone: 2 },
  ],
  edges: [
    { from: 'raw', to: 'batch' },
    { from: 'defs', to: 'batch' },
    { from: 'batch', to: 'offline' },
    { from: 'batch', to: 'online', label: 'materialise', labelSide: 'left' },
    { from: 'offline', to: 'train' },
    { from: 'online', to: 'serve', label: 'milliseconds', labelSide: 'right' },
  ],
}

/** One set of feature definitions feeding a historical store for training and a latest-value store for serving. */
export function FeatureStoreDiagram() {
  return (
    <Figure
      title="One definition, two stores"
      hoverReadout={false}
      caption="Each feature is defined once. Batch and streaming jobs compute it from raw data and write every value, with the time it refers to, to the offline store, and the latest value per entity to the online store. Training sets are built from the offline store by point-in-time joins; predictions read the online store by entity key. Because both stores are filled from the same definitions and computations, a feature has the same meaning in training and in serving."
    >
      <Diagram
        spec={spec}
        ariaLabel="Raw data and versioned feature definitions feed batch and streaming computation, which writes every timestamped value to an offline store and the latest value per entity to an online store; the offline store builds training sets by point-in-time joins and the online store answers prediction requests by entity key"
      />
    </Figure>
  )
}
