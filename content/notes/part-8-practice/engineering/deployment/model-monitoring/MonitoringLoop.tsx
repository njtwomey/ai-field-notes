import { Figure } from 'aifn-render'
import { Diagram } from 'aifn-render'
import type { DiagramNode, DiagramSpec } from 'aifn-render'

const stage = (id: string, x: number, label: string, tone: number | 'neutral' = 'neutral'): DiagramNode => ({
  id,
  x,
  y: 0,
  w: 2.8,
  h: 1,
  label,
  tone,
})
const monitor = (id: string, x: number, label: string): DiagramNode => ({ id, x, y: 2.3, w: 3.2, h: 1, label, tone: 2 })

const spec: DiagramSpec = {
  unit: 38,
  spread: [1.25, 1],
  nodes: [
    stage('train', 0, 'training pipeline', 0),
    stage('serve', 4, 'serving:\nfeatures, model'),
    stage('pred', 8, 'predictions\nand actions'),
    stage('out', 12, 'outcomes, labels\n(delayed)'),
    monitor('m1', 4, 'service health, data\nquality, training–serving skew'),
    monitor('m2', 8, 'input and prediction\ndistributions (drift)'),
    monitor('m3', 12, 'accuracy, calibration,\nbusiness metrics by slice'),
    { id: 'alert', x: 8, y: 4.6, shape: 'pill', w: 3.2, h: 0.9, label: 'outside expected range?', tone: 1 },
    { id: 'act', x: 2.6, y: 4.6, w: 3.4, h: 1, label: 'respond: roll back,\nretrain or fix the pipeline', tone: 1 },
  ],
  edges: [
    { from: 'train', to: 'serve', label: 'deploy' },
    { from: 'serve', to: 'pred' },
    { from: 'pred', to: 'out' },
    { from: 'serve', to: 'm1', dashed: true },
    { from: 'pred', to: 'm2', dashed: true },
    { from: 'out', to: 'm3', dashed: true },
    {
      from: 'm1:s',
      to: 'alert:n',
      via: [
        [4, 3.55],
        [8, 3.55],
      ],
    },
    { from: 'm2', to: 'alert' },
    {
      from: 'm3:s',
      to: 'alert:n',
      via: [
        [12, 3.55],
        [8, 3.55],
      ],
    },
    { from: 'alert', to: 'act', label: 'alert' },
    { from: 'act:w', to: 'train:s', via: [[0, 4.6]] },
  ],
}

/** The monitoring loop around a deployed model. */
export function MonitoringLoop() {
  return (
    <Figure
      title="The monitoring loop"
      purpose="Follow the signals a deployed model logs, from service health to delayed accuracy, and where each one triggers action."
      caption="Signals are logged at each stage of the serving path (dashed). Service health and data checks are available immediately, prediction distributions within a window, and accuracy only once the true outcomes arrive. A measurement outside its expected range raises an alert, and the response feeds back into training or deployment."
    >
      <Diagram
        spec={spec}
        ariaLabel="A training pipeline deploys a model to serving, which produces predictions and later outcomes; monitors on each stage feed an alert that triggers a rollback, retraining or a pipeline fix"
      />
    </Figure>
  )
}
