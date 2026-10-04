import { Diagram, Figure, MathText } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const box = (id: string, x: number, y: number, label: string, tone: number | 'neutral') => ({
  id,
  x,
  y,
  w: 2.8,
  h: 1.2,
  label,
  tone,
})

const spec: DiagramSpec = {
  nodes: [
    box('acc', 0, 1.3, 'wrist and pocket\naccelerometers', 'neutral'),
    box('feat', 3.6, 1.3, 'mean, variance\n+ time of day', 0),
    box('clf', 7.2, 0, 'activity classifier\n(labelled data)', 1),
    box('km', 7.2, 2.6, '$k$-means clusters\n(no labels)', 1),
    box('vocab', 10.8, 1.3, 'soft word counts\nper 30-min window', 2),
    box('lda', 14.4, 1.3, 'LDA\n$T$ topics', 2),
    box('routine', 14.4, 3.9, 'topic activations\n= routines', 3),
  ],
  edges: [
    { from: 'acc', to: 'feat' },
    { from: 'feat:e', to: 'clf:w' },
    { from: 'feat:e', to: 'km:w' },
    { from: 'clf:e', to: 'vocab:w' },
    { from: 'km:e', to: 'vocab:w' },
    { from: 'vocab', to: 'lda' },
    { from: 'lda:s', to: 'routine:n' },
  ],
}

export function RoutinePipeline() {
  return (
    <Figure
      title="From acceleration to daily routines"
      caption={
        <MathText text="Short-window features are turned into discrete tokens, either activity labels from a supervised classifier or cluster indices from $k$-means. Soft token counts over 30-minute windows form documents. LDA's topics are activity patterns, and the topic proportions of successive windows trace the routines of the day." />
      }
    >
      <Diagram spec={spec} ariaLabel="Pipeline from accelerometer data to topic activations over a day" />
    </Figure>
  )
}
