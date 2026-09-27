import { MathText } from '@/components/content/MathText'
import { Diagram } from '@/components/diagram/Diagram'
import { link, variable } from '@/components/diagram/components'
import type { DiagramSpec } from '@/components/diagram/types'
import { Interactive } from '@/components/viz'

const FEATURES = ['1', '2', '3', 'd']
const X = [0, 1.4, 2.8, 5.2]

const spec: DiagramSpec = {
  unit: 52,
  nodes: [
    variable('y', 2.6, 0, '$y$'),
    ...FEATURES.map((j, i) => variable(`x${j}`, X[i], 1.8, `$x_${j}$`)),
    { id: 'more', x: 4, y: 1.8, shape: 'text', w: 0.6, label: '$\\cdots$' },
  ],
  edges: FEATURES.map((j) => link('y', `x${j}`)),
}

/** The class as the common parent of every feature. */
export function NaiveBayesGraph() {
  return (
    <Interactive
      title="Naive Bayes as a graphical model"
      caption={
        <MathText text="The class $y$ is the only parent of each feature $x_j$. With no edges between features, observing $y$ blocks every path between them: the features are conditionally independent given the class." />
      }
    >
      <Diagram spec={spec} ariaLabel="Class node y with arrows to features x_1 to x_d" />
    </Interactive>
  )
}
