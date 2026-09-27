import { MathText } from '@/components/content/MathText'
import { GraphDiagram, Interactive } from '@/components/viz'

const FEATURES = ['1', '2', '3', 'd']
const X = [0, 1, 2, 3.6]

/** The class as the common parent of every feature. */
export function NaiveBayesGraph() {
  return (
    <Interactive
      title="Naive Bayes as a graphical model"
      caption={
        <MathText text="The class $y$ is the only parent of each feature $x_j$. With no edges between features, observing $y$ blocks every path between them: the features are conditionally independent given the class." />
      }
    >
      <GraphDiagram
        nodes={[
          { id: 'y', x: 1.8, y: 0 },
          ...FEATURES.map((j, i) => ({ id: `x${j}`, label: `x_${j}`, x: X[i], y: 1.2 })),
          { id: 'more', label: '⋯', x: 2.8, y: 1.2, kind: 'text' },
        ]}
        edges={FEATURES.map((j) => ({ source: 'y', target: `x${j}` }))}
        height={180}
        ariaLabel="Class node y with arrows to features x_1 to x_d"
      />
    </Interactive>
  )
}
