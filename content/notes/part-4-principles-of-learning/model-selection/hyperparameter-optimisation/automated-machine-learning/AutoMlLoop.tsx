import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const BOX = { w: 3.4, h: 1.5 }

const spec: DiagramSpec = {
  unit: 40,
  nodes: [
    { id: 'data', x: 0, y: 0, ...BOX, tone: 'neutral', label: 'new data set' },
    { id: 'warm', x: 4.6, y: 0, ...BOX, w: 3.8, tone: 2, label: 'warm start\nconfigurations of past tasks' },
    { id: 'bo', x: 9.6, y: 0, ...BOX, w: 3.8, tone: 0, label: 'Bayesian optimiser\nsurrogate + acquisition' },
    { id: 'prep', x: 5.6, y: 3.4, ...BOX, tone: 1, label: 'data\npreprocessor' },
    { id: 'feat', x: 9.6, y: 3.4, ...BOX, tone: 1, label: 'feature\npreprocessor' },
    { id: 'clf', x: 13.6, y: 3.4, ...BOX, tone: 1, label: 'learning\nalgorithm' },
    { id: 'store', x: 13.6, y: 6.8, ...BOX, tone: 'neutral', label: 'stored validation\npredictions' },
    { id: 'ens', x: 9.6, y: 6.8, ...BOX, tone: 3, label: 'ensemble\nselection' },
    { id: 'out', x: 5.6, y: 6.8, ...BOX, tone: 'ink', label: 'final\nensemble' },
  ],
  groups: [
    { id: 'pipe', around: ['prep', 'feat', 'clf'], pad: 0.4, label: 'pipeline with configuration $\\lambdavec$' },
  ],
  edges: [
    { from: 'data:e', to: 'warm:w' },
    { from: 'warm:e', to: 'bo:w' },
    { from: 'bo:s', to: 'feat:n', label: 'next $\\lambdavec$', labelSide: 'right', labelRotate: false },
    { from: 'prep:e', to: 'feat:w' },
    { from: 'feat:e', to: 'clf:w' },
    { from: 'clf:n', to: 'bo:e', via: [[13.6, 0]], label: 'validation loss' },
    { from: 'clf:s', to: 'store:n', label: 'every model' },
    { from: 'store:w', to: 'ens:e' },
    { from: 'ens:w', to: 'out:e' },
  ],
}

/** The loop of an AutoML system in the style of Auto-sklearn: warm start, Bayesian optimisation, ensemble selection. */
export function AutoMlLoop() {
  return (
    <Figure
      title="The loop of an AutoML system"
      caption="A system in the style of Auto-sklearn. The search space is the set of pipelines: a data preprocessor, a feature preprocessor and a learning algorithm, each with its own hyperparameters, chosen together as one configuration λ. The first trials come from earlier data sets. A Bayesian optimiser then proposes each next configuration from the validation losses so far. Every model it trains keeps its predictions on the validation data, and ensemble selection combines the stored models at the end instead of returning only the single best one."
    >
      <Diagram
        spec={spec}
        ariaLabel="A new data set feeds a warm start of configurations from past tasks, which seeds a Bayesian optimiser. The optimiser proposes a configuration for a pipeline of data preprocessor, feature preprocessor and learning algorithm; the pipeline's validation loss returns to the optimiser. Every trained model's validation predictions are stored, and ensemble selection combines them into the final ensemble."
      />
    </Figure>
  )
}
