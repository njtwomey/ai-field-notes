import type { Specimen } from '../../specimen'
import { CounterfactualShowcase } from './_explain/counterfactuals'
import { RulesShowcase } from './_explain/rules'
import { AttributionsShowcase } from './_explain/attributions'
import { ConceptsShowcase } from './_explain/concepts'
import { EffectsShowcase } from './_explain/effects'
import { ExplainShowcase } from './_explain/showcase'
import { ValuationShowcase } from './_explain/valuation'

export const specimens: Specimen[] = [
  {
    module: 'learning/explain',
    title: 'Explaining a model: SHAP, LIME and gradients',
    description:
      'A random forest, a decision tree or a small MLP trained in the browser on a task with main effects, an interaction and irrelevant features: a clicked row explained by TreeSHAP, KernelSHAP, LIME, integrated gradients and SmoothGrad; KernelSHAP converging to exact Shapley values as coalitions grow; permutation importance and partial dependence with ICE curves.',
    tags: [
      'showcase',
      'SHAP',
      'Shapley values',
      'KernelSHAP',
      'TreeSHAP',
      'LIME',
      'integrated gradients',
      'SmoothGrad',
      'permutation importance',
      'partial dependence',
      'ICE',
    ],
    render: () => <ExplainShowcase />,
  },
  {
    module: 'learning/explain',
    title: 'Counterfactual explanations: Wachter, DiCE and FACE',
    description:
      'An MLP trained in the browser on two classes joined by a corridor of data: a pinned point explained by Wachter et al.’s gradient search, DiCE’s diverse counterfactuals and FACE’s density-weighted shortest paths over a graph of the data, with a draggable density threshold and immutable or monotone features; Growing Spheres’ shells.',
    tags: [
      'showcase',
      'counterfactual explanations',
      'Wachter',
      'DiCE',
      'FACE',
      'actionable recourse',
      'Growing Spheres',
      'Dijkstra',
      'kernel density estimation',
    ],
    render: () => <CounterfactualShowcase />,
  },
  {
    module: 'learning/explain',
    title: 'Anchors and surrogate rules',
    description:
      'A random forest on the two moons: a pinned point’s anchor (a box of quantile-bin predicates found by beam search with KL-LUCB) with its precision and coverage; surrogate trees of growing depth and a sequential-covering decision list fitted to the forest, with their fidelity on fresh data.',
    tags: ['showcase', 'anchors', 'KL-LUCB', 'rule lists', 'global surrogate', 'fidelity', 'decision tree'],
    render: () => <RulesShowcase />,
  },
  {
    module: 'learning/explain',
    title: 'Attributions compared and evaluated',
    description:
      'An MLP trained in the browser on noise images or sequences with a planted motif: gradient, gradient × input, integrated gradients, SmoothGrad, DeepLIFT, expected gradients and occlusion beside the true motif cells, with a choice of baseline; deletion and insertion curves; sanity checks by cascading model randomisation and random labels.',
    tags: [
      'showcase',
      'saliency',
      'integrated gradients',
      'DeepLIFT',
      'occlusion',
      'SmoothGrad',
      'deletion curve',
      'insertion curve',
      'sanity checks',
    ],
    render: () => <AttributionsShowcase />,
  },
  {
    module: 'learning/explain',
    title: 'Which training points matter: influence, TracIn and data Shapley',
    description:
      'Two Gaussian blobs with planted label flips and an L2 logistic regression: influence functions, self-influence, TracIn, exact KNN-Shapley and TMC data Shapley (streamed from the worker) rank the training points, and gain curves show how quickly each finds the flipped labels.',
    tags: ['showcase', 'influence functions', 'TracIn', 'data Shapley', 'KNN-Shapley', 'label noise', 'data valuation'],
    render: () => <ValuationShowcase />,
  },
  {
    module: 'learning/explain',
    title: 'ALE against partial dependence',
    description:
      'Gradient-boosted trees fitted to two strongly correlated features with a known additive target: partial dependence queries combinations off the data and bends away from the true effect, accumulated local effects stay on it; for an additive model with an interaction, Friedman’s H-statistics and the functional ANOVA (variance shares and purified main effects) of GAMs with and without a tensor term.',
    tags: [
      'showcase',
      'accumulated local effects',
      'partial dependence',
      'H-statistic',
      'functional ANOVA',
      'Sobol indices',
      'GAM',
      'interactions',
    ],
    render: () => <EffectsShowcase />,
  },
  {
    module: 'learning/explain',
    title: 'Concept activation vectors (TCAV)',
    description:
      'An MLP trained in the browser on images built from stripes, a corner dot and a bar, labelled by a rule over them: concept activation vectors from linear probes on a hidden layer, TCAV scores of class 1 per concept against several random sets, and a t-test against random-against-random directions.',
    tags: ['showcase', 'TCAV', 'concept activation vectors', 'linear probe', 'concept-based explanations'],
    render: () => <ConceptsShowcase />,
  },
]
