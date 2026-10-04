import type { Specimen } from '@lab/specimen'
import { LabelModelShowcase } from './_weak-supervision/label-models'
import { ActiveCurvesFigure, ActiveRunFigure } from './_weak-supervision/active'
import { GammaHeuristicFigure, LlpComparisonFigure, LpllpShowcase } from './_weak-supervision/lpllp'

export const specimens: Specimen[] = [
  {
    module: 'part-5-learning-paradigms/weak-supervision',
    title: 'Showcase: weak supervision, from noisy labelling functions to labels',
    description:
      'Labelling functions with known accuracies vote on unlabelled data; majority vote, Dawid–Skene EM played step by step and the data-programming label model of Snorkel estimate each function’s accuracy from the votes alone and label the data, compared with the hidden truth.',
    tags: [
      'showcase',
      'weak supervision',
      'labelling functions',
      'Snorkel',
      'data programming',
      'Dawid–Skene',
      'majority vote',
      'labelModelReport',
    ],
    render: () => <LabelModelShowcase />,
  },
  {
    module: 'part-5-learning-paradigms/weak-supervision',
    title: 'Showcase: label propagation for learning with label proportions',
    description:
      'Poyiadzi, Santos-Rodriguez and Twomey (2018): bags reveal only their class proportions; LP-LLP spreads each bag’s proportion over a similarity graph and projects back onto the bags’ class masses, played half-step by half-step, with bags to pin and points to move between them, γ chosen by smoothness, and the accuracy against InvCal, alter-∝SVM, MeanMap and the proportion loss as bag size and purity vary.',
    tags: [
      'showcase',
      'learning from label proportions',
      'label propagation',
      'LP-LLP',
      'alternating projections',
      'InvCal',
      '∝SVM',
      'MeanMap',
      'lpllpSteps',
      'llpComparison',
      'bagsByProportion',
    ],
    render: () => (
      <>
        <LpllpShowcase />
        <GammaHeuristicFigure />
        <LlpComparisonFigure />
      </>
    ),
  },
  {
    module: 'part-5-learning-paradigms/weak-supervision',
    title: 'Showcase: active learning with label proportions',
    description:
      'Poyiadzis, Santos-Rodriguez and Twomey (2019): the learner builds bags from a pool and an LLP-oracle answers with each bag’s class proportion; US-LP, US-Mass, random bags and an exact oracle, played query by query on LP-LLP, and the accuracy curves of the paper’s experiment.',
    tags: [
      'showcase',
      'active learning',
      'learning from label proportions',
      'uncertainty sampling',
      'LLP-oracle',
      'activeProportionsRun',
      'activeProportionsCurves',
    ],
    render: () => (
      <>
        <ActiveRunFigure />
        <ActiveCurvesFigure />
      </>
    ),
  },
]
