/**
 * `aifn/pgm`: probabilistic graphical models (plan §7): a model description language, the structures derived from
 * it, and inference engines built by construction, each a traceable `Algorithm`.
 *
 * - Description: `model`, `dist` (Normal, Bernoulli, Categorical, Binomial, Poisson, Beta, Gamma, Dirichlet, realised
 *   by `aifn/distributions`), plates with fixed or ragged sizes, constants, deterministic nodes, `ref.at(selector)`.
 *   `expandModel` unrolls plates; `logJoint`, `sampleModel` (ancestral), `nestedValues`.
 * - Structure: `toFactorGraph`, `toDiscreteFactorGraph` (data clamped), `markovBlanket`, and diagram specs for the
 *   lab: `toPlateDiagram`, `toFactorDiagram` (with a highlighted Markov blanket).
 * - Discrete factor graphs: `discreteFactor`, `discreteFactorGraph`, the factor algebra (`factorProduct`,
 *   `factorMarginalise`, `factorReduce`, `normaliseFactor`), `bipartiteGraph` (`aifn/graph`), `isTree`,
 *   `isingModel`, `gridEdges`.
 * - Exact: `enumerationSteps` / `enumerate`, `jointDistribution`, `variableEliminationSteps` /
 *   `variableElimination` (fixed, min-degree or min-fill order).
 * - Message passing: `beliefPropagationSteps` / `beliefPropagation` (sum- and max-product; tree, flooding,
 *   sequential or explicit schedules; damping; messages exposed per step), `factorBeliefs`, `betheLogZ`,
 *   `decodeBeliefs`; `gaussianBeliefPropagationSteps` / `gaussianBeliefPropagation`.
 * - Chains: `hmm`, `dishonestCasino`, `forwardBackward` (scaled), `viterbi`, `forwardBackwardSteps`,
 *   `viterbiSteps`, `sampleHmm`, `sampleHiddenPath` (FFBS), `chainForwardBackward` and `chainViterbi` on
 *   log-potentials; the linear-chain CRF: `linearChainCrf`, `crfPotentials`, `crfMarginals`, `crfViterbi`,
 *   `crfScore`, `crfLogLikelihood`, `crfGradient`.
 * - Sampling: `factorGraphGibbsSteps`, `gibbsMarginals`, and `gibbsSteps` on a model (enumerated discrete
 *   conditionals, conjugate Beta, Dirichlet, Normal and Gamma updates).
 * - Engines: `infer`, `registerEngine`, `registeredEngines`; LDA as the registered example: `ldaModel`, `matchLda`,
 *   `ldaCollapsedGibbsSteps`, `ldaEstimates`, `ldaEngine`.
 */

export {
  bipartiteGraph,
  discreteFactor,
  discreteFactorGraph,
  factorGraphEdges,
  factorGraphNeighbours,
  factorMarginalise,
  factorProduct,
  factorProductAll,
  factorReduce,
  forEachAssignment,
  gridEdges,
  isingModel,
  isTree,
  logPotential,
  normaliseFactor,
  variableName,
  type DiscreteFactor,
  type DiscreteFactorGraph,
  type FactorGraphEdge,
} from './factors'
export {
  eliminationResult,
  enumerate,
  enumerationSteps,
  jointDistribution,
  variableElimination,
  variableEliminationSteps,
  type EliminationEvent,
  type EliminationOptions,
  type EliminationOrder,
  type EliminationResult,
  type EliminationState,
  type EnumerationOptions,
  type EnumerationState,
  type ExactResult,
} from './exact'
export {
  beliefPropagation,
  beliefPropagationSteps,
  betheLogZ,
  decodeBeliefs,
  factorBeliefs,
  type BeliefPropagationOptions,
  type BeliefPropagationResult,
  type BeliefPropagationState,
  type MessageUpdate,
  type Schedule,
} from './bp'
export {
  gaussianBeliefPropagation,
  gaussianBeliefPropagationSteps,
  type GaussianBpOptions,
  type GaussianBpState,
} from './gaussianBp'
export {
  chainForwardBackward,
  chainViterbi,
  dishonestCasino,
  forwardBackward,
  forwardBackwardSteps,
  hmm,
  hmmNodePotentials,
  sampleHiddenPath,
  sampleHmm,
  viterbi,
  viterbiSteps,
  type ChainMarginals,
  type ChainStepOptions,
  type ForwardBackwardResult,
  type ForwardBackwardState,
  type Hmm,
  type ViterbiResult,
  type ViterbiState,
} from './chain'
export {
  crfGradient,
  crfLogLikelihood,
  crfMarginals,
  crfPotentials,
  crfScore,
  crfViterbi,
  linearChainCrf,
  type CrfGradient,
  type LinearChainCrf,
} from './crf'
export {
  argValue,
  cardinalityOf,
  conditionalOf,
  dependencyMaps,
  dist,
  environment,
  evaluateOp,
  expandModel,
  instanceKey,
  instanceLogDensity,
  logJoint,
  model,
  modelMarkovBlanket,
  nestedValues,
  plateChain,
  realise,
  resolveRef,
  sampleModel,
  stochasticParents,
  type Arg,
  type Bindings,
  type DeterministicOp,
  type DistSpec,
  type Env,
  type ExpandedModel,
  type Family,
  type Instance,
  type MarkovBlanket,
  type Model,
  type ModelBuilder,
  type ModelNode,
  type ModelScope,
  type Nested,
  type NodeHandle,
  type NodeOptions,
  type NodeRef,
  type NodeValue,
  type Plate,
  type SizeRef,
} from './model'
export {
  markovBlanket,
  toDiscreteFactorGraph,
  toFactorDiagram,
  toFactorGraph,
  toPlateDiagram,
  type DiagramData,
  type DiagramEdgeData,
  type DiagramGroupData,
  type DiagramNodeData,
  type FactorDiagramOptions,
  type ModelDiscreteGraph,
  type ModelFactor,
  type ModelFactorGraph,
  type ModelVariable,
  type PlateDiagramOptions,
} from './structure'
export {
  factorGraphGibbsSteps,
  gibbsMarginals,
  gibbsSteps,
  type ConditionalKind,
  type FactorGraphGibbsOptions,
  type FactorGraphGibbsState,
  type GibbsOptions,
  type GibbsState,
} from './gibbs'
export {
  infer,
  ldaCollapsedGibbsSteps,
  ldaEngine,
  ldaEstimates,
  ldaModel,
  matchLda,
  registerEngine,
  registeredEngines,
  type BuiltInEngine,
  type EngineRegistration,
  type Inference,
  type LdaOptions,
  type LdaState,
} from './engines'
