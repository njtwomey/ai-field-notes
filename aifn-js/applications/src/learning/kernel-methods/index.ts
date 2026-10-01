/**
 * `aifn-applied/learning/kernel-methods`: kernel methods: support vector machines (SMO, Pegasos, dual coordinate
 * descent, linear SVM), Platt scaling of their scores into probabilities, and the Crammer–Singer multiclass SVM.
 */

export { plattScaling, type PlattOptions, type PlattScaling } from './platt'
export {
  dualCoordinateSteps,
  dualDecision,
  linearSvm,
  pegasosSteps,
  smoSteps,
  supportVectorMachine,
  type LinearSvmModel,
  type LinearSvmProblem,
  type LinearSvmState,
  type SmoProblem,
  type SmoState,
  type SupportVectorMachineModel,
} from './svm'
export {
  type CrammerSingerProblem,
  type CrammerSingerState,
  crammerSingerSteps,
  type CrammerSingerModel,
  crammerSinger,
} from './crammerSinger'
