/**
 * `aifn/dynamics/control`: state feedback for linear plants x′ = Ax + Bu: the linear-quadratic regulator (`lqr`,
 * `dlqr`, on the Riccati solvers of `aifn/numerics/linalg`) and pole placement by Ackermann's formula (`ackermann`).
 */

export { closedLoopPoles, dlqr, lqr, type LqrResult, type StateFeedbackPlant } from './lqr'
export { ackermann, type PolePlacement } from './ackermann'
