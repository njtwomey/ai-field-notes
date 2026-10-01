/**
 * `aifn-applied/gym/environments/control`: classic-control environments with continuous observations and
 * differentiable dynamics models, stepped on core's ODE solvers. The inverted pendulum (Gymnasium's `Pendulum-v1`).
 */

export {
  pendulumEnergy,
  pendulumEnvironment,
  wrapAngle,
  type PendulumAction,
  type PendulumEnvironment,
  type PendulumOptions,
  type PendulumParameters,
  type PendulumState,
} from './pendulum'
export {
  cartPoleEnvironment,
  type CartPoleEnvironment,
  type CartPoleOptions,
  type CartPoleParameters,
  type CartPoleState,
} from './cartpole'
