import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

export function ClassicControlVisualizer() {
  const state = useFigureState({
    env: choice<'cartpole' | 'mountain_car' | 'pendulum'>(
      [
        { value: 'cartpole', label: 'CartPole-v1: Inverted pendulum balancing' },
        { value: 'mountain_car', label: 'MountainCar-v0: Momentum accumulation' },
        { value: 'pendulum', label: 'Pendulum-v1: Non-linear continuous torque swing-up' },
      ],
      'cartpole',
      { label: 'benchmark environment' },
    ),
    policy: choice<'random' | 'heuristic'>(
      [
        { value: 'heuristic', label: 'Stabilizing / Energy-pumping feedback policy' },
        { value: 'random', label: 'Random action policy (uncontrolled baseline)' },
      ],
      'heuristic',
      { label: 'control policy' },
    ),
    simulationSteps: slider(20, 100, 60, { step: 5, label: 'trajectory horizon (steps)' }),
  })

  const results = useMemo(() => {
    const rng = stream(17)
    const steps = state.simulationSteps
    const timeSteps: number[] = []
    const primaryTrajectory: number[] = []
    const secondaryTrajectory: number[] = []

    let reward = 0
    let stateDim = 4
    let actionSpace = 'Discrete(2)'
    let success = false

    if (state.env === 'cartpole') {
      stateDim = 4 // [x, x_dot, theta, theta_dot]
      actionSpace = 'Discrete(2): {Left, Right}'

      let x = 0
      let theta = 0.03
      let xDot = 0
      let thetaDot = 0

      for (let t = 0; t < steps; t++) {
        timeSteps.push(t)
        primaryTrajectory.push(theta * (180 / Math.PI)) // angle in degrees
        secondaryTrajectory.push(x) // cart position

        if (Math.abs(theta) > 0.209 || Math.abs(x) > 2.4) {
          // Fallen over
          break
        }
        reward += 1.0

        // Step physics
        let force = 0
        if (state.policy === 'heuristic') {
          // PD feedback: push toward falling direction
          force = theta + 0.1 * thetaDot > 0 ? 10.0 : -10.0
        } else {
          force = normal(rng, 0, 1) > 0 ? 10.0 : -10.0
        }

        const costheta = Math.cos(theta)
        const sintheta = Math.sin(theta)
        const temp = (force + 0.05 * thetaDot * thetaDot * sintheta) / 1.1
        const thetaAcc = (9.8 * sintheta - costheta * temp) / (0.5 * (4.0 / 3.0 - (0.1 * costheta * costheta) / 1.1))
        const xAcc = temp - (0.05 * thetaAcc * costheta) / 1.1

        x += 0.02 * xDot
        xDot += 0.02 * xAcc
        theta += 0.02 * thetaDot
        thetaDot += 0.02 * thetaAcc
      }

      success = reward >= steps
    } else if (state.env === 'mountain_car') {
      stateDim = 2 // [position x, velocity v]
      actionSpace = 'Discrete(3): {Left, None, Right}'

      let pos = -0.5
      let vel = 0

      for (let t = 0; t < steps; t++) {
        timeSteps.push(t)
        primaryTrajectory.push(pos)
        secondaryTrajectory.push(vel * 10)

        if (pos >= 0.5) {
          success = true
          reward = -(t + 1)
          break
        }
        reward -= 1.0

        let act = 0 // -1, 0, 1
        if (state.policy === 'heuristic') {
          // Energy pumping: push in direction of velocity
          act = vel >= 0 ? 1 : -1
        } else {
          act = (Math.floor(normal(rng, 1, 1)) % 3) - 1
        }

        vel += (act - 1) * 0.001 - 0.0025 * Math.cos(3 * pos)
        vel = Math.max(-0.07, Math.min(0.07, vel))
        pos += vel
        pos = Math.max(-1.2, Math.min(0.6, pos))
        if (pos === -1.2 && vel < 0) vel = 0
      }
    } else {
      // Inverted Pendulum
      stateDim = 3 // [cos(theta), sin(theta), theta_dot]
      actionSpace = 'Box(-2.0, 2.0, (1,)): Continuous Torque'

      let th = Math.PI - 0.2 // started near bottom
      let thDot = 0

      for (let t = 0; t < steps; t++) {
        timeSteps.push(t)
        // Normalize angle to [-pi, pi]
        const normalizedTh = ((th + Math.PI) % (2 * Math.PI)) - Math.PI
        primaryTrajectory.push(normalizedTh)
        secondaryTrajectory.push(thDot)

        const stepCost = normalizedTh * normalizedTh + 0.1 * thDot * thDot
        reward -= stepCost

        let u = 0
        if (state.policy === 'heuristic') {
          // Energy shaping swing-up controller
          const energy = 0.5 * 0.33 * thDot * thDot - 9.8 * Math.cos(th)
          const targetEnergy = 9.8
          u = Math.max(-2.0, Math.min(2.0, -1.5 * (energy - targetEnergy) * thDot))
        } else {
          u = Math.max(-2.0, Math.min(2.0, normal(rng, 0, 1.2)))
        }

        const thAcc = ((-3 * 9.8) / 2) * Math.sin(th + Math.PI) + (3 / (1 * 1)) * u
        th += 0.05 * thDot
        thDot += 0.05 * thAcc
        thDot = Math.max(-8.0, Math.min(8.0, thDot))
      }

      success = Math.abs(primaryTrajectory[primaryTrajectory.length - 1]) < 0.3
    }

    // Reference zero baseline
    const zeroLineX = [0, steps]
    const zeroLineY = [0, 0]

    return {
      timeSteps,
      primaryTrajectory,
      secondaryTrajectory,
      zeroLineX,
      zeroLineY,
      reward,
      stateDim,
      actionSpace,
      success,
    }
  }, [state.env, state.policy, state.simulationSteps])

  const yLabel =
    state.env === 'cartpole'
      ? 'pole deflection angle (degrees)'
      : state.env === 'mountain_car'
        ? 'car position x (goal >= 0.5)'
        : 'pendulum angle (rad, upright = 0)'

  const xAxis = useAxis({
    label: 'simulation time step t',
    range: [0, state.simulationSteps],
  })
  const yAxis = useAxis({
    label: yLabel,
    range: state.env === 'cartpole' ? [-15, 15] : state.env === 'mountain_car' ? [-1.3, 0.7] : [-3.5, 3.5],
  })

  return (
    <Figure
      title="Classic control benchmarks: state dynamics, feedback policies, and physical constraints"
      state={state}
      hoverReadout={false}
      caption="Simulating classic control environments from the Gymnasium benchmark suite. In CartPole, the agent must balance an inherently unstable inverted pendulum. In MountainCar, the motor cannot climb the hill directly and must accumulate gravitational potential energy by rocking. In Pendulum, continuous torque must swing up and stabilize non-linear rotational dynamics."
      readouts={
        <>
          <Readout label="state space" value={`${results.stateDim}-dimensional vector`} />
          <Readout label="action space" value={results.actionSpace} />
          <Readout label="episode return" value={results.reward.toFixed(1)} />
          <Readout label="goal achieved" value={results.success ? 'Success (Stabilized)' : 'Failed / Incomplete'} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Trajectory of primary state variable over time">
        <Curve
          name="reference equilibrium baseline"
          x={results.zeroLineX}
          y={results.zeroLineY}
          slot={2}
          dashed
          width={1}
        />
        <Curve
          name="system state trajectory"
          x={results.timeSteps}
          y={results.primaryTrajectory}
          slot={0}
          width={2.5}
        />
        <Points name="discrete step states" x={results.timeSteps} y={results.primaryTrajectory} slot={0} size={3} />
      </Plot>
    </Figure>
  )
}
