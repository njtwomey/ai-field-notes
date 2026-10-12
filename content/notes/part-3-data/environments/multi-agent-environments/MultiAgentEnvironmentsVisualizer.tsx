import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

export function MultiAgentEnvironmentsVisualizer() {
  const state = useFigureState({
    scenario: choice<'cooperative_spread' | 'predator_prey'>(
      [
        { value: 'cooperative_spread', label: 'Cooperative Navigation: 3 agents cover 3 distinct landmarks' },
        { value: 'predator_prey', label: 'Predator-Prey (Adversarial): 3 predators encircle 1 agile prey' },
      ],
      'cooperative_spread',
      { label: 'multi-agent task' },
    ),
    coordinationRegime: choice<'independent' | 'centralized_ctde'>(
      [
        { value: 'centralized_ctde', label: 'Centralized Training Decentralized Execution (CTDE / QMIX)' },
        { value: 'independent', label: 'Independent Learners (IQL: non-stationarity / collisions)' },
      ],
      'centralized_ctde',
      { label: 'coordination regime' },
    ),
    collisionPenalty: slider(0.1, 2.0, 1.0, { step: 0.1, label: 'inter-agent collision penalty' }),
  })

  const results = useMemo(() => {
    const rng = stream(66)
    const T = 50

    // Coordinates of landmarks
    const landmarkX = [-1.5, 0.0, 1.5]
    const landmarkY = [1.2, -1.0, 1.2]

    // 3 agent trajectories
    const a0X: number[] = []
    const a0Y: number[] = []
    const a1X: number[] = []
    const a1Y: number[] = []
    const a2X: number[] = []
    const a2Y: number[] = []

    let cur0 = [-1.8, -1.5]
    let cur1 = [0.0, -1.8]
    let cur2 = [1.8, -1.5]

    let collisions = 0
    let teamReward = 0

    for (let t = 0; t <= T; t++) {
      a0X.push(cur0[0])
      a0Y.push(cur0[1])
      a1X.push(cur1[0])
      a1Y.push(cur1[1])
      a2X.push(cur2[0])
      a2Y.push(cur2[1])

      if (state.scenario === 'cooperative_spread') {
        if (state.coordinationRegime === 'centralized_ctde') {
          // Coordinated assignment: agent 0 -> landmark 0, agent 1 -> landmark 1, agent 2 -> landmark 2
          cur0[0] += (landmarkX[0] - cur0[0]) * 0.08
          cur0[1] += (landmarkY[0] - cur0[1]) * 0.08
          cur1[0] += (landmarkX[1] - cur1[0]) * 0.08
          cur1[1] += (landmarkY[1] - cur1[1]) * 0.08
          cur2[0] += (landmarkX[2] - cur2[0]) * 0.08
          cur2[1] += (landmarkY[2] - cur2[1]) * 0.08
        } else {
          // Independent: both agent 0 and agent 1 greedily race to nearest central landmark 1
          cur0[0] += (landmarkX[1] - cur0[0]) * 0.08 + normal(rng, 0, 0.05)
          cur0[1] += (landmarkY[1] - cur0[1]) * 0.08 + normal(rng, 0, 0.05)
          cur1[0] += (landmarkX[1] - cur1[0]) * 0.08 + normal(rng, 0, 0.05)
          cur1[1] += (landmarkY[1] - cur1[1]) * 0.08 + normal(rng, 0, 0.05)
          cur2[0] += (landmarkX[2] - cur2[2]) * 0.08
          cur2[1] += (landmarkY[2] - cur2[1]) * 0.08

          // Check collision between agent 0 and 1
          const dist01 = Math.hypot(cur0[0] - cur1[0], cur0[1] - cur1[1])
          if (dist01 < 0.3) {
            collisions++
            teamReward -= state.collisionPenalty
          }
        }
      } else {
        // Predator-prey scenario
        if (state.coordinationRegime === 'centralized_ctde') {
          // Encirclement pincer movement around origin
          cur0[0] += (-0.6 - cur0[0]) * 0.07
          cur0[1] += (0.6 - cur0[1]) * 0.07
          cur1[0] += (0.6 - cur1[0]) * 0.07
          cur1[1] += (0.6 - cur1[1]) * 0.07
          cur2[0] += (0.0 - cur2[0]) * 0.07
          cur2[1] += (-0.8 - cur2[1]) * 0.07
        } else {
          // Simple chase behind prey without trapping
          cur0[0] += normal(rng, 0.04, 0.08)
          cur0[1] += normal(rng, 0.04, 0.08)
          cur1[0] += normal(rng, 0.02, 0.08)
          cur1[1] += normal(rng, 0.02, 0.08)
          cur2[0] += normal(rng, -0.02, 0.08)
          cur2[1] += normal(rng, 0.02, 0.08)
        }
      }
    }

    if (state.coordinationRegime === 'centralized_ctde') {
      teamReward = state.scenario === 'cooperative_spread' ? 84.5 : 91.0
    } else {
      teamReward = state.scenario === 'cooperative_spread' ? 32.0 - collisions * state.collisionPenalty : 42.0
    }

    return {
      landmarkX,
      landmarkY,
      a0X,
      a0Y,
      a1X,
      a1Y,
      a2X,
      a2Y,
      collisions,
      teamReward,
    }
  }, [state.scenario, state.coordinationRegime, state.collisionPenalty])

  const axis = useAxis({
    label: 'multi-agent continuous 2D arena space',
    range: [-2.5, 2.5],
  })

  return (
    <Figure
      title="Multi-agent reinforcement learning: decentralized execution, credit assignment, and collision avoidance"
      state={state}
      hoverReadout={false}
      caption="Simulating multi-agent coordination in PettingZoo / Multi-Agent Particle Environments. Under Independent Q-Learning (IQL), agents treat others as passive environment noise; multiple agents greedily contest the same landmark, causing collisions and leaving distant targets uncovered. Under Centralized Training with Decentralized Execution (CTDE), agents learn factored joint-action values, achieving conflict-free target partitioning."
      readouts={
        <>
          <Readout label="team episodic return" value={results.teamReward.toFixed(1)} />
          <Readout label="collision count" value={`${results.collisions} collisions`} />
          <Readout
            label="environmental stationarity"
            value={
              state.coordinationRegime === 'centralized_ctde'
                ? 'Stationary (Joint Critic Factorization)'
                : 'Non-Stationary (Independent Moving Targets)'
            }
          />
          <Readout
            label="coordination status"
            value={
              state.coordinationRegime === 'centralized_ctde'
                ? 'Coordinated Partitioning'
                : 'Greedy Congestion / Conflict'
            }
          />
        </>
      }
    >
      <Plot x={axis} y={axis} title="Agent trajectories and landmark destinations in 2D space">
        <Points
          name="target landmark locations (goals)"
          x={results.landmarkX}
          y={results.landmarkY}
          slot={2}
          size={8}
        />
        <Curve name="agent 1 trajectory (blue)" x={results.a0X} y={results.a0Y} slot={0} width={2} />
        <Curve name="agent 2 trajectory (orange)" x={results.a1X} y={results.a1Y} slot={1} width={2} />
        <Curve name="agent 3 trajectory (green)" x={results.a2X} y={results.a2Y} slot={3} width={2} />
        <Points
          name="agent current positions"
          x={[
            results.a0X[results.a0X.length - 1],
            results.a1X[results.a1X.length - 1],
            results.a2X[results.a2X.length - 1],
          ]}
          y={[
            results.a0Y[results.a0Y.length - 1],
            results.a1Y[results.a1Y.length - 1],
            results.a2Y[results.a2Y.length - 1],
          ]}
          slot={0}
          size={6}
        />
      </Plot>
    </Figure>
  )
}
