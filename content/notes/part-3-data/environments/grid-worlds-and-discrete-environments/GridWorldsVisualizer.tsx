import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

export function GridWorldsVisualizer() {
  const state = useFigureState({
    environment: choice<'cliff_walking' | 'frozen_lake' | 'taxi'>(
      [
        { value: 'cliff_walking', label: 'Cliff Walking (4x12): SARSA vs Q-learning risk' },
        { value: 'frozen_lake', label: 'Frozen Lake (4x4): Stochastic slippery transitions' },
        { value: 'taxi', label: 'Taxi-v3 (5x5): Hierarchical state space (500 states)' },
      ],
      'cliff_walking',
      { label: 'discrete grid environment' },
    ),
    policyMode: choice<'optimal_edge' | 'safe_path' | 'exploratory'>(
      [
        { value: 'optimal_edge', label: 'Q-learning greedy path (edge-hugging / risk)' },
        { value: 'safe_path', label: 'SARSA conservative path (buffered safety)' },
        { value: 'exploratory', label: 'Epsilon-greedy exploratory walk' },
      ],
      'safe_path',
      { label: 'agent policy' },
    ),
    stochasticity: slider(0.0, 0.4, 0.1, { step: 0.05, label: 'transition slip probability' }),
  })

  const results = useMemo(() => {
    const rng = stream(33)
    const pathX: number[] = []
    const pathY: number[] = []
    const hazardX: number[] = []
    const hazardY: number[] = []

    let stateCount = 48
    let actionCount = 4
    let returnScore = 0
    let stepCount = 0
    let hazardFalls = 0

    if (state.environment === 'cliff_walking') {
      stateCount = 48 // 4 rows x 12 cols
      actionCount = 4 // Up, Down, Left, Right

      // Cliff is at y = 0, x = 1 to 10
      for (let x = 1; x <= 10; x++) {
        hazardX.push(x)
        hazardY.push(0)
      }

      let curX = 0
      let curY = 0
      pathX.push(curX)
      pathY.push(curY)

      if (state.policyMode === 'optimal_edge') {
        // Optimal greedy hugs y = 1 right above the cliff
        // Start (0, 0) -> Up (0, 1) -> Right 11 times -> Down (11, 0)
        const waypoints = [
          [0, 1],
          [1, 1],
          [2, 1],
          [3, 1],
          [4, 1],
          [5, 1],
          [6, 1],
          [7, 1],
          [8, 1],
          [9, 1],
          [10, 1],
          [11, 1],
          [11, 0],
        ]
        for (const [wx, wy] of waypoints) {
          stepCount++
          returnScore -= 1
          let actualX = wx
          let actualY = wy
          // Stochastic slip check
          if (normal(rng, 0, 1) < state.stochasticity * 2 - 1 && wy === 1 && wx > 0 && wx < 11) {
            actualY = 0 // Slipped into cliff!
            returnScore -= 100
            hazardFalls++
          }
          pathX.push(actualX)
          pathY.push(actualY)
          if (actualY === 0 && actualX > 0 && actualX < 11) {
            // Reset to start on cliff fall
            pathX.push(0)
            pathY.push(0)
          }
        }
      } else if (state.policyMode === 'safe_path') {
        // SARSA safe path walks along top row y = 3
        const waypoints = [
          [0, 1],
          [0, 2],
          [0, 3],
          [1, 3],
          [2, 3],
          [3, 3],
          [4, 3],
          [5, 3],
          [6, 3],
          [7, 3],
          [8, 3],
          [9, 3],
          [10, 3],
          [11, 3],
          [11, 2],
          [11, 1],
          [11, 0],
        ]
        for (const [wx, wy] of waypoints) {
          stepCount++
          returnScore -= 1
          pathX.push(wx)
          pathY.push(wy)
        }
      } else {
        // Exploratory walk
        for (let s = 0; s < 25; s++) {
          stepCount++
          returnScore -= 1
          curX = Math.max(0, Math.min(11, curX + Math.round(normal(rng, 0.4, 0.8))))
          curY = Math.max(0, Math.min(3, curY + Math.round(normal(rng, 0.2, 0.7))))
          if (curY === 0 && curX > 0 && curX < 11) {
            returnScore -= 100
            hazardFalls++
            curX = 0
            curY = 0
          }
          pathX.push(curX)
          pathY.push(curY)
          if (curX === 11 && curY === 0) break
        }
      }
    } else if (state.environment === 'frozen_lake') {
      stateCount = 16 // 4x4
      actionCount = 4
      // Holes at (1, 1), (3, 1), (3, 2), (0, 3)
      hazardX.push(1, 3, 3, 0)
      hazardY.push(1, 1, 2, 3)

      const waypoints = [
        [0, 0],
        [1, 0],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 3],
      ]
      for (const [wx, wy] of waypoints) {
        stepCount++
        pathX.push(wx)
        pathY.push(wy)
      }
      returnScore = pathX[pathX.length - 1] === 3 && pathY[pathY.length - 1] === 3 ? 1.0 : 0.0
    } else {
      // Taxi (5x5 grid with passenger pickup and dropoff)
      stateCount = 500 // 25 locations x 5 passenger locations x 4 destinations
      actionCount = 6 // South, North, East, West, Pickup, Dropoff
      // Obstacle walls represented as hazards
      hazardX.push(1.5, 1.5, 3.5, 3.5)
      hazardY.push(1, 2, 3, 4)

      const taxiPath = [
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 2],
        [2, 2],
        [3, 2],
        [4, 2],
        [4, 3],
        [4, 4],
      ]
      for (const [tx, ty] of taxiPath) {
        stepCount++
        returnScore -= 1
        pathX.push(tx)
        pathY.push(ty)
      }
      returnScore += 20 // Successful delivery reward
    }

    return {
      pathX,
      pathY,
      hazardX,
      hazardY,
      stateCount,
      actionCount,
      stepCount,
      returnScore,
      hazardFalls,
    }
  }, [state.environment, state.policyMode, state.stochasticity])

  const xDomain = state.environment === 'cliff_walking' ? 12 : state.environment === 'frozen_lake' ? 4 : 5
  const yDomain = state.environment === 'cliff_walking' ? 4 : state.environment === 'frozen_lake' ? 4 : 5

  const xAxis = useAxis({
    label: 'discrete grid column coordinate X',
    range: [-0.5, xDomain - 0.5],
  })
  const yAxis = useAxis({
    label: 'discrete grid row coordinate Y',
    range: [-0.5, yDomain - 0.5],
  })

  return (
    <Figure
      title="Discrete grid worlds: state cardinality, stochastic transitions, and path risk"
      state={state}
      hoverReadout={false}
      caption="Evaluating discrete reinforcement learning benchmarks. In Cliff Walking (4x12), Q-learning learns the optimal dangerous path hugging the cliff edge (13 steps), which suffers catastrophic penalties (-100) under stochastic slips. In contrast, SARSA incorporates policy exploration into its value updates, discovering the longer but safe path along the upper perimeter."
      readouts={
        <>
          <Readout label="state space cardinality |S|" value={`${results.stateCount} states`} />
          <Readout label="action space cardinality |A|" value={`${results.actionCount} actions`} />
          <Readout label="episode trajectory steps" value={`${results.stepCount} steps`} />
          <Readout label="cumulative episode return" value={results.returnScore.toFixed(0)} />
          <Readout
            label="cliff-fall / penalty events"
            value={results.hazardFalls > 0 ? `${results.hazardFalls} falls (-100 penalty)` : 'Zero falls (Safe)'}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Grid map, hazard locations, and agent trajectory">
        <Points name="hazard / cliff barrier cells" x={results.hazardX} y={results.hazardY} slot={1} size={7} />
        <Curve name="agent policy trajectory" x={results.pathX} y={results.pathY} slot={0} width={2.5} />
        <Points name="visited state coordinates" x={results.pathX} y={results.pathY} slot={0} size={4} />
      </Plot>
    </Figure>
  )
}
