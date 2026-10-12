import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function TrajectoryDataVisualizer() {
  const state = useFigureState({
    mode: choice<'hmm_map_matching' | 'douglas_peucker' | 'stay_point_detection'>(
      [
        {
          value: 'hmm_map_matching',
          label: 'HMM Map Matching: Emission Noise vs Network Route Transition (Newson 2009)',
        },
        {
          value: 'douglas_peucker',
          label: 'Trajectory Simplification: Douglas-Peucker Epsilon vs Point Retention (Zheng 2015)',
        },
        {
          value: 'stay_point_detection',
          label: 'Stay-Point Discovery: Spatial Radius D_max vs Dwell Time T_min',
        },
      ],
      'hmm_map_matching',
      { label: 'trajectory processing analysis' },
    ),
    param1: slider(5, 50, 20, {
      step: 5,
      label: 'GPS measurement noise sigma_z (m) / tolerance epsilon (m) / radius D_max (m)',
    }),
    param2: slider(10, 100, 40, {
      step: 10,
      label: 'network discrepancy scale beta (m) / path length / min duration T_min (min)',
    }),
  })

  const { mode, param1, param2 } = state

  const results = useMemo(() => {
    if (mode === 'hmm_map_matching') {
      const sigmaZ = param1 // 5 to 50 m
      const beta = param2 // 10 to 100 m

      // Plot distance from road candidate (0 to 100 m) vs emission probability
      const dists = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
      const emissionProbs: number[] = []
      const transitionProbs: number[] = []

      for (const d of dists) {
        // Gaussian emission: exp(-d^2 / (2 * sigmaZ^2))
        const pEmit = Math.exp(-Math.pow(d, 2) / (2 * Math.pow(sigmaZ, 2)))
        // Exponential transition: exp(-d / beta)
        const pTrans = Math.exp(-d / beta)

        emissionProbs.push(pEmit * 100)
        transitionProbs.push(pTrans * 100)
      }

      const activeEmit = Math.exp(-Math.pow(25, 2) / (2 * Math.pow(sigmaZ, 2))) * 100
      const activeTrans = Math.exp(-25 / beta) * 100

      return {
        x: dists,
        y1: emissionProbs,
        y2: transitionProbs,
        stat1: `GPS σ_z = ${sigmaZ} m, Route β = ${beta} m`,
        stat2: `Emission P(25m) = ${activeEmit.toFixed(1)}%`,
        stat3: `Transition P(25m) = ${activeTrans.toFixed(1)}%`,
        insight: 'HMM balances GPS sensor distance against topological street network connectivity',
        axisXLabel: 'Perpendicular Distance / Route Discrepancy (m)',
        axisYLabel: 'HMM Probability Density (% of peak)',
        xRange: [-2, 105] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else if (mode === 'douglas_peucker') {
      const epsilon = param1 // 5 to 50 m tolerance
      const epsilons = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50]
      const retainedPct: number[] = []
      const maxErrors: number[] = []

      for (const eps of epsilons) {
        // As tolerance grows, retained vertices drop exponentially
        const retain = Math.max(4, 100 * Math.exp(-0.06 * eps))
        retainedPct.push(retain)
        maxErrors.push(eps)
      }

      const activeRetain = Math.max(4, 100 * Math.exp(-0.06 * epsilon))

      return {
        x: epsilons,
        y1: retainedPct,
        y2: maxErrors,
        stat1: `Tolerance ε = ${epsilon} m`,
        stat2: `Points retained: ${activeRetain.toFixed(1)}%`,
        stat3: `Compression ratio: ${(100 / activeRetain).toFixed(1)}x`,
        insight: 'Douglas-Peucker compresses linear vertices while guaranteeing max perpendicular deviation ≤ ε',
        axisXLabel: 'Simplification Tolerance ε (m)',
        axisYLabel: 'Points Retained (%) & Deviation Bound (m)',
        xRange: [0, 55] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else {
      // Stay point detection across duration (1 to 60 min)
      const durations = [2, 5, 10, 15, 20, 25, 30, 40, 50, 60]
      const dwellProb: number[] = []
      const jitterRadius: number[] = []

      const dMax = param1 * 1.5 // 7.5 to 75 m
      const tMin = param2 * 0.5 // 5 to 50 min

      for (const t of durations) {
        // Probability of meeting stay-point condition: sigmoid around tMin
        const pStay = 100 / (1 + Math.exp(-0.2 * (t - tMin)))
        dwellProb.push(pStay)
        jitterRadius.push(Math.min(dMax, 12 + 0.6 * t))
      }

      return {
        x: durations,
        y1: dwellProb,
        y2: jitterRadius,
        stat1: `D_max = ${dMax.toFixed(0)} m, T_min = ${tMin.toFixed(0)} min`,
        stat2: 'Stay points compress dwelling episodes',
        stat3: 'Filters GPS urban multipath jitter',
        insight: 'Discovers semantic places of interest (work, home, stops) from raw kinematic streams',
        axisXLabel: 'Episode Dwell Time (minutes)',
        axisYLabel: 'Stay Point Confidence (%) & Spatial Radius (m)',
        xRange: [0, 65] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    }
  }, [mode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Trajectory Data: HMM Map Matching, Douglas-Peucker Simplification, and Stay-Point Discovery"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Pipeline Setting" value={results.stat1} />
          <Readout label="Primary Metric" value={results.stat2} />
          <Readout label="Performance / Scale" value={results.stat3} />
          <Readout label="Kinematic Insight" value={results.insight} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Trajectory Transformation Dynamics and Algorithmic Responses">
        <Curve
          name={
            mode === 'hmm_map_matching'
              ? 'HMM Emission Probability p(z|r) (%)'
              : mode === 'douglas_peucker'
                ? 'Retained Trajectory Points (%)'
                : 'Stay-Point Discovery Confidence (%)'
          }
          x={results.x}
          y={results.y1}
          slot={0}
          width={2.5}
        />
        <Curve
          name={
            mode === 'hmm_map_matching'
              ? 'HMM Route Transition Probability p(r_k|r_j) (%)'
              : mode === 'douglas_peucker'
                ? 'Maximum Perpendicular Distance Error Bound (m)'
                : 'Observed Spatial Wandering Envelope (m)'
          }
          x={results.x}
          y={results.y2}
          slot={1}
          dashed
          width={2.0}
        />
        <Points name="Evaluated Regimes" x={results.x} y={results.y1} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
