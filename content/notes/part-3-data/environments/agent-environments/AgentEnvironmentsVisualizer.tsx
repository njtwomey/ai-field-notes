import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function AgentEnvironmentsVisualizer() {
  const state = useFigureState({
    environmentType: choice<'web_arena' | 'os_world' | 'alfworld'>(
      [
        { value: 'web_arena', label: 'WebArena: Browser DOM tree & accessibility actions' },
        { value: 'os_world', label: 'OSWorld: Desktop OS, bash shell & filesystem tools' },
        { value: 'alfworld', label: 'ALFWorld: Embodied text-guided household tasks' },
      ],
      'web_arena',
      { label: 'agent benchmark' },
    ),
    agentStrategy: choice<'grounded_react' | 'direct_plan'>(
      [
        { value: 'grounded_react', label: 'ReAct loop (Observe -> Reason -> Act with error recovery)' },
        { value: 'direct_plan', label: 'Open-loop planning (fails on unexpected DOM changes)' },
      ],
      'grounded_react',
      { label: 'agent execution strategy' },
    ),
    taskHorizon: slider(4, 12, 8, { step: 1, label: 'sub-goal task steps' }),
  })

  const results = useMemo(() => {
    const H = state.taskHorizon
    const stepIndices: number[] = []
    const completionPct: number[] = []

    let currentCompletion = 0
    let toolSuccessCount = 0
    let recoveredErrors = 0
    let finalSuccess = false

    // An environmental perturbation (e.g. cookie popup, missing file, dynamic DOM change) occurs at step 3
    const perturbationStep = 3

    for (let step = 0; step <= H; step++) {
      stepIndices.push(step)

      if (step === 0) {
        completionPct.push(0)
        continue
      }

      if (state.agentStrategy === 'direct_plan') {
        if (step < perturbationStep) {
          currentCompletion += 100 / H
          toolSuccessCount++
        } else if (step === perturbationStep) {
          // Encounters unexpected environment state: plan desynchronizes
          // Model hallucinates next actions without effect
          toolSuccessCount = Math.max(0, toolSuccessCount)
          // Progress halts
        }
      } else {
        // Grounded ReAct
        if (step === perturbationStep) {
          // Detects unexpected modal/error, executes self-correction action
          recoveredErrors++
          currentCompletion += 50 / H // Partial recovery progress
          toolSuccessCount++
        } else {
          currentCompletion = Math.min(100, currentCompletion + 100 / H)
          toolSuccessCount++
        }
      }

      completionPct.push(Math.min(100, currentCompletion))
    }

    finalSuccess = currentCompletion >= 95

    return {
      stepIndices,
      completionPct,
      toolSuccessCount,
      recoveredErrors,
      finalSuccess,
      finalScore: currentCompletion,
    }
  }, [state.environmentType, state.agentStrategy, state.taskHorizon])

  const xAxis = useAxis({
    label: 'interaction step in environment (actions & tool calls)',
    range: [0, state.taskHorizon],
  })
  const yAxis = useAxis({
    label: 'verified task completion percentage (%)',
    range: [0, 105],
  })

  return (
    <Figure
      title="Web, tool-use, and embodied agent environments: grounding, error recovery, and task horizons"
      state={state}
      hoverReadout={false}
      caption="Simulating multi-step autonomous agent execution in WebArena, OSWorld, and ALFWorld. Open-loop planning fails when real environments present unexpected state transitions (e.g. dynamic cookie banners, missing bash dependencies, or moved household items), halting progress. Closed-loop ReAct agents parse observations at each step, ground actions in live DOM/system state, and execute recovery sub-plans to achieve verified task completion."
      readouts={
        <>
          <Readout label="task completion" value={`${results.finalScore.toFixed(0)}%`} />
          <Readout label="successful tool calls" value={`${results.toolSuccessCount} / ${state.taskHorizon}`} />
          <Readout label="environmental recoveries" value={`${results.recoveredErrors} error corrections`} />
          <Readout
            label="evaluation status"
            value={results.finalSuccess ? 'Task Completed & Verified' : 'Failed / Desynchronized'}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Step-by-step progress towards task completion">
        <Curve
          name="task completion trajectory"
          x={results.stepIndices}
          y={results.completionPct}
          slot={state.agentStrategy === 'grounded_react' ? 0 : 1}
          width={2.5}
        />
        <Points
          name="environment tool step checkpoints"
          x={results.stepIndices}
          y={results.completionPct}
          slot={state.agentStrategy === 'grounded_react' ? 0 : 1}
          size={5}
        />
      </Plot>
    </Figure>
  )
}
