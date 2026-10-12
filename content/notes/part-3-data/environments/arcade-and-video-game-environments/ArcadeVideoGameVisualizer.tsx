import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ArcadeVideoGameVisualizer() {
  const state = useFigureState({
    pipelineStage: choice<'raw_rgb' | 'downsampled_gray' | 'frame_stacked'>(
      [
        { value: 'frame_stacked', label: 'Stacked frames [4, 84, 84] (Resolves velocity & momentum)' },
        { value: 'downsampled_gray', label: 'Single 84x84 grayscale (Velocity ambiguous)' },
        { value: 'raw_rgb', label: 'Raw Atari 210x160x3 RGB (High bandwidth)' },
      ],
      'frame_stacked',
      { label: 'observation pipeline' },
    ),
    game: choice<'breakout' | 'pong' | 'space_invaders'>(
      [
        { value: 'breakout', label: 'Breakout (ball velocity and brick collisions)' },
        { value: 'pong', label: 'Pong (ball trajectory and paddle intercept)' },
        { value: 'space_invaders', label: 'Space Invaders (fleet descent and laser timing)' },
      ],
      'breakout',
      { label: 'Atari benchmark' },
    ),
    stickyActionProb: slider(0.0, 0.5, 0.25, {
      step: 0.05,
      label: 'sticky action execution stochasticity',
    }),
  })

  const results = useMemo(() => {
    const trailX: number[] = []
    const trailY: number[] = []
    const paddleX: number[] = []
    const paddleY: number[] = []
    const brickX: number[] = []
    const brickY: number[] = []

    // Coordinates scaled to 84x84 grid
    if (state.game === 'breakout') {
      // Bricks at top row y in [70, 78]
      for (let bx = 10; bx <= 74; bx += 8) {
        brickX.push(bx)
        brickY.push(75)
        brickX.push(bx)
        brickY.push(70)
      }
      // Paddle at bottom y = 10
      for (let px = 34; px <= 50; px += 4) {
        paddleX.push(px)
        paddleY.push(10)
      }
      // Ball historical trail (last 4 frames)
      const ballBaseX = 46
      const ballBaseY = 42
      const vx = 3.2
      const vy = 4.0
      for (let k = 3; k >= 0; k--) {
        trailX.push(ballBaseX - k * vx)
        trailY.push(ballBaseY - k * vy)
      }
    } else if (state.game === 'pong') {
      // Left paddle at x = 8, right paddle at x = 76
      for (let py = 30; py <= 50; py += 5) {
        paddleX.push(8, 76)
        paddleY.push(py, py + 4)
      }
      // Ball trail
      for (let k = 3; k >= 0; k--) {
        trailX.push(42 - k * 5.0)
        trailY.push(38 + k * 2.5)
      }
    } else {
      // Space Invaders
      for (let r = 50; r <= 70; r += 8) {
        for (let c = 16; c <= 68; c += 10) {
          brickX.push(c)
          brickY.push(r)
        }
      }
      // Player turret at bottom
      for (let px = 38; px <= 46; px += 2) {
        paddleX.push(px)
        paddleY.push(8)
      }
      // Laser projectile trail
      for (let k = 3; k >= 0; k--) {
        trailX.push(42)
        trailY.push(30 - k * 6.0)
      }
    }

    let inputBytes = 100800 // 210 x 160 x 3
    let tensorShape = '[210, 160, 3]'
    let velocityObservable = 'No (Single static frame)'

    if (state.pipelineStage === 'downsampled_gray') {
      inputBytes = 7056 // 84 x 84 x 1
      tensorShape = '[84, 84, 1]'
      velocityObservable = 'No (Sign of dy/dt unknown)'
    } else if (state.pipelineStage === 'frame_stacked') {
      inputBytes = 28224 // 4 x 84 x 84
      tensorShape = '[4, 84, 84]'
      velocityObservable = 'YES (Optical flow captured in tensor channels)'
    }

    return {
      trailX,
      trailY,
      paddleX,
      paddleY,
      brickX,
      brickY,
      inputBytes,
      tensorShape,
      velocityObservable,
    }
  }, [state.pipelineStage, state.game, state.stickyActionProb])

  const axis = useAxis({
    label: 'pixel spatial coordinate (84x84 normalized viewport)',
    range: [0, 84],
  })

  return (
    <Figure
      title="Arcade Learning Environment: pixel preprocessing, frame stacking, and POMDP dynamics"
      state={state}
      hoverReadout={false}
      caption="Evaluating visual reinforcement learning in the Arcade Learning Environment (ALE). A single raw frame creates a Partially Observable MDP (POMDP) because instantaneous velocity and projectile direction cannot be inferred from a single static image. Stacking k=4 consecutive downsampled 84x84 frames resolves direction and acceleration while reducing network bandwidth by 72% relative to raw RGB video."
      readouts={
        <>
          <Readout label="observation tensor" value={results.tensorShape} />
          <Readout label="frame size (uncompressed)" value={`${(results.inputBytes / 1024).toFixed(1)} KB`} />
          <Readout label="velocity observability" value={results.velocityObservable} />
          <Readout
            label="sticky action setting"
            value={
              state.stickyActionProb > 0
                ? `${(state.stickyActionProb * 100).toFixed(0)}% repeat prob (Machado 2018)`
                : 'Deterministic (Memorization risk)'
            }
          />
        </>
      }
    >
      <Plot x={axis} y={axis} title="Observation viewport and multi-frame motion history">
        {/* Game objects */}
        {results.brickX.length > 0 && (
          <Points name="target game elements / bricks" x={results.brickX} y={results.brickY} slot={1} size={5} />
        )}
        <Points name="agent controllable avatar / paddle" x={results.paddleX} y={results.paddleY} slot={2} size={5} />

        {/* Temporal stacked trail */}
        {state.pipelineStage === 'frame_stacked' ? (
          <>
            <Curve
              name="temporal optical flow vector"
              x={results.trailX}
              y={results.trailY}
              slot={0}
              dashed
              width={2}
            />
            <Points
              name="stacked frame ball positions (t-3 to t)"
              x={results.trailX}
              y={results.trailY}
              slot={0}
              size={6}
            />
          </>
        ) : (
          <Points
            name="instantaneous ball position (t)"
            x={[results.trailX[results.trailX.length - 1]]}
            y={[results.trailY[results.trailY.length - 1]]}
            slot={0}
            size={6}
          />
        )}
      </Plot>
    </Figure>
  )
}
