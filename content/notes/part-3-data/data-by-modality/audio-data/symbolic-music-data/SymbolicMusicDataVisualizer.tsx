import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function SymbolicMusicDataVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'piano_roll_grid' | 'tokenization_density' | 'dtw_tempo_alignment'>(
      [
        {
          value: 'piano_roll_grid',
          label: 'Piano Roll Discretization: Grid Resolution vs Matrix Sparsity (Müller 2015)',
        },
        { value: 'tokenization_density', label: 'Sequence Tokenization: Dense Piano Roll vs Sparse Event Tokens' },
        {
          value: 'dtw_tempo_alignment',
          label: 'Audio-to-Score Alignment: Rubato Performance vs Metronomic Grid (Hawthorne 2018)',
        },
      ],
      'piano_roll_grid',
      { label: 'symbolic representation perspective' },
    ),
    gridStepMs: slider(10, 100, 25, {
      step: 5,
      label: 'piano roll time-step resolution (ms)',
    }),
    polyphonicNotes: slider(1, 8, 4, {
      step: 1,
      label: 'average active notes / polyphony voice count',
    }),
  })

  const { visualizerMode, gridStepMs, polyphonicNotes } = state

  const results = useMemo(() => {
    if (visualizerMode === 'piano_roll_grid') {
      // Step resolutions from 10 ms to 100 ms
      const steps = [10, 20, 25, 30, 40, 50, 60, 80, 100]
      // 128 MIDI pitches. In any time-step, polyphonicNotes pitches are active.
      // Sparsity = 1 - (polyphonicNotes / 128)
      // Jitter error (ms) = step / 2
      const jitterError = steps.map((s) => s / 2)
      const sparsityPercent = steps.map(() => (1 - polyphonicNotes / 128) * 100)

      const currJitter = gridStepMs / 2
      const currSparsity = (1 - polyphonicNotes / 128) * 100

      return {
        x: steps,
        y: jitterError,
        baseline: sparsityPercent,
        stat1: `Resolution: ${gridStepMs} ms / frame`,
        stat2: `Timing jitter: ±${currJitter.toFixed(1)} ms`,
        stat3: `Matrix sparsity: ${currSparsity.toFixed(1)}%`,
        warning:
          gridStepMs >= 50
            ? 'Coarse quantization: Time steps >= 50 ms erase human expressive microtiming, swing, and ornaments'
            : 'Fine temporal grid: Captures expressive performance nuance, but 2D piano roll matrix exceeds 96% sparsity',
        axisXLabel: 'Piano Roll Time-Step Quantization Grid (ms)',
        axisYLabel: 'Max Timing Discretization Jitter Error (ms)',
        xRange: [5, 105] as [number, number],
        yRange: [0, 55] as [number, number],
      }
    } else if (visualizerMode === 'tokenization_density') {
      // Piece length in seconds: 10, 20, 30, 60, 90, 120
      const lengths = [10, 20, 30, 45, 60, 90, 120]
      // Dense 2D piano roll frames: length / (gridStepMs / 1000)
      const frames = lengths.map((l) => Math.round(l / (gridStepMs / 1000)))
      // Event tokens (REMI / MIDI-like): ~4 events per note (Pitch, Velocity, Duration, TimeShift)
      // Note rate ~ 4 notes per second * polyphony
      const noteRate = 4 * polyphonicNotes
      const eventTokens = lengths.map((l) => Math.round(l * noteRate * 3.5))

      const currFrames = Math.round(60 / (gridStepMs / 1000))
      const currTokens = Math.round(60 * noteRate * 3.5)

      return {
        x: lengths,
        y: eventTokens,
        baseline: frames,
        stat1: `Active voice count: ${polyphonicNotes}`,
        stat2: `60s event tokens: ${currTokens.toLocaleString()}`,
        stat3: `60s piano roll frames: ${currFrames.toLocaleString()}`,
        warning:
          currTokens < currFrames
            ? 'Event tokenization achieves superior compression: Serializing discrete note events avoids storing empty piano roll silence'
            : 'High polyphony: Complex orchestral scores with dense chord voicings increase token sequence length',
        axisXLabel: 'Musical Excerpt Duration (Seconds)',
        axisYLabel: 'Sequence Length / Element Count',
        xRange: [5, 125] as [number, number],
        yRange: [0, Math.max(...eventTokens, ...frames) * 1.1] as [number, number],
      }
    } else {
      // Audio to score DTW alignment across 10 musical bars
      const bars = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      // Strict metronomic score tempo (constant 2.0 seconds per bar = cumulative 2 * bar)
      const scoreTime = bars.map((b) => b * 2.0)
      // Expressive human performance with rubato (accel and ritardando)
      const performanceTime = bars.map((b) => b * 2.0 + Math.sin(b * 0.7) * 1.2)

      const maxDrift = 1.2

      return {
        x: bars,
        y: performanceTime,
        baseline: scoreTime,
        stat1: 'Alignment: Chroma-based DTW',
        stat2: `Max rubato tempo drift: ±${maxDrift.toFixed(1)} s`,
        stat3: 'Dynamic Time Warping Converged',
        warning:
          'Human performances deviate significantly from strict score grids (rubato); DTW aligns acoustic chroma features to symbolic pitches',
        axisXLabel: 'Musical Measure Index (Bars)',
        axisYLabel: 'Cumulative Performance Elapsed Time (Seconds)',
        xRange: [0.5, 10.5] as [number, number],
        yRange: [0, 24] as [number, number],
      }
    }
  }, [visualizerMode, gridStepMs, polyphonicNotes])

  const axisX = useAxis({
    label: results.axisXLabel,
    range: results.xRange,
  })

  const axisY = useAxis({
    label: results.axisYLabel,
    range: results.yRange,
  })

  return (
    <Figure
      title="Symbolic Music Data: Piano Roll Grids, Event Tokenization, and Score-Audio Alignment"
      state={state}
      hoverReadout={false}
      caption="Representing symbolic musical compositions: Trade-offs between continuous time-step quantization and matrix sparsity, discrete event tokenization compression, and acoustic DTW alignment under expressive rubato."
      readouts={
        <>
          <Readout label="Symbolic Setup" value={results.stat1} />
          <Readout label="Precision Metric" value={results.stat2} />
          <Readout
            label={
              visualizerMode === 'piano_roll_grid'
                ? 'Matrix Sparsity'
                : visualizerMode === 'tokenization_density'
                  ? 'Sequence Scale'
                  : 'Alignment Status'
            }
            value={results.stat3}
            color={visualizerMode === 'piano_roll_grid' ? 'var(--amber-600)' : 'var(--blue-600)'}
          />
          <Readout label="Representation Directive" value={results.warning} color="var(--red-600)" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Symbolic Music Representation Profile">
        <Curve
          name={
            visualizerMode === 'piano_roll_grid'
              ? 'Matrix Sparsity Reference Threshold (%)'
              : visualizerMode === 'tokenization_density'
                ? 'Dense Piano Roll Frame Count'
                : 'Strict Metronomic Score Reference (Constant Tempo)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'piano_roll_grid'
              ? 'Discretization Timing Jitter (ms)'
              : visualizerMode === 'tokenization_density'
                ? 'Sparse Event Token Count (REMI Format)'
                : 'Observed Human Performance Timeline (Rubato)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Configuration Anchors" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
