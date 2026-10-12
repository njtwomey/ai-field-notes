import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function SpeechTranscriptsAndAlignmentVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'forced_alignment' | 'text_normalization' | 'diarization_collar'>(
      [
        {
          value: 'forced_alignment',
          label: 'Forced Alignment: Viterbi Trellis Frame-to-Phoneme Path (McAuliffe 2017)',
        },
        { value: 'text_normalization', label: 'Text Normalization: Written vs Spoken Verbalization Mismatch' },
        {
          value: 'diarization_collar',
          label: 'Speaker Diarization: Boundary Collar Tolerance vs Diarization Error Rate',
        },
      ],
      'forced_alignment',
      { label: 'speech-text alignment perspective' },
    ),
    utteranceSeconds: slider(2, 10, 4, {
      step: 1,
      label: 'audio utterance duration (seconds)',
    }),
    collarToleranceMs: slider(50, 500, 250, {
      step: 50,
      label: 'diarization evaluation collar (ms)',
    }),
  })

  const { visualizerMode, utteranceSeconds, collarToleranceMs } = state

  const results = useMemo(() => {
    if (visualizerMode === 'forced_alignment') {
      // 10 phoneme sequence positions across timeline
      const phonemePositions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      // Time frames (100 frames per second): Total frames = utteranceSeconds * 100
      // Viterbi forward path accumulates alignment frame timestamps
      const totalFrames = utteranceSeconds * 100
      const alignedFrames = phonemePositions.map((pos) => Math.round((pos / 10) * totalFrames))
      const linearTheoretical = phonemePositions.map((pos) => Math.round((pos / 10) * totalFrames))

      const avgPhoneDurationMs = Math.round((utteranceSeconds * 1000) / 10)

      return {
        x: phonemePositions,
        y: alignedFrames,
        baseline: linearTheoretical,
        stat1: `Utterance duration: ${utteranceSeconds}s (${totalFrames} frames)`,
        stat2: `Phonemes aligned: 10`,
        stat3: `Mean phone length: ${avgPhoneDurationMs} ms`,
        warning:
          'Viterbi forced alignment constrains acoustic decoding to the exact transcript, finding frame-level phone and word boundaries',
        axisXLabel: 'Transcript Phoneme State Index (/k/ -> /ae/ -> /t/ ...)',
        axisYLabel: 'Acoustic Frame Timestamp Index (10 ms / frame)',
        xRange: [0.5, 10.5] as [number, number],
        yRange: [0, totalFrames * 1.05] as [number, number],
      }
    } else if (visualizerMode === 'text_normalization') {
      // Numbers, currencies, dates, abbreviations: token expansion factor
      const entityTypes = [1, 2, 3, 4, 5, 6] // 1: Currencies ($20), 2: Numbers (1,500), 3: Dates (2026), 4: Acronyms (NASA), 5: Times (4:30), 6: Units (5kg)
      // Written character count (short: e.g. 3 to 5 chars)
      const writtenLengths = [3, 5, 4, 4, 4, 3]
      // Spoken verbalized character count (e.g. "twenty dollars" = 14, "fifteen hundred" = 15)
      const spokenLengths = [14, 15, 23, 16, 12, 12]

      return {
        x: entityTypes,
        y: spokenLengths,
        baseline: writtenLengths,
        stat1: 'Entity types: Numbers, Currencies, Dates',
        stat2: 'Spoken expansion: 3.5x characters',
        stat3: 'Verbalization Critical',
        warning:
          'Mismatch between written text ("$50") and spoken acoustics ("fifty dollars") severely penalizes ASR training unless normalized',
        axisXLabel: 'Non-Standard Word Category (1=$, 2=Num, 3=Year, 4=Acronym, 5=Time, 6=Unit)',
        axisYLabel: 'Character String Representation Length',
        xRange: [0.5, 6.5] as [number, number],
        yRange: [0, 26] as [number, number],
      }
    } else {
      // Collar tolerance from 50 ms to 500 ms
      const collars = [50, 100, 150, 200, 250, 300, 350, 400, 500]
      // Diarization Error Rate (DER) decreases as collar expands (boundary ambiguity forgiven)
      // Base DER ~ 18% at 0 collar, drops to ~7% at 500ms collar
      const der = collars.map((c) => Math.max(5, 22 - Math.sqrt(c) * 0.7))
      const strictBaseline = collars.map(() => 15.0)

      const currDER = Math.max(5, 22 - Math.sqrt(collarToleranceMs) * 0.7)

      return {
        x: collars,
        y: der,
        baseline: strictBaseline,
        stat1: `Collar tolerance: ±${collarToleranceMs} ms`,
        stat2: `Evaluated DER: ${currDER.toFixed(1)}%`,
        stat3: collarToleranceMs >= 250 ? 'Standard NIST Collar' : 'Strict Collar',
        warning:
          collarToleranceMs < 150
            ? 'Strict collar penalizes human reaction time and speaker transition overlap, inflating measured diarization error'
            : 'NIST 250 ms collar ignores small boundary misalignments, focusing error metrics on speaker confusion',
        axisXLabel: 'Diarization Evaluation Forgiveness Collar (ms)',
        axisYLabel: 'Diarization Error Rate DER (%)',
        xRange: [30, 520] as [number, number],
        yRange: [0, 25] as [number, number],
      }
    }
  }, [visualizerMode, utteranceSeconds, collarToleranceMs])

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
      title="Speech Transcripts & Alignment: Forced Alignment Trellis, Verbalization, and Diarization Collars"
      state={state}
      hoverReadout={false}
      caption="Coupling acoustics to textual annotations: Frame-to-phoneme Viterbi forced alignment, the written-to-spoken verbalization gap, and speaker diarization collar evaluation dynamics."
      readouts={
        <>
          <Readout label="Alignment Task" value={results.stat1} />
          <Readout label="Annotation Scale" value={results.stat2} />
          <Readout
            label={
              visualizerMode === 'forced_alignment'
                ? 'Mean Duration'
                : visualizerMode === 'text_normalization'
                  ? 'Alignment Status'
                  : 'Collar Standard'
            }
            value={results.stat3}
            color={visualizerMode === 'text_normalization' ? 'var(--amber-600)' : 'var(--blue-600)'}
          />
          <Readout label="ASR Practice Directive" value={results.warning} color="var(--red-600)" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Speech-Text Alignment Dynamics">
        <Curve
          name={
            visualizerMode === 'forced_alignment'
              ? 'Linear Proportional Reference Timeline'
              : visualizerMode === 'text_normalization'
                ? 'Raw Written Representation Length'
                : '15% Baseline Acceptable DER Threshold'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'forced_alignment'
              ? 'Viterbi Trellis Optimal Frame Boundaries'
              : visualizerMode === 'text_normalization'
                ? 'Verbalized Spoken Expansion Length'
                : 'Evaluated Diarization Error Rate (DER)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Lattice Coordinates" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
