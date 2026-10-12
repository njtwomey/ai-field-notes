import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function PhysiologicalSignalsVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'ecg_filtering' | 'subject_split' | 'artefact_snr'>(
      [
        {
          value: 'ecg_filtering',
          label: 'Signal Conditioning: Raw ECG with Wander vs Pan-Tompkins Filtered (Pan 1985)',
        },
        {
          value: 'subject_split',
          label: 'Evaluation Integrity: Random Window Leakage vs Subject-Wise Split (Saeb 2017)',
        },
        { value: 'artefact_snr', label: 'Motion Artefact Rejection: Motion Intensity vs Clinical Usable Signal (%)' },
      ],
      'ecg_filtering',
      { label: 'signal processing perspective' },
    ),
    subjectCohort: slider(10, 100, 30, {
      step: 5,
      label: 'patient subject cohort N_subjects',
    }),
    artefactIntensity: slider(1, 10, 4, {
      step: 1,
      label: 'motion artefact / baseline wander level',
    }),
  })

  const { visualizerMode, subjectCohort, artefactIntensity } = state

  const results = useMemo(() => {
    if (visualizerMode === 'ecg_filtering') {
      // Time steps across two cardiac cycles (0 to 100 points, arbitrary time)
      const samples = [0, 10, 20, 25, 30, 35, 40, 50, 60, 70, 75, 80, 85, 90, 100]
      // Synthetic clean QRS morphology: baseline 0, P wave at 20, Q at 28 (-2), R peak at 30 (+15), S at 32 (-4), T wave at 45 (+3)
      const cleanECG = samples.map((t) => {
        const cycle = t % 50
        if (cycle === 10) return 2.0 // P wave
        if (cycle === 18) return -2.0 // Q
        if (cycle === 20) return 15.0 // R peak
        if (cycle === 22) return -4.0 // S
        if (cycle === 35) return 3.5 // T wave
        return 0.0
      })

      // Raw ECG corrupted by respiratory baseline wander + powerline hum
      const rawECG = samples.map((t, idx) => {
        const wander = Math.sin((t / 100) * Math.PI * 2) * (artefactIntensity * 2.5)
        const powerline = (idx % 2 === 0 ? 1 : -1) * (artefactIntensity * 0.8)
        return cleanECG[idx] + wander + powerline
      })

      const wanderPeak = artefactIntensity * 2.5

      return {
        x: samples,
        y: rawECG,
        baseline: cleanECG,
        stat1: `Wander intensity: Level ${artefactIntensity}`,
        stat2: `Peak wander offset: ±${wanderPeak.toFixed(1)} mV`,
        stat3: 'Pan-Tompkins 0.5-40Hz Pass',
        warning:
          wanderPeak > 10
            ? 'Severe baseline wander: Distorts ST-segment elevation, inducing false myocardial infarction classifications'
            : 'Linear-phase high-pass filter (>0.5 Hz) removes respiratory drift without shifting QRS phase',
        axisXLabel: 'Cardiac Cycle Time Samples (Arbitrary Sampling Grid)',
        axisYLabel: 'Electrophysiological Amplitude (mV)',
        xRange: [-2, 102] as [number, number],
        yRange: [-30, 35] as [number, number],
      }
    } else if (visualizerMode === 'subject_split') {
      // Subject cohort sizes N from 10 to 100
      const cohorts = [10, 20, 30, 40, 50, 60, 80, 100]
      // Random window split accuracy: stays artificially high ~98% across all cohort sizes because individual identity leaks
      const randomSplitAcc = cohorts.map(() => 98.2)
      // Subject-wise split: true clinical generalization improves with cohort size N
      const subjectWiseAcc = cohorts.map((n) => Math.min(92.0, 72.0 + Math.log2(n / 5) * 4.5))

      const currSubjectAcc = Math.min(92.0, 72.0 + Math.log2(subjectCohort / 5) * 4.5)
      const leakageGap = 98.2 - currSubjectAcc

      return {
        x: cohorts,
        y: subjectWiseAcc,
        baseline: randomSplitAcc,
        stat1: `Cohort size N = ${subjectCohort}`,
        stat2: `Subject-split accuracy: ${currSubjectAcc.toFixed(1)}%`,
        stat3: `Random-split leakage gap: +${leakageGap.toFixed(1)}%`,
        warning:
          'CRITICAL VALIDATION FLAW: Random window splitting overfits individual patient electrophysiology; only subject-wise splits assess clinical generalization',
        axisXLabel: 'Patient Cohort Sample Size (Unique Subjects N)',
        axisYLabel: 'Diagnostic Model Accuracy (%)',
        xRange: [5, 105] as [number, number],
        yRange: [65, 102] as [number, number],
      }
    } else {
      // Motion artefact vs usable clean recording fraction
      const intensities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      // Usable signal fraction (% of epochs passing signal quality index SQI)
      const usableFraction = intensities.map((i) => Math.max(10, Math.round(100 * Math.exp(-0.22 * i))))
      const clinicalMinThreshold = intensities.map(() => 70)

      const currUsable = Math.max(10, Math.round(100 * Math.exp(-0.22 * artefactIntensity)))

      return {
        x: intensities,
        y: usableFraction,
        baseline: clinicalMinThreshold,
        stat1: `Motion artefact level: ${artefactIntensity}`,
        stat2: `Retained signal epochs: ${currUsable}%`,
        stat3: currUsable >= 70 ? 'Clinical SQI Passed' : 'High Rejection Rate',
        warning:
          currUsable < 70
            ? 'Excessive motion contamination: Over 30% of recording rejected; PPG/EEG sensors require active accelerometer cancellation'
            : 'Acceptable signal-to-noise ratio: Sufficient clean physiological epochs for downstream diagnostic classification',
        axisXLabel: 'Physical Motion Artefact Severity Index',
        axisYLabel: 'Retained Usable Signal Fraction (% Epochs)',
        xRange: [0.5, 10.5] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    }
  }, [visualizerMode, subjectCohort, artefactIntensity])

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
      title="Physiological Signals: Signal Conditioning, Subject-Wise Generalization, and Artefact Screening"
      state={state}
      hoverReadout={false}
      caption="Processing electrophysiological recordings: Respiratory baseline wander and powerline filtering, optimistic evaluation leakage under record-level splits, and motion artefact signal quality index (SQI) screening."
      readouts={
        <>
          <Readout label="Conditioning Setup" value={results.stat1} />
          <Readout label="Performance Metric" value={results.stat2} />
          <Readout
            label={
              visualizerMode === 'ecg_filtering'
                ? 'Filter Specification'
                : visualizerMode === 'subject_split'
                  ? 'Evaluation Bias'
                  : 'Signal Quality'
            }
            value={results.stat3}
            color={visualizerMode === 'subject_split' ? 'var(--red-600)' : 'var(--blue-600)'}
          />
          <Readout label="Clinical Directive" value={results.warning} color="var(--amber-600)" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Electrophysiological Signal Profile">
        <Curve
          name={
            visualizerMode === 'ecg_filtering'
              ? 'Filtered Diagnostic ECG (Pan-Tompkins Reference)'
              : visualizerMode === 'subject_split'
                ? 'Naive Random Window Split (Identity Leakage ~98%)'
                : 'Clinical 70% Retained Data Threshold'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'ecg_filtering'
              ? 'Raw Corrupted ECG with Respiratory Wander & Hum'
              : visualizerMode === 'subject_split'
                ? 'Subject-Wise Cross-Validation Generalization'
                : 'Observed Usable Epoch Fraction (Post-SQI)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Operating Measurement" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
