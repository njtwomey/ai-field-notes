import { useMemo } from 'react'
import { choice, Figure, Readout, useFigureState } from 'aifn-render'

export function CleaningEvaluationFlowExplorer() {
  const state = useFigureState({
    pipelineDesign: choice<'valid' | 'global_clean' | 'manual_test_clean'>(
      [
        { value: 'valid', label: 'rigorous pipeline (fit on train, frozen at serving)' },
        { value: 'global_clean', label: 'global cleaning leakage (clean before split)' },
        { value: 'manual_test_clean', label: 'manual test set cleaning (hand-edited test)' },
      ],
      'valid',
      { label: 'cleaning & evaluation protocol' },
    ),
  })

  const results = useMemo(() => {
    if (state.pipelineDesign === 'valid') {
      return {
        trainCleaning: 'Automated rules fitted strictly on training partition',
        testCleaning: 'Frozen pipeline transforms raw test features automatically',
        servingCleaning: 'Identical frozen pipeline deployed in production API',
        testScore: '84.2% Accuracy (unbiased & honest)',
        prodScore: '83.9% Accuracy (matches test expectation)',
        optimismGap: '+0.3% (negligible)',
        status: 'Valid & Production-Ready',
        isSound: true,
      }
    }

    if (state.pipelineDesign === 'global_clean') {
      return {
        trainCleaning: 'Imputation & outlier thresholds fitted on full pooled dataset',
        testCleaning: 'Contaminated by training statistics during pooled pre-processing',
        servingCleaning: 'Inference crashes or drifts without pooled global statistics',
        testScore: '92.8% Accuracy (falsely optimistic)',
        prodScore: '71.4% Accuracy (severe real-world collapse)',
        optimismGap: '+21.4% (catastrophic leakage bias)',
        status: 'Flawed (Data Leakage)',
        isSound: false,
      }
    }

    // Manual test set cleaning:
    return {
      trainCleaning: 'Cleaned via ad-hoc scripts and outlier removal',
      testCleaning: 'Human analysts manually repaired or deleted difficult test anomalies',
      servingCleaning: 'Live API receives raw un-sanitized edge cases without human repair',
      testScore: '95.1% Accuracy (evaluates artificial clean world)',
      prodScore: '68.0% Accuracy (fails on real-world raw inputs)',
      optimismGap: '+27.1% (support truncation bias)',
      status: 'Flawed (Support Truncation)',
      isSound: false,
    }
  }, [state.pipelineDesign])

  return (
    <Figure
      title="Impact of data cleaning protocols on evaluation validity and production performance"
      state={state}
      hoverReadout={false}
      caption="Contrasting data cleaning protocols across the machine learning lifecycle. A rigorous pipeline fits cleaning transformations strictly on training data, freezes the transformer, and applies it identically to test features and serving payloads. Global cleaning leaks test distributions into training. Manually editing test features inflates test scores on an idealized synthetic distribution, causing catastrophic accuracy collapse when deployed to raw production streams."
      readouts={
        <>
          <Readout label="protocol integrity" value={results.status} />
          <Readout label="reported test metric" value={results.testScore} />
          <Readout label="live production score" value={results.prodScore} />
          <Readout label="optimism bias gap" value={results.optimismGap} />
        </>
      }
    >
      <div style={{ padding: '14px 16px', fontSize: '13px', lineHeight: 1.5 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid rgba(128,128,128,0.3)', fontWeight: 600 }}>
              <th style={{ padding: '6px 8px' }}>Lifecycle Stage</th>
              <th style={{ padding: '6px 8px' }}>Execution Protocol</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid rgba(128,128,128,0.15)' }}>
              <td style={{ padding: '6px 8px', fontWeight: 600 }}>Training Set Clean</td>
              <td style={{ padding: '6px 8px' }}>{results.trainCleaning}</td>
            </tr>
            <tr style={{ borderBottom: '1px solid rgba(128,128,128,0.15)' }}>
              <td style={{ padding: '6px 8px', fontWeight: 600 }}>Test Evaluation Features</td>
              <td style={{ padding: '6px 8px' }}>{results.testCleaning}</td>
            </tr>
            <tr style={{ borderBottom: '1px solid rgba(128,128,128,0.15)' }}>
              <td style={{ padding: '6px 8px', fontWeight: 600 }}>Production Serving API</td>
              <td style={{ padding: '6px 8px' }}>{results.servingCleaning}</td>
            </tr>
            <tr style={{ borderBottom: '1px solid rgba(128,128,128,0.15)' }}>
              <td style={{ padding: '6px 8px', fontWeight: 600 }}>Generalisation Gap</td>
              <td
                style={{
                  padding: '6px 8px',
                  fontWeight: 600,
                  color: results.isSound ? '#16a34a' : '#ef4444',
                }}
              >
                {results.optimismGap} {results.isSound ? '(Reliable)' : '(Unreliable / Severe Overfitting)'}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </Figure>
  )
}
