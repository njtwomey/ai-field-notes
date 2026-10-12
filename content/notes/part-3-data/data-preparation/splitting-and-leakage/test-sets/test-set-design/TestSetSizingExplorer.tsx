import React, { useMemo, useState } from 'react'

export default function TestSetSizingExplorer(): React.ReactElement {
  const [testSizeN, setTestSizeN] = useState<number>(500)
  const [observedAccuracyPct, setObservedAccuracyPct] = useState<number>(90)
  const [minoritySlicePct, setMinoritySlicePct] = useState<number>(4)
  const [testLabelNoisePct, setTestLabelNoisePct] = useState<number>(3)

  // Computations for global confidence interval (95% Wald CI)
  const p = observedAccuracyPct / 100
  const seGlobal = useMemo(() => {
    return Math.sqrt((p * (1 - p)) / testSizeN)
  }, [p, testSizeN])

  const marginOfErrorPct = seGlobal * 1.96 * 100
  const ciLower = Math.max(0, observedAccuracyPct - marginOfErrorPct)
  const ciUpper = Math.min(100, observedAccuracyPct + marginOfErrorPct)

  // Subgroup slice calculations
  const sliceCountN = Math.max(1, Math.round((testSizeN * minoritySlicePct) / 100))
  const seSlice = useMemo(() => {
    return Math.sqrt((p * (1 - p)) / sliceCountN)
  }, [p, sliceCountN])
  const sliceMarginPct = seSlice * 1.96 * 100
  const sliceCiLower = Math.max(0, observedAccuracyPct - sliceMarginPct)
  const sliceCiUpper = Math.min(100, observedAccuracyPct + sliceMarginPct)

  // Maximum measurable accuracy ceiling
  const accuracyCeilingPct = 100 - testLabelNoisePct

  return (
    <div
      style={{
        margin: '1.75rem 0',
        padding: '1.25rem',
        borderRadius: '10px',
        border: '1px solid var(--color-border-subtle, #333)',
        background: 'var(--color-bg-subtle, #14171d)',
        fontFamily: 'inherit',
      }}
    >
      {/* Header */}
      <div style={{ marginBottom: '1.1rem' }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '0.75rem',
          }}
        >
          <h4 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 600, color: 'var(--color-text-emphasis, #f0f3f6)' }}>
            Test-Set Statistical Power & Slicing Calculator
          </h4>
          <span
            style={{
              fontSize: '0.74rem',
              padding: '0.2rem 0.55rem',
              borderRadius: '999px',
              background: 'rgba(56, 189, 248, 0.15)',
              color: '#38bdf8',
              fontWeight: 600,
            }}
          >
            N = {testSizeN.toLocaleString()} Samples &middot; Margin: &plusmn;{marginOfErrorPct.toFixed(1)}%
          </span>
        </div>
        <p
          style={{
            margin: '0.4rem 0 0',
            fontSize: '0.82rem',
            color: 'var(--color-text-muted, #94a3b8)',
            lineHeight: 1.45,
          }}
        >
          A test set is an empirical measurement instrument. Its total count dictates precision, its sub-slices dictate
          demographic fairness, and its label error rate caps measurable accuracy.
        </p>
      </div>

      {/* Control row */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '0.85rem',
          marginBottom: '1.25rem',
          padding: '0.75rem',
          borderRadius: '8px',
          background: 'rgba(0, 0, 0, 0.25)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
        }}
      >
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
            <label style={{ fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', color: '#94a3b8' }}>
              Test Set Size (N): <strong style={{ color: '#38bdf8' }}>{testSizeN.toLocaleString()}</strong>
            </label>
          </div>
          <input
            type="range"
            min={100}
            max={5000}
            step={100}
            value={testSizeN}
            onChange={(e) => setTestSizeN(parseInt(e.target.value, 10))}
            style={{ width: '100%', cursor: 'pointer', accentColor: '#38bdf8' }}
          />
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
            <label style={{ fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', color: '#94a3b8' }}>
              Observed Accuracy: <strong style={{ color: '#34d399' }}>{observedAccuracyPct}%</strong>
            </label>
          </div>
          <input
            type="range"
            min={70}
            max={99}
            step={1}
            value={observedAccuracyPct}
            onChange={(e) => setObservedAccuracyPct(parseInt(e.target.value, 10))}
            style={{ width: '100%', cursor: 'pointer', accentColor: '#34d399' }}
          />
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
            <label style={{ fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', color: '#94a3b8' }}>
              Minority Slice Prevalence: <strong style={{ color: '#f59e0b' }}>{minoritySlicePct}%</strong>
            </label>
          </div>
          <input
            type="range"
            min={1}
            max={20}
            step={1}
            value={minoritySlicePct}
            onChange={(e) => setMinoritySlicePct(parseInt(e.target.value, 10))}
            style={{ width: '100%', cursor: 'pointer', accentColor: '#f59e0b' }}
          />
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
            <label style={{ fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', color: '#94a3b8' }}>
              Test Label Noise: <strong style={{ color: '#f43f5e' }}>{testLabelNoisePct}%</strong>
            </label>
          </div>
          <input
            type="range"
            min={0}
            max={15}
            step={1}
            value={testLabelNoisePct}
            onChange={(e) => setTestLabelNoisePct(parseInt(e.target.value, 10))}
            style={{ width: '100%', cursor: 'pointer', accentColor: '#f43f5e' }}
          />
        </div>
      </div>

      {/* Statistical Precision Metrics Cards */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '0.75rem',
          marginBottom: '1.25rem',
        }}
      >
        {/* Global CI */}
        <div
          style={{
            padding: '0.75rem',
            borderRadius: '6px',
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid rgba(255, 255, 255, 0.05)',
          }}
        >
          <span style={{ display: 'block', fontSize: '0.68rem', color: '#94a3b8', textTransform: 'uppercase' }}>
            Global 95% Confidence Interval
          </span>
          <strong style={{ fontSize: '1.25rem', color: '#38bdf8' }}>
            [{ciLower.toFixed(1)}%, {ciUpper.toFixed(1)}%]
          </strong>
          <span style={{ display: 'block', fontSize: '0.7rem', color: '#64748b', marginTop: '0.2rem' }}>
            Margin of error: &plusmn;{marginOfErrorPct.toFixed(1)}% around {observedAccuracyPct}%
          </span>
        </div>

        {/* Minority Slice CI */}
        <div
          style={{
            padding: '0.75rem',
            borderRadius: '6px',
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid rgba(255, 255, 255, 0.05)',
          }}
        >
          <span style={{ display: 'block', fontSize: '0.68rem', color: '#94a3b8', textTransform: 'uppercase' }}>
            Minority Slice Precision (N = {sliceCountN})
          </span>
          <strong style={{ fontSize: '1.25rem', color: sliceMarginPct > 8 ? '#f43f5e' : '#f59e0b' }}>
            [{sliceCiLower.toFixed(1)}%, {sliceCiUpper.toFixed(1)}%]
          </strong>
          <span style={{ display: 'block', fontSize: '0.7rem', color: '#64748b', marginTop: '0.2rem' }}>
            Slice error margin: &plusmn;{sliceMarginPct.toFixed(1)}% ({sliceMarginPct > 8 ? 'Too wide!' : 'Acceptable'})
          </span>
        </div>

        {/* Label Noise Ceiling */}
        <div
          style={{
            padding: '0.75rem',
            borderRadius: '6px',
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid rgba(255, 255, 255, 0.05)',
          }}
        >
          <span style={{ display: 'block', fontSize: '0.68rem', color: '#94a3b8', textTransform: 'uppercase' }}>
            Measurable Accuracy Ceiling
          </span>
          <strong style={{ fontSize: '1.25rem', color: accuracyCeilingPct < 95 ? '#f59e0b' : '#34d399' }}>
            {accuracyCeilingPct.toFixed(1)}% Max
          </strong>
          <span style={{ display: 'block', fontSize: '0.7rem', color: '#64748b', marginTop: '0.2rem' }}>
            A perfect model cannot score above this ceiling
          </span>
        </div>
      </div>

      {/* Visual representation of CI interval widths */}
      <div
        style={{
          padding: '0.85rem',
          borderRadius: '8px',
          background: 'rgba(0, 0, 0, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
          marginBottom: '1rem',
        }}
      >
        <div
          style={{
            fontSize: '0.72rem',
            fontWeight: 600,
            color: '#94a3b8',
            marginBottom: '0.6rem',
            textTransform: 'uppercase',
          }}
        >
          Interval Resolution: Global Test vs Slice Precision
        </div>

        {/* Global CI bar */}
        <div style={{ marginBottom: '0.75rem' }}>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: '0.72rem',
              color: '#cbd5e1',
              marginBottom: '0.25rem',
            }}
          >
            <span>Full Population (N = {testSizeN})</span>
            <span style={{ color: '#38bdf8' }}>&plusmn;{marginOfErrorPct.toFixed(1)}%</span>
          </div>
          <div
            style={{
              position: 'relative',
              width: '100%',
              height: '14px',
              background: 'rgba(255,255,255,0.05)',
              borderRadius: '3px',
            }}
          >
            <div
              style={{
                position: 'absolute',
                left: `${ciLower}%`,
                width: `${ciUpper - ciLower}%`,
                height: '100%',
                background: '#38bdf8',
                borderRadius: '3px',
                opacity: 0.8,
              }}
            />
          </div>
        </div>

        {/* Slice CI bar */}
        <div>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: '0.72rem',
              color: '#cbd5e1',
              marginBottom: '0.25rem',
            }}
          >
            <span>Minority Slice (N = {sliceCountN} samples)</span>
            <span style={{ color: '#f59e0b' }}>&plusmn;{sliceMarginPct.toFixed(1)}%</span>
          </div>
          <div
            style={{
              position: 'relative',
              width: '100%',
              height: '14px',
              background: 'rgba(255,255,255,0.05)',
              borderRadius: '3px',
            }}
          >
            <div
              style={{
                position: 'absolute',
                left: `${sliceCiLower}%`,
                width: `${sliceCiUpper - sliceCiLower}%`,
                height: '100%',
                background: '#f59e0b',
                borderRadius: '3px',
                opacity: 0.8,
              }}
            />
          </div>
        </div>
      </div>

      {/* Analytical Callout */}
      <div
        style={{
          padding: '0.75rem',
          borderRadius: '7px',
          background: 'rgba(0, 0, 0, 0.25)',
          fontSize: '0.78rem',
          color: '#cbd5e1',
          lineHeight: 1.5,
        }}
      >
        <span style={{ fontWeight: 600, color: '#f8fafc' }}>Key Takeaway: </span>
        Even when a test set of N = 500 has a tight global confidence interval (&plusmn;{marginOfErrorPct.toFixed(1)}%),
        a 4% subpopulation slice contains only {sliceCountN} examples, blowing out its confidence interval to a massive
        &plusmn;{sliceMarginPct.toFixed(1)}%. Test sets must be intentionally sized and stratified across critical
        operational slices, not sampled blindly.
      </div>
    </div>
  )
}
