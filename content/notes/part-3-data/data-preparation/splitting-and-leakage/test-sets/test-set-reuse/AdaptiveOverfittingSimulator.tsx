import React, { useMemo, useState } from 'react'

type DefenseMode = 'none' | 'thresholdout' | 'budget'

interface QueryStep {
  queryIdx: number
  apparentHoldoutAcc: number
  trueGenAcc: number
  overfitGap: number
}

// Generate simulation trajectory of adaptive overfitting across 100 queries
function generateTrajectory(defense: DefenseMode): QueryStep[] {
  const steps: QueryStep[] = []
  const baseTrueAcc = 78.0

  let currentApparent = baseTrueAcc + 0.2
  let currentTrue = baseTrueAcc

  for (let q = 0; q <= 60; q++) {
    if (q === 0) {
      steps.push({
        queryIdx: 0,
        apparentHoldoutAcc: baseTrueAcc + 0.2,
        trueGenAcc: baseTrueAcc,
        overfitGap: 0.2,
      })
      continue
    }

    if (defense === 'none') {
      // Unrestricted reuse: opportunistic hill-climbing overfits holdout noise
      // Apparent score inflates smoothly towards 91%, while true accuracy slightly decays
      const noiseGain = Math.sqrt(q) * 1.65
      currentApparent = Math.min(92.5, baseTrueAcc + noiseGain + Math.sin(q * 0.3) * 0.4)
      currentTrue = Math.max(74.5, baseTrueAcc - q * 0.04)
    } else if (defense === 'thresholdout') {
      // Dwork et al. 2015: Thresholdout with Laplacian noise clamps spurious gains
      // Only rewards genuine advances, keeping holdout score locked to true score
      currentApparent = baseTrueAcc + Math.sin(q * 0.1) * 0.5
      currentTrue = baseTrueAcc + (q > 30 ? 0.8 : 0.2)
    } else {
      // Strict submission budget: capped at 10 queries total
      if (q <= 10) {
        currentApparent = baseTrueAcc + Math.sqrt(q) * 0.9
        currentTrue = baseTrueAcc + 0.1
      } else {
        // Budget exhausted: no more queries permitted
        const lastValid = steps[10]
        currentApparent = lastValid.apparentHoldoutAcc
        currentTrue = lastValid.trueGenAcc
      }
    }

    steps.push({
      queryIdx: q,
      apparentHoldoutAcc: parseFloat(currentApparent.toFixed(1)),
      trueGenAcc: parseFloat(currentTrue.toFixed(1)),
      overfitGap: parseFloat((currentApparent - currentTrue).toFixed(1)),
    })
  }

  return steps
}

export default function AdaptiveOverfittingSimulator(): React.ReactElement {
  const [defense, setDefense] = useState<DefenseMode>('none')
  const [currentQuery, setCurrentQuery] = useState<number>(35)

  const trajectory = useMemo(() => generateTrajectory(defense), [defense])
  const activeStep = trajectory[Math.min(currentQuery, trajectory.length - 1)]

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
            Adaptive Test-Set Overfitting Simulator
          </h4>
          <span
            style={{
              fontSize: '0.74rem',
              padding: '0.2rem 0.55rem',
              borderRadius: '999px',
              background: activeStep.overfitGap > 5 ? 'rgba(244, 63, 94, 0.15)' : 'rgba(52, 211, 153, 0.15)',
              color: activeStep.overfitGap > 5 ? '#f43f5e' : '#34d399',
              fontWeight: 600,
            }}
          >
            {activeStep.overfitGap > 5 ? `Overfit Gap: +${activeStep.overfitGap.toFixed(1)}%` : 'Calibrated Holdout'}
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
          Every engineering decision made by inspecting a test set score leaks information from the test set, gradually
          fitting its idiosyncratic sampling noise.
        </p>
      </div>

      {/* Defense Strategy Tabs */}
      <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '1.25rem', flexWrap: 'wrap' }}>
        {(
          [
            ['none', '1. Unrestricted Reuse (Naive Overfitting)'],
            ['thresholdout', '2. Thresholdout (Dwork et al. 2015)'],
            ['budget', '3. Strict Holdout Budget (10 Submissions)'],
          ] as const
        ).map(([dKey, label]) => (
          <button
            key={dKey}
            type="button"
            onClick={() => {
              setDefense(dKey)
              if (dKey === 'budget' && currentQuery > 10) setCurrentQuery(10)
            }}
            style={{
              flex: 1,
              minWidth: '170px',
              padding: '0.4rem 0.6rem',
              fontSize: '0.75rem',
              borderRadius: '5px',
              border: defense === dKey ? '1px solid #38bdf8' : '1px solid #444',
              background: defense === dKey ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
              color: defense === dKey ? '#38bdf8' : '#aaa',
              cursor: 'pointer',
              fontWeight: defense === dKey ? 600 : 400,
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Query Slider */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '1rem',
          marginBottom: '1.25rem',
          padding: '0.65rem 0.9rem',
          borderRadius: '7px',
          background: 'rgba(0, 0, 0, 0.25)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
        }}
      >
        <label style={{ fontSize: '0.76rem', fontWeight: 600, color: '#f8fafc', minWidth: '180px' }}>
          Evaluation Queries: <strong style={{ color: '#38bdf8' }}>{activeStep.queryIdx} / 60</strong>
        </label>
        <input
          type="range"
          min={0}
          max={60}
          value={currentQuery}
          onChange={(e) => setCurrentQuery(parseInt(e.target.value, 10))}
          style={{ flex: 1, cursor: 'pointer', accentColor: '#38bdf8' }}
        />
        <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>
          {defense === 'budget' && activeStep.queryIdx >= 10 ? 'BUDGET EXHAUSTED' : 'Iterative trial & error'}
        </span>
      </div>

      {/* Score Comparison Cards */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '0.75rem',
          marginBottom: '1.25rem',
        }}
      >
        <div
          style={{
            padding: '0.75rem',
            borderRadius: '6px',
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid rgba(255, 255, 255, 0.05)',
          }}
        >
          <span style={{ display: 'block', fontSize: '0.68rem', color: '#94a3b8', textTransform: 'uppercase' }}>
            Apparent Holdout Accuracy
          </span>
          <strong style={{ fontSize: '1.35rem', color: '#38bdf8' }}>{activeStep.apparentHoldoutAcc.toFixed(1)}%</strong>
          <span style={{ display: 'block', fontSize: '0.7rem', color: '#64748b', marginTop: '0.2rem' }}>
            What the public leaderboard reports
          </span>
        </div>

        <div
          style={{
            padding: '0.75rem',
            borderRadius: '6px',
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid rgba(255, 255, 255, 0.05)',
          }}
        >
          <span style={{ display: 'block', fontSize: '0.68rem', color: '#94a3b8', textTransform: 'uppercase' }}>
            True Out-of-Sample Accuracy
          </span>
          <strong style={{ fontSize: '1.35rem', color: '#34d399' }}>{activeStep.trueGenAcc.toFixed(1)}%</strong>
          <span style={{ display: 'block', fontSize: '0.7rem', color: '#64748b', marginTop: '0.2rem' }}>
            Actual performance on fresh replication test
          </span>
        </div>

        <div
          style={{
            padding: '0.75rem',
            borderRadius: '6px',
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid rgba(255, 255, 255, 0.05)',
          }}
        >
          <span style={{ display: 'block', fontSize: '0.68rem', color: '#94a3b8', textTransform: 'uppercase' }}>
            Overfitting Divergence (Gap)
          </span>
          <strong style={{ fontSize: '1.35rem', color: activeStep.overfitGap > 5 ? '#f43f5e' : '#34d399' }}>
            +{activeStep.overfitGap.toFixed(1)}%
          </strong>
          <span style={{ display: 'block', fontSize: '0.7rem', color: '#64748b', marginTop: '0.2rem' }}>
            {activeStep.overfitGap > 5 ? 'Severe Leaderboard Shakeup Ahead' : 'Well Calibrated'}
          </span>
        </div>
      </div>

      {/* Trajectory Timeline Preview */}
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
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: '0.72rem',
            color: '#94a3b8',
            marginBottom: '0.5rem',
          }}
        >
          <span>Accuracy Divergence Curve (0 to 60 Queries)</span>
          <span style={{ color: '#f43f5e' }}>Apparent (Cyan) vs Real (Green)</span>
        </div>

        {/* Mini SVG Curve */}
        <svg viewBox="0 0 300 80" style={{ width: '100%', height: 'auto', display: 'block' }}>
          {/* Baseline 78% line */}
          <line x1="0" y1="50" x2="300" y2="50" stroke="#475569" strokeWidth="1" strokeDasharray="3 3" />
          <text x="5" y="45" fill="#64748b" fontSize="8">
            Base (78%)
          </text>

          {/* Apparent Accuracy line */}
          <polyline
            fill="none"
            stroke="#38bdf8"
            strokeWidth="2"
            points={trajectory
              .map((st) => {
                const x = (st.queryIdx / 60) * 300
                // Map 74%..94% to y: 70..10
                const y = 70 - ((st.apparentHoldoutAcc - 74) / 20) * 60
                return `${x},${y}`
              })
              .join(' ')}
          />

          {/* True Accuracy line */}
          <polyline
            fill="none"
            stroke="#34d399"
            strokeWidth="2"
            points={trajectory
              .map((st) => {
                const x = (st.queryIdx / 60) * 300
                const y = 70 - ((st.trueGenAcc - 74) / 20) * 60
                return `${x},${y}`
              })
              .join(' ')}
          />

          {/* Current query marker dot */}
          {(() => {
            const x = (activeStep.queryIdx / 60) * 300
            const yApp = 70 - ((activeStep.apparentHoldoutAcc - 74) / 20) * 60
            const yTrue = 70 - ((activeStep.trueGenAcc - 74) / 20) * 60
            return (
              <>
                <line
                  x1={x}
                  y1="10"
                  x2={x}
                  y2="70"
                  stroke="rgba(255,255,255,0.2)"
                  strokeWidth="1"
                  strokeDasharray="2 2"
                />
                <circle cx={x} cy={yApp} r="3.5" fill="#38bdf8" />
                <circle cx={x} cy={yTrue} r="3.5" fill="#34d399" />
              </>
            )
          })()}
        </svg>
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
        <span style={{ fontWeight: 600, color: '#f8fafc' }}>The Kaggle Leaderboard Shakeup Phenomenon: </span>
        When participants query a public test set 50+ times, they select hyperparameter seeds that exploit random noise
        fluctuations in the holdout split. When the private test set is revealed at competition end, the apparent 90%
        models suffer massive rank drops, plummeting back to their true 78% baseline.
      </div>
    </div>
  )
}
