import React, { useState } from 'react'

type ProtocolMode = 'flawed' | 'honest'
type DataRegime = 'pure_noise' | 'weak_signal'

interface SimulationResult {
  cvReportedAccuracy: number
  trueTestAccuracy: number
  optimismGap: number
  featureOverlapPct: number
  summaryExplanation: string
}

const SIMULATIONS: Record<DataRegime, Record<ProtocolMode, SimulationResult>> = {
  pure_noise: {
    flawed: {
      cvReportedAccuracy: 95.8,
      trueTestAccuracy: 50.1,
      optimismGap: 45.7,
      featureOverlapPct: 100.0,
      summaryExplanation:
        'Catastrophic selection bias! Screening all 2,000 pure noise features on the entire dataset finds spurious correlations that perfectly overfit validation folds, creating a phantom 95.8% accuracy.',
    },
    honest: {
      cvReportedAccuracy: 49.6,
      trueTestAccuracy: 50.2,
      optimismGap: -0.6,
      featureOverlapPct: 1.4,
      summaryExplanation:
        'Honest nested evaluation correctly reveals the truth: random guessing (49.6%). Because selection ran strictly inside each training fold, held-out validation samples remain completely untainted.',
    },
  },
  weak_signal: {
    flawed: {
      cvReportedAccuracy: 88.4,
      trueTestAccuracy: 64.2,
      optimismGap: 24.2,
      featureOverlapPct: 92.0,
      summaryExplanation:
        'Severe optimism: the true signal is modest (64.2%), but unshielded selection picks up sample-specific noise alongside the signal, inflating reported CV performance by +24.2%.',
    },
    honest: {
      cvReportedAccuracy: 63.8,
      trueTestAccuracy: 64.5,
      optimismGap: -0.7,
      featureOverlapPct: 58.0,
      summaryExplanation:
        'Calibrated generalization: the reported CV score (63.8%) closely mirrors actual out-of-sample deployment accuracy (64.5%). Only stable signals survive across folds.',
    },
  },
}

export default function SelectionBiasCvVisualizer(): React.ReactElement {
  const [dataRegime, setDataRegime] = useState<DataRegime>('pure_noise')
  const [protocol, setProtocol] = useState<ProtocolMode>('flawed')

  const currentResult = SIMULATIONS[dataRegime][protocol]

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
            Selection Bias & Cross-Validation Leakage Simulator
          </h4>
          <span
            style={{
              fontSize: '0.74rem',
              padding: '0.2rem 0.55rem',
              borderRadius: '999px',
              background: protocol === 'flawed' ? 'rgba(244, 63, 94, 0.15)' : 'rgba(52, 211, 153, 0.15)',
              color: protocol === 'flawed' ? '#f43f5e' : '#34d399',
              fontWeight: 600,
            }}
          >
            {protocol === 'flawed' ? 'LEAKED (OUTSIDE CV)' : 'VALID (INSIDE FOLDS)'}
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
          {dataRegime === 'pure_noise'
            ? 'Demonstration on n = 60 samples and p = 2,000 pure Gaussian noise features (true Bayes accuracy = 50.0%).'
            : 'Demonstration on n = 60 samples with p = 2,000 features containing 5 weak signals and 1,995 noise distractors.'}
        </p>
      </div>

      {/* Control row */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '0.75rem',
          marginBottom: '1.25rem',
          padding: '0.75rem',
          borderRadius: '8px',
          background: 'rgba(0, 0, 0, 0.25)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
        }}
      >
        <div>
          <label
            style={{
              display: 'block',
              fontSize: '0.72rem',
              fontWeight: 600,
              textTransform: 'uppercase',
              color: '#94a3b8',
              marginBottom: '0.35rem',
            }}
          >
            Data Generation Setup
          </label>
          <div style={{ display: 'flex', gap: '0.4rem' }}>
            <button
              type="button"
              onClick={() => setDataRegime('pure_noise')}
              style={{
                flex: 1,
                padding: '0.35rem 0.6rem',
                fontSize: '0.76rem',
                borderRadius: '5px',
                border: dataRegime === 'pure_noise' ? '1px solid #38bdf8' : '1px solid #444',
                background: dataRegime === 'pure_noise' ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
                color: dataRegime === 'pure_noise' ? '#38bdf8' : '#aaa',
                cursor: 'pointer',
              }}
            >
              Pure Noise (50% Truth)
            </button>
            <button
              type="button"
              onClick={() => setDataRegime('weak_signal')}
              style={{
                flex: 1,
                padding: '0.35rem 0.6rem',
                fontSize: '0.76rem',
                borderRadius: '5px',
                border: dataRegime === 'weak_signal' ? '1px solid #38bdf8' : '1px solid #444',
                background: dataRegime === 'weak_signal' ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
                color: dataRegime === 'weak_signal' ? '#38bdf8' : '#aaa',
                cursor: 'pointer',
              }}
            >
              Weak Signal + 2k Noise
            </button>
          </div>
        </div>

        <div>
          <label
            style={{
              display: 'block',
              fontSize: '0.72rem',
              fontWeight: 600,
              textTransform: 'uppercase',
              color: '#94a3b8',
              marginBottom: '0.35rem',
            }}
          >
            Validation Protocol
          </label>
          <div style={{ display: 'flex', gap: '0.4rem' }}>
            <button
              type="button"
              onClick={() => setProtocol('flawed')}
              style={{
                flex: 1,
                padding: '0.35rem 0.6rem',
                fontSize: '0.76rem',
                borderRadius: '5px',
                border: protocol === 'flawed' ? '1px solid #f43f5e' : '1px solid #444',
                background: protocol === 'flawed' ? 'rgba(244, 63, 94, 0.2)' : 'transparent',
                color: protocol === 'flawed' ? '#f43f5e' : '#aaa',
                cursor: 'pointer',
              }}
            >
              Outside CV (Flawed)
            </button>
            <button
              type="button"
              onClick={() => setProtocol('honest')}
              style={{
                flex: 1,
                padding: '0.35rem 0.6rem',
                fontSize: '0.76rem',
                borderRadius: '5px',
                border: protocol === 'honest' ? '1px solid #34d399' : '1px solid #444',
                background: protocol === 'honest' ? 'rgba(52, 211, 153, 0.2)' : 'transparent',
                color: protocol === 'honest' ? '#34d399' : '#aaa',
                cursor: 'pointer',
              }}
            >
              Inside Folds (Honest)
            </button>
          </div>
        </div>
      </div>

      {/* Pipeline Diagram */}
      <div
        style={{
          padding: '0.85rem',
          borderRadius: '8px',
          background: 'rgba(0, 0, 0, 0.35)',
          border: '1px solid rgba(255, 255, 255, 0.06)',
          marginBottom: '1.25rem',
        }}
      >
        <div
          style={{
            fontSize: '0.72rem',
            fontWeight: 600,
            textTransform: 'uppercase',
            color: '#94a3b8',
            marginBottom: '0.6rem',
          }}
        >
          Data Flow Architecture
        </div>
        {protocol === 'flawed' ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', fontSize: '0.78rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
              <span
                style={{
                  padding: '0.3rem 0.6rem',
                  borderRadius: '4px',
                  background: 'rgba(255,255,255,0.08)',
                  color: '#fff',
                }}
              >
                Full Dataset (n=60, p=2000)
              </span>
              <span style={{ color: '#f43f5e' }}>&rarr;</span>
              <span
                style={{
                  padding: '0.3rem 0.6rem',
                  borderRadius: '4px',
                  background: 'rgba(244,63,94,0.25)',
                  color: '#fda4af',
                  border: '1px solid #f43f5e',
                }}
              >
                Feature Screening (Top 15 picked using ALL rows!)
              </span>
              <span style={{ color: '#f43f5e' }}>&rarr;</span>
              <span
                style={{
                  padding: '0.3rem 0.6rem',
                  borderRadius: '4px',
                  background: 'rgba(255,255,255,0.08)',
                  color: '#cbd5e1',
                }}
              >
                Cross-Validation (Folds 1..5)
              </span>
            </div>
            <div style={{ fontSize: '0.72rem', color: '#f43f5e', marginTop: '0.2rem' }}>
              &Delta; LEAKAGE DETECTED: The validation labels were used to choose the top 15 features before fold
              splitting!
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', fontSize: '0.78rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
              <span
                style={{
                  padding: '0.3rem 0.6rem',
                  borderRadius: '4px',
                  background: 'rgba(255,255,255,0.08)',
                  color: '#fff',
                }}
              >
                Full Dataset
              </span>
              <span style={{ color: '#34d399' }}>&rarr;</span>
              <span
                style={{
                  padding: '0.3rem 0.6rem',
                  borderRadius: '4px',
                  background: 'rgba(52,211,153,0.15)',
                  color: '#6ee7b7',
                  border: '1px solid #34d399',
                }}
              >
                Split Folds 1..5
              </span>
              <span style={{ color: '#34d399' }}>&rarr;</span>
              <span
                style={{
                  padding: '0.3rem 0.6rem',
                  borderRadius: '4px',
                  background: 'rgba(52,211,153,0.25)',
                  color: '#a7f3d0',
                }}
              >
                [Train Fold: Screen Top 15 & Fit Model]
              </span>
              <span style={{ color: '#34d399' }}>&rarr;</span>
              <span
                style={{
                  padding: '0.3rem 0.6rem',
                  borderRadius: '4px',
                  background: 'rgba(255,255,255,0.08)',
                  color: '#cbd5e1',
                }}
              >
                [Val Fold: Unbiased Test Evaluation]
              </span>
            </div>
            <div style={{ fontSize: '0.72rem', color: '#34d399', marginTop: '0.2rem' }}>
              &check; SHIELDED PROTOCOL: Validation folds are strictly held out prior to feature ranking and model
              estimation.
            </div>
          </div>
        )}
      </div>

      {/* Metrics Comparison Grid */}
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
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.05)',
          }}
        >
          <span style={{ display: 'block', fontSize: '0.68rem', color: '#94a3b8', textTransform: 'uppercase' }}>
            Reported CV Accuracy
          </span>
          <strong style={{ fontSize: '1.35rem', color: protocol === 'flawed' ? '#f43f5e' : '#34d399' }}>
            {currentResult.cvReportedAccuracy.toFixed(1)}%
          </strong>
          <span style={{ display: 'block', fontSize: '0.7rem', color: '#64748b', marginTop: '0.2rem' }}>
            What the data scientist sees
          </span>
        </div>

        <div
          style={{
            padding: '0.75rem',
            borderRadius: '6px',
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.05)',
          }}
        >
          <span style={{ display: 'block', fontSize: '0.68rem', color: '#94a3b8', textTransform: 'uppercase' }}>
            True Test Accuracy
          </span>
          <strong style={{ fontSize: '1.35rem', color: '#38bdf8' }}>
            {currentResult.trueTestAccuracy.toFixed(1)}%
          </strong>
          <span style={{ display: 'block', fontSize: '0.7rem', color: '#64748b', marginTop: '0.2rem' }}>
            Actual performance in production
          </span>
        </div>

        <div
          style={{
            padding: '0.75rem',
            borderRadius: '6px',
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.05)',
          }}
        >
          <span style={{ display: 'block', fontSize: '0.68rem', color: '#94a3b8', textTransform: 'uppercase' }}>
            Optimism Bias (Gap)
          </span>
          <strong
            style={{ fontSize: '1.35rem', color: Math.abs(currentResult.optimismGap) > 5 ? '#f59e0b' : '#34d399' }}
          >
            {currentResult.optimismGap > 0
              ? `+${currentResult.optimismGap.toFixed(1)}%`
              : `${currentResult.optimismGap.toFixed(1)}%`}
          </strong>
          <span style={{ display: 'block', fontSize: '0.7rem', color: '#64748b', marginTop: '0.2rem' }}>
            {Math.abs(currentResult.optimismGap) > 5 ? 'Severe Overconfidence' : 'Accurately Calibrated'}
          </span>
        </div>

        <div
          style={{
            padding: '0.75rem',
            borderRadius: '6px',
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.05)',
          }}
        >
          <span style={{ display: 'block', fontSize: '0.68rem', color: '#94a3b8', textTransform: 'uppercase' }}>
            Cross-Fold Subset Overlap
          </span>
          <strong style={{ fontSize: '1.35rem', color: '#cbd5e1' }}>
            {currentResult.featureOverlapPct.toFixed(1)}%
          </strong>
          <span style={{ display: 'block', fontSize: '0.7rem', color: '#64748b', marginTop: '0.2rem' }}>
            {protocol === 'flawed' ? 'Identical subset forced on all folds' : 'Varies as folds isolate local noise'}
          </span>
        </div>
      </div>

      {/* Explanatory callout */}
      <div
        style={{
          padding: '0.8rem 0.95rem',
          borderRadius: '7px',
          background: protocol === 'flawed' ? 'rgba(244, 63, 94, 0.08)' : 'rgba(52, 211, 153, 0.08)',
          borderLeft: protocol === 'flawed' ? '3px solid #f43f5e' : '3px solid #34d399',
          fontSize: '0.82rem',
          color: '#e2e8f0',
          lineHeight: 1.5,
        }}
      >
        <strong>Diagnostic Summary:</strong> {currentResult.summaryExplanation}
      </div>
    </div>
  )
}
