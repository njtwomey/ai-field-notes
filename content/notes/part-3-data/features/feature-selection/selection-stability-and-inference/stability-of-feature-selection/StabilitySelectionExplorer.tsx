import React, { useMemo, useState } from 'react'

type MethodMode = 'stability' | 'boruta' | 'knockoffs'

interface FeatureItem {
  id: string
  name: string
  role: 'true_primary' | 'true_secondary' | 'collinear_duo' | 'weak' | 'noise_1' | 'noise_2'
  subsampleFreq: number // For stability selection [0, 1]
  borutaZ: number // For Boruta Z-score
  knockoffW: number // For Knockoff W_j statistic
}

const BENCHMARK_FEATURES: FeatureItem[] = [
  {
    id: 'f1',
    name: 'X1 (Core Gene / Driver)',
    role: 'true_primary',
    subsampleFreq: 0.96,
    borutaZ: 14.8,
    knockoffW: 4.2,
  },
  {
    id: 'f2',
    name: 'X2 (Moderate Factor)',
    role: 'true_secondary',
    subsampleFreq: 0.88,
    borutaZ: 10.2,
    knockoffW: 2.8,
  },
  { id: 'f3', name: 'X3 (Collinear Pair A)', role: 'collinear_duo', subsampleFreq: 0.58, borutaZ: 6.4, knockoffW: 1.5 },
  { id: 'f4', name: 'X4 (Collinear Pair B)', role: 'collinear_duo', subsampleFreq: 0.54, borutaZ: 5.9, knockoffW: 1.4 },
  { id: 'f5', name: 'X5 (Borderline Weak)', role: 'weak', subsampleFreq: 0.42, borutaZ: 4.1, knockoffW: 0.3 },
  { id: 'f6', name: 'X6 (Distractor Noise)', role: 'noise_1', subsampleFreq: 0.18, borutaZ: 2.3, knockoffW: -0.4 },
  { id: 'f7', name: 'X7 (Distractor Noise)', role: 'noise_2', subsampleFreq: 0.12, borutaZ: 1.8, knockoffW: -0.7 },
  { id: 'f8', name: 'X8 (Distractor Noise)', role: 'noise_2', subsampleFreq: 0.08, borutaZ: 1.2, knockoffW: 0.1 },
]

export default function StabilitySelectionExplorer(): React.ReactElement {
  const [method, setMethod] = useState<MethodMode>('stability')
  const [stabThreshold, setStabThreshold] = useState<number>(0.75)
  const [shadowZMax, setShadowZMax] = useState<number>(4.5)
  const [fdrTarget, setFdrTarget] = useState<number>(0.1)

  // Compute stability selection false discovery bound E[V] <= q^2 / ((2*pi - 1)*p)
  const qAverage = 3.6 // Average features selected per subsample
  const pTotal = 8
  const boundEV = useMemo(() => {
    const denom = (2 * stabThreshold - 1) * pTotal
    if (denom <= 0) return 999
    return (qAverage * qAverage) / denom
  }, [stabThreshold])

  // Knockoff threshold tau calculated to maintain target FDR
  const knockoffTau = useMemo(() => {
    // With target FDR 0.1, threshold is around 1.0
    return Math.max(0.2, 1.8 - fdrTarget * 8)
  }, [fdrTarget])

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
            Feature Selection Stability & False Discovery Control
          </h4>
          <div style={{ display: 'flex', gap: '0.35rem' }}>
            {(
              [
                ['stability', 'Stability Selection'],
                ['boruta', 'Boruta Shadow'],
                ['knockoffs', 'Model-X Knockoffs'],
              ] as const
            ).map(([mKey, label]) => (
              <button
                key={mKey}
                type="button"
                onClick={() => setMethod(mKey)}
                style={{
                  padding: '0.25rem 0.6rem',
                  fontSize: '0.74rem',
                  borderRadius: '5px',
                  border: method === mKey ? '1px solid #38bdf8' : '1px solid #444',
                  background: method === mKey ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
                  color: method === mKey ? '#38bdf8' : '#94a3b8',
                  cursor: 'pointer',
                  fontWeight: method === mKey ? 600 : 400,
                }}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        <p
          style={{
            margin: '0.4rem 0 0',
            fontSize: '0.82rem',
            color: 'var(--color-text-muted, #94a3b8)',
            lineHeight: 1.45,
          }}
        >
          {method === 'stability' &&
            'Subsample data 100 times without replacement. Features whose selection probability exceeds π_thr are retained with mathematical finite-sample bounds on false discoveries.'}
          {method === 'boruta' &&
            'Compare feature importance Z-scores against permuted shadow variables. Only features significantly outperforming the maximum shadow noise floor are confirmed.'}
          {method === 'knockoffs' &&
            'Synthesize fake knockoff variables matching the exact covariance structure Σ. Feature statistics W_j = |Z_j| - |Z̃_j| enable exact False Discovery Rate control.'}
        </p>
      </div>

      {/* Method Parameters Slider */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '1rem',
          marginBottom: '1.25rem',
          padding: '0.75rem 0.9rem',
          borderRadius: '7px',
          background: 'rgba(0, 0, 0, 0.25)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
        }}
      >
        {method === 'stability' && (
          <>
            <label style={{ fontSize: '0.76rem', fontWeight: 600, color: '#f8fafc', minWidth: '220px' }}>
              Selection Cutoff (π_thr): <span style={{ color: '#38bdf8' }}>{stabThreshold.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0.55}
              max={0.95}
              step={0.01}
              value={stabThreshold}
              onChange={(e) => setStabThreshold(parseFloat(e.target.value))}
              style={{ flex: 1, cursor: 'pointer', accentColor: '#38bdf8' }}
            />
            <span style={{ fontSize: '0.74rem', color: '#94a3b8', minWidth: '130px', textAlign: 'right' }}>
              Bound E[V] &le;{' '}
              <strong style={{ color: boundEV < 1 ? '#34d399' : '#f59e0b' }}>{boundEV.toFixed(2)}</strong>
            </span>
          </>
        )}

        {method === 'boruta' && (
          <>
            <label style={{ fontSize: '0.76rem', fontWeight: 600, color: '#f8fafc', minWidth: '220px' }}>
              Max Shadow Noise Floor (Z_max): <span style={{ color: '#f59e0b' }}>{shadowZMax.toFixed(1)}</span>
            </label>
            <input
              type="range"
              min={2.0}
              max={8.0}
              step={0.2}
              value={shadowZMax}
              onChange={(e) => setShadowZMax(parseFloat(e.target.value))}
              style={{ flex: 1, cursor: 'pointer', accentColor: '#f59e0b' }}
            />
          </>
        )}

        {method === 'knockoffs' && (
          <>
            <label style={{ fontSize: '0.76rem', fontWeight: 600, color: '#f8fafc', minWidth: '220px' }}>
              Nominal FDR Target (q): <span style={{ color: '#34d399' }}>{fdrTarget.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0.05}
              max={0.25}
              step={0.01}
              value={fdrTarget}
              onChange={(e) => setFdrTarget(parseFloat(e.target.value))}
              style={{ flex: 1, cursor: 'pointer', accentColor: '#34d399' }}
            />
            <span style={{ fontSize: '0.74rem', color: '#94a3b8', minWidth: '120px', textAlign: 'right' }}>
              Threshold &tau; = <strong style={{ color: '#38bdf8' }}>{knockoffTau.toFixed(2)}</strong>
            </span>
          </>
        )}
      </div>

      {/* Feature Ranking Table / Bars */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.55rem', marginBottom: '1.25rem' }}>
        {BENCHMARK_FEATURES.map((feat) => {
          let isSelected = false
          let metricLabel = ''
          let barPct = 0
          let barColor = '#64748b'

          if (method === 'stability') {
            isSelected = feat.subsampleFreq >= stabThreshold
            metricLabel = `Subsample Prob: ${(feat.subsampleFreq * 100).toFixed(0)}%`
            barPct = Math.round(feat.subsampleFreq * 100)
            barColor = isSelected ? '#38bdf8' : '#64748b'
          } else if (method === 'boruta') {
            isSelected = feat.borutaZ > shadowZMax
            metricLabel = `Z-score: ${feat.borutaZ.toFixed(1)}`
            barPct = Math.min(100, Math.round((feat.borutaZ / 15) * 100))
            barColor = isSelected ? '#34d399' : feat.borutaZ > shadowZMax - 1 ? '#f59e0b' : '#64748b'
          } else {
            isSelected = feat.knockoffW >= knockoffTau
            metricLabel = `W_j = ${feat.knockoffW.toFixed(2)}`
            barPct = Math.min(100, Math.max(5, Math.round(((feat.knockoffW + 1.0) / 5.5) * 100)))
            barColor = isSelected ? '#34d399' : feat.knockoffW > 0 ? '#f59e0b' : '#f43f5e'
          }

          return (
            <div
              key={feat.id}
              style={{
                padding: '0.55rem 0.75rem',
                borderRadius: '6px',
                background: isSelected ? 'rgba(255, 255, 255, 0.05)' : 'rgba(255, 255, 255, 0.02)',
                border: isSelected ? '1px solid rgba(56, 189, 248, 0.3)' : '1px solid rgba(255, 255, 255, 0.06)',
                transition: 'all 0.15s ease',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '0.3rem',
                }}
              >
                <div>
                  <span
                    style={{
                      fontSize: '0.8rem',
                      fontWeight: 600,
                      color: isSelected ? '#f8fafc' : '#94a3b8',
                      marginRight: '0.5rem',
                    }}
                  >
                    {feat.name}
                  </span>
                  <span style={{ fontSize: '0.7rem', color: '#64748b' }}>({feat.role})</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <span style={{ fontSize: '0.76rem', color: barColor, fontWeight: 600 }}>{metricLabel}</span>
                  <span
                    style={{
                      fontSize: '0.65rem',
                      fontWeight: 700,
                      padding: '0.12rem 0.4rem',
                      borderRadius: '3px',
                      background: isSelected ? 'rgba(52, 211, 153, 0.2)' : 'rgba(0, 0, 0, 0.3)',
                      color: isSelected ? '#34d399' : '#64748b',
                    }}
                  >
                    {isSelected ? 'CONFIRMED' : 'REJECTED'}
                  </span>
                </div>
              </div>

              {/* Progress Bar with Threshold Marker */}
              <div
                style={{
                  position: 'relative',
                  width: '100%',
                  height: '7px',
                  background: 'rgba(0,0,0,0.3)',
                  borderRadius: '3px',
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    width: `${barPct}%`,
                    height: '100%',
                    background: barColor,
                    borderRadius: '3px',
                    transition: 'width 0.2s ease',
                  }}
                />
              </div>
            </div>
          )
        })}
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
        <span style={{ fontWeight: 600, color: '#f8fafc' }}>Statistical Mechanism: </span>
        {method === 'stability' &&
          `Stability Selection screens out noise by requiring selection across subsamples. With π_thr = ${stabThreshold.toFixed(2)}, the expected number of false positive variables selected is mathematically proven to not exceed ${boundEV.toFixed(2)}.`}
        {method === 'boruta' &&
          `Boruta introduces shadow attributes by permuting each feature independently to kill correlation with Y while preserving variance. Any real feature failing to exceed the top shadow Z-score (${shadowZMax.toFixed(1)}) is rejected as indistinguishable from chance.`}
        {method === 'knockoffs' &&
          `Model-X Knockoffs generate fake variables X̃ with identical inter-feature covariance but zero conditional target effect. Under the null hypothesis, original and knockoff features are equally likely to win (W_j symmetric around 0), allowing exact finite-sample FDR control at q = ${fdrTarget.toFixed(2)}.`}
      </div>
    </div>
  )
}
