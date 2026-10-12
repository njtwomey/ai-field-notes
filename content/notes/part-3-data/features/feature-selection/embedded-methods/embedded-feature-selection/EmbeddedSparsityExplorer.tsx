import React, { useMemo, useState } from 'react'

type PenaltyType = 'lasso' | 'ridge' | 'elasticnet'
type ViewMode = 'path' | 'geometry'

interface FeatureTrajectory {
  name: string
  olsWeight: number
  color: string
  role: string
  computeWeight: (lambda: number, penalty: PenaltyType) => number
}

const FEATURES: FeatureTrajectory[] = [
  {
    name: 'X1 (Primary Factor)',
    olsWeight: 2.8,
    color: '#38bdf8',
    role: 'Strong primary signal',
    computeWeight: (lambda, penalty) => {
      if (penalty === 'lasso') return Math.max(0, 2.8 - lambda * 2.8)
      if (penalty === 'ridge') return 2.8 / (1 + lambda * 2.5)
      return Math.max(0, (2.8 - lambda * 1.5) / (1 + lambda * 1.2))
    },
  },
  {
    name: 'X2 (Moderate Factor)',
    olsWeight: 1.6,
    color: '#34d399',
    role: 'Moderate independent signal',
    computeWeight: (lambda, penalty) => {
      if (penalty === 'lasso') return Math.max(0, 1.6 - lambda * 3.2)
      if (penalty === 'ridge') return 1.6 / (1 + lambda * 2.5)
      return Math.max(0, (1.6 - lambda * 1.8) / (1 + lambda * 1.2))
    },
  },
  {
    name: 'X3 (Collinear Shadow)',
    olsWeight: 1.2,
    color: '#a78bfa',
    role: 'High correlation with X1 (r = 0.88)',
    computeWeight: (lambda, penalty) => {
      if (penalty === 'lasso') {
        // Lasso abruptly zeros out one collinear variable early
        return lambda > 0.18 ? 0 : Math.max(0, 1.2 - lambda * 6.5)
      }
      if (penalty === 'ridge') {
        // Ridge splits weight evenly across collinear pairs
        return 1.2 / (1 + lambda * 2.5)
      }
      // Elastic Net preserves grouping
      return Math.max(0, (1.2 - lambda * 2.0) / (1 + lambda * 1.4))
    },
  },
  {
    name: 'X4 (Weak Feature)',
    olsWeight: 0.65,
    color: '#fbbf24',
    role: 'Weak secondary factor',
    computeWeight: (lambda, penalty) => {
      if (penalty === 'lasso') return Math.max(0, 0.65 - lambda * 3.5)
      if (penalty === 'ridge') return 0.65 / (1 + lambda * 2.5)
      return Math.max(0, (0.65 - lambda * 2.2) / (1 + lambda * 1.2))
    },
  },
  {
    name: 'X5 (Spurious Noise)',
    olsWeight: 0.35,
    color: '#f43f5e',
    role: 'Pure Gaussian noise',
    computeWeight: (lambda, penalty) => {
      if (penalty === 'lasso') return Math.max(0, 0.35 - lambda * 4.5)
      if (penalty === 'ridge') return 0.35 / (1 + lambda * 2.8)
      return Math.max(0, (0.35 - lambda * 3.0) / (1 + lambda * 1.5))
    },
  },
]

export default function EmbeddedSparsityExplorer(): React.ReactElement {
  const [viewMode, setViewMode] = useState<ViewMode>('path')
  const [penalty, setPenalty] = useState<PenaltyType>('lasso')
  const [lambda, setLambda] = useState<number>(0.25)
  const [budgetC, setBudgetC] = useState<number>(1.2)

  // Compute active weights for path view
  const currentWeights = useMemo(() => {
    return FEATURES.map((f) => ({
      ...f,
      weight: f.computeWeight(lambda, penalty),
    }))
  }, [lambda, penalty])

  const activeCount = currentWeights.filter((f) => Math.abs(f.weight) > 0.001).length

  // Geometry calculations: 2D plane (w1, w2)
  // Unconstrained OLS center = (w1_hat, w2_hat) = (2.0, 1.4)
  const olsCenter = { x: 2.0, y: 1.4 }
  // Under L1: constraint is |w1| + |w2| <= budgetC
  // With elliptical loss (w1-2)^2 + 2*(w2-1.4)^2 = const,
  // the first contact point with diamond for budgetC in [0.5, 2.5] often hits the vertex (budgetC, 0)
  // if budgetC is moderate, say budgetC <= 2.0.
  const l1Tangent = useMemo(() => {
    // If budget is large enough, touches interior or edge
    if (budgetC >= 3.4) return { ...olsCenter, isCorner: false }
    // For elliptical contours, corner hit at (budgetC, 0) is typical when budget < 2.0
    if (budgetC <= 2.0) {
      return { x: budgetC, y: 0.0, isCorner: true }
    }
    // Edge hit between (2.0, 0) and (0, 2.0)
    const t = budgetC - 2.0
    return { x: 2.0 - t * 0.5, y: t * 0.5, isCorner: false }
  }, [budgetC, olsCenter])

  // Under L2: constraint is w1^2 + w2^2 <= budgetC^2
  // Tangent point on circle is smooth shrink along gradient vector
  const l2Tangent = useMemo(() => {
    const r = budgetC
    const normCenter = Math.sqrt(olsCenter.x * olsCenter.x + olsCenter.y * olsCenter.y)
    if (r >= normCenter) return olsCenter
    const scale = r / normCenter
    return { x: olsCenter.x * scale, y: olsCenter.y * scale, isCorner: false }
  }, [budgetC, olsCenter])

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
            Embedded Regularization & Sparsity Explorer
          </h4>
          <div style={{ display: 'flex', gap: '0.35rem' }}>
            <button
              type="button"
              onClick={() => setViewMode('path')}
              style={{
                padding: '0.25rem 0.65rem',
                fontSize: '0.75rem',
                borderRadius: '5px',
                border: viewMode === 'path' ? '1px solid #38bdf8' : '1px solid #444',
                background: viewMode === 'path' ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
                color: viewMode === 'path' ? '#38bdf8' : '#94a3b8',
                cursor: 'pointer',
              }}
            >
              Regularization Path
            </button>
            <button
              type="button"
              onClick={() => setViewMode('geometry')}
              style={{
                padding: '0.25rem 0.65rem',
                fontSize: '0.75rem',
                borderRadius: '5px',
                border: viewMode === 'geometry' ? '1px solid #38bdf8' : '1px solid #444',
                background: viewMode === 'geometry' ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
                color: viewMode === 'geometry' ? '#38bdf8' : '#94a3b8',
                cursor: 'pointer',
              }}
            >
              Loss Geometry (2D)
            </button>
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
          {viewMode === 'path'
            ? 'Observe how increasing penalty parameter λ drives non-essential coefficients strictly to zero in Lasso, while Ridge retains all variables.'
            : 'Visualize why the sharp corners of the L1 diamond set coefficients exactly to zero on coordinate axes, while the smooth L2 sphere produces dense shrinkage.'}
        </p>
      </div>

      {viewMode === 'path' ? (
        <>
          {/* Controls row */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
              gap: '0.85rem',
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
                Penalty Formulation
              </label>
              <div style={{ display: 'flex', gap: '0.35rem' }}>
                {(
                  [
                    ['lasso', 'Lasso (L1)'],
                    ['ridge', 'Ridge (L2)'],
                    ['elasticnet', 'Elastic Net (α = 0.5)'],
                  ] as const
                ).map(([pKey, label]) => (
                  <button
                    key={pKey}
                    type="button"
                    onClick={() => setPenalty(pKey)}
                    style={{
                      flex: 1,
                      padding: '0.35rem 0.4rem',
                      fontSize: '0.74rem',
                      borderRadius: '5px',
                      border: penalty === pKey ? '1px solid #38bdf8' : '1px solid #444',
                      background: penalty === pKey ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
                      color: penalty === pKey ? '#38bdf8' : '#aaa',
                      cursor: 'pointer',
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                <label style={{ fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', color: '#94a3b8' }}>
                  Regularization Strength (λ): <span style={{ color: '#38bdf8' }}>{lambda.toFixed(2)}</span>
                </label>
                <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>
                  Active Features:{' '}
                  <strong style={{ color: activeCount < 5 ? '#34d399' : '#f59e0b' }}>{activeCount} / 5</strong>
                </span>
              </div>
              <input
                type="range"
                min={0.0}
                max={0.8}
                step={0.01}
                value={lambda}
                onChange={(e) => setLambda(parseFloat(e.target.value))}
                style={{ width: '100%', cursor: 'pointer', accentColor: '#38bdf8' }}
              />
            </div>
          </div>

          {/* Feature Coefficients Bar & active status */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', marginBottom: '1.25rem' }}>
            {currentWeights.map((feat) => {
              const isZero = Math.abs(feat.weight) < 0.001
              const barWidthPct = Math.min(100, Math.round((feat.weight / 2.8) * 100))

              return (
                <div
                  key={feat.name}
                  style={{
                    padding: '0.55rem 0.75rem',
                    borderRadius: '6px',
                    background: isZero ? 'rgba(255, 255, 255, 0.02)' : 'rgba(255, 255, 255, 0.05)',
                    border: isZero ? '1px dashed rgba(255, 255, 255, 0.1)' : '1px solid rgba(255, 255, 255, 0.08)',
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
                          color: isZero ? '#64748b' : feat.color,
                          marginRight: '0.45rem',
                        }}
                      >
                        {feat.name}
                      </span>
                      <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>({feat.role})</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ fontSize: '0.78rem', fontWeight: 700, color: isZero ? '#64748b' : '#f8fafc' }}>
                        w = {feat.weight.toFixed(2)}
                      </span>
                      <span
                        style={{
                          fontSize: '0.65rem',
                          padding: '0.1rem 0.35rem',
                          borderRadius: '3px',
                          background: isZero ? 'rgba(244, 63, 94, 0.2)' : 'rgba(52, 211, 153, 0.2)',
                          color: isZero ? '#fda4af' : '#6ee7b7',
                        }}
                      >
                        {isZero ? 'ZERO (DROPPED)' : 'ACTIVE'}
                      </span>
                    </div>
                  </div>

                  {/* Magnitude Bar */}
                  <div
                    style={{
                      width: '100%',
                      height: '7px',
                      background: 'rgba(0, 0, 0, 0.35)',
                      borderRadius: '3px',
                      overflow: 'hidden',
                    }}
                  >
                    <div
                      style={{
                        width: `${barWidthPct}%`,
                        height: '100%',
                        background: isZero ? 'transparent' : feat.color,
                        borderRadius: '3px',
                        transition: 'width 0.15s ease',
                      }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </>
      ) : (
        <>
          {/* Geometry View (2D) */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '1rem',
              marginBottom: '1rem',
              padding: '0.65rem 0.9rem',
              borderRadius: '7px',
              background: 'rgba(255, 255, 255, 0.03)',
            }}
          >
            <label style={{ fontSize: '0.78rem', fontWeight: 600, color: '#f8fafc', minWidth: '160px' }}>
              Constraint Budget (C): <span style={{ color: '#38bdf8' }}>{budgetC.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0.4}
              max={2.6}
              step={0.05}
              value={budgetC}
              onChange={(e) => setBudgetC(parseFloat(e.target.value))}
              style={{ flex: 1, cursor: 'pointer', accentColor: '#38bdf8' }}
            />
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
              gap: '1rem',
              marginBottom: '1.25rem',
            }}
          >
            {/* L1 Geometry SVG */}
            <div
              style={{
                padding: '0.85rem',
                borderRadius: '8px',
                background: 'rgba(0, 0, 0, 0.3)',
                border: '1px solid rgba(255, 255, 255, 0.08)',
              }}
            >
              <div style={{ fontSize: '0.8rem', fontWeight: 600, color: '#38bdf8', marginBottom: '0.4rem' }}>
                L1 Penalty (Diamond: |w₁| + |w₂| ≤ C)
              </div>
              <svg viewBox="0 0 240 240" style={{ width: '100%', height: 'auto', display: 'block' }}>
                {/* Axes */}
                <line x1="20" y1="120" x2="220" y2="120" stroke="#475569" strokeWidth="1" />
                <line x1="120" y1="20" x2="120" y2="220" stroke="#475569" strokeWidth="1" />
                <text x="215" y="115" fill="#64748b" fontSize="10">
                  w₁
                </text>
                <text x="125" y="30" fill="#64748b" fontSize="10">
                  w₂
                </text>

                {/* L1 Diamond */}
                {(() => {
                  const scale = 38
                  const cPix = budgetC * scale
                  const points = `120,${120 - cPix} ${120 + cPix},120 120,${120 + cPix} ${120 - cPix},120`
                  return <polygon points={points} fill="rgba(56, 189, 248, 0.15)" stroke="#38bdf8" strokeWidth="1.5" />
                })()}

                {/* OLS Unconstrained Optimum */}
                {(() => {
                  const scale = 38
                  const olsX = 120 + olsCenter.x * scale
                  const olsY = 120 - olsCenter.y * scale
                  const solX = 120 + l1Tangent.x * scale
                  const solY = 120 - l1Tangent.y * scale
                  const rx = Math.abs(solX - olsX)
                  const ry = Math.abs(solY - olsY) || 12

                  return (
                    <>
                      {/* Loss Ellipse touching diamond */}
                      <ellipse
                        cx={olsX}
                        cy={olsY}
                        rx={Math.max(15, rx)}
                        ry={Math.max(12, ry * 1.3)}
                        fill="none"
                        stroke="#f59e0b"
                        strokeWidth="1.2"
                        strokeDasharray="3 3"
                      />
                      {/* OLS Point */}
                      <circle cx={olsX} cy={olsY} r="3.5" fill="#f59e0b" />
                      <text x={olsX + 6} y={olsY - 4} fill="#f59e0b" fontSize="9">
                        OLS (ŵ)
                      </text>

                      {/* Optimal constrained solution */}
                      <circle cx={solX} cy={solY} r="4.5" fill="#ef4444" />
                      <text x={solX + 6} y={solY + 12} fill="#ef4444" fontSize="9" fontWeight="bold">
                        {l1Tangent.isCorner ? 'Sparse vertex (w₂ = 0)' : 'Contact point'}
                      </text>
                    </>
                  )
                })()}
              </svg>
              <div style={{ fontSize: '0.72rem', color: '#94a3b8', marginTop: '0.4rem' }}>
                Solution: w₁ = <strong style={{ color: '#fff' }}>{l1Tangent.x.toFixed(2)}</strong>, w₂ ={' '}
                <strong style={{ color: l1Tangent.y === 0 ? '#34d399' : '#fff' }}>{l1Tangent.y.toFixed(2)}</strong>
                {l1Tangent.y === 0 && ' (Feature 2 completely excluded!)'}
              </div>
            </div>

            {/* L2 Geometry SVG */}
            <div
              style={{
                padding: '0.85rem',
                borderRadius: '8px',
                background: 'rgba(0, 0, 0, 0.3)',
                border: '1px solid rgba(255, 255, 255, 0.08)',
              }}
            >
              <div style={{ fontSize: '0.8rem', fontWeight: 600, color: '#34d399', marginBottom: '0.4rem' }}>
                L2 Penalty (Circle: w₁² + w₂² ≤ C²)
              </div>
              <svg viewBox="0 0 240 240" style={{ width: '100%', height: 'auto', display: 'block' }}>
                {/* Axes */}
                <line x1="20" y1="120" x2="220" y2="120" stroke="#475569" strokeWidth="1" />
                <line x1="120" y1="20" x2="120" y2="220" stroke="#475569" strokeWidth="1" />
                <text x="215" y="115" fill="#64748b" fontSize="10">
                  w₁
                </text>
                <text x="125" y="30" fill="#64748b" fontSize="10">
                  w₂
                </text>

                {/* L2 Circle */}
                {(() => {
                  const scale = 38
                  const cPix = budgetC * scale
                  return (
                    <circle
                      cx="120"
                      cy="120"
                      r={cPix}
                      fill="rgba(52, 211, 153, 0.15)"
                      stroke="#34d399"
                      strokeWidth="1.5"
                    />
                  )
                })()}

                {/* OLS Unconstrained Optimum */}
                {(() => {
                  const scale = 38
                  const olsX = 120 + olsCenter.x * scale
                  const olsY = 120 - olsCenter.y * scale
                  const solX = 120 + l2Tangent.x * scale
                  const solY = 120 - l2Tangent.y * scale
                  const dist = Math.hypot(solX - olsX, solY - olsY)

                  return (
                    <>
                      {/* Loss Ellipse touching circle */}
                      <ellipse
                        cx={olsX}
                        cy={olsY}
                        rx={Math.max(15, dist)}
                        ry={Math.max(12, dist * 1.2)}
                        fill="none"
                        stroke="#f59e0b"
                        strokeWidth="1.2"
                        strokeDasharray="3 3"
                      />
                      {/* OLS Point */}
                      <circle cx={olsX} cy={olsY} r="3.5" fill="#f59e0b" />
                      <text x={olsX + 6} y={olsY - 4} fill="#f59e0b" fontSize="9">
                        OLS (ŵ)
                      </text>

                      {/* Optimal constrained solution */}
                      <circle cx={solX} cy={solY} r="4.5" fill="#38bdf8" />
                      <text x={solX + 6} y={solY + 12} fill="#38bdf8" fontSize="9" fontWeight="bold">
                        Smooth tangent (both w₁, w₂ &gt; 0)
                      </text>
                    </>
                  )
                })()}
              </svg>
              <div style={{ fontSize: '0.72rem', color: '#94a3b8', marginTop: '0.4rem' }}>
                Solution: w₁ = <strong style={{ color: '#fff' }}>{l2Tangent.x.toFixed(2)}</strong>, w₂ ={' '}
                <strong style={{ color: '#fff' }}>{l2Tangent.y.toFixed(2)}</strong> (Neither is zero)
              </div>
            </div>
          </div>
        </>
      )}

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
        {penalty === 'lasso' &&
          'Lasso performs continuous feature selection because its non-differentiable L1 norm has singular corners at the coordinate axes. When the elliptical contours of the RSS loss expand, they frequently touch a corner first, exactly zeroing out features.'}
        {penalty === 'ridge' &&
          'Ridge distributes shrinkage smoothly across all features. Because the L2 ball has no corners or singular points, the loss contour almost never contacts an axis coordinate, meaning Ridge never yields true sparse feature selection.'}
        {penalty === 'elasticnet' &&
          'Elastic Net combines L1 and L2 penalties. The L1 component enforces sparsity, while the strictly convex L2 component ensures that collinear features share weights and enter or exit the active set as a cohesive group.'}
      </div>
    </div>
  )
}
