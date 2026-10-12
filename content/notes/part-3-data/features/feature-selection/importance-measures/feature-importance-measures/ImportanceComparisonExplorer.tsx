import React, { useState } from 'react'

type Scenario = 'cardinality' | 'collinear'
type MeasureKey = 'mdi' | 'mda' | 'dropcol' | 'conditional'

interface FeatureImportanceResult {
  id: string
  name: string
  description: string
  mdiScore: number
  mdaScore: number
  dropColScore: number
  condScore: number
  isNoise: boolean
}

const SCENARIO_DATA: Record<
  Scenario,
  {
    title: string
    description: string
    features: FeatureImportanceResult[]
  }
> = {
  cardinality: {
    title: 'High-Cardinality Trap (Noise vs Signal)',
    description:
      'A dataset with 1 true binary signal, 1 high-cardinality random ID (pure noise with 80 unique levels), and 1 continuous Gaussian noise variable.',
    features: [
      {
        id: 'signal',
        name: 'X_signal (True Binary)',
        description: 'Ground-truth binary driver (r = 0.72 with Y)',
        mdiScore: 0.35,
        mdaScore: 0.78,
        dropColScore: 0.75,
        condScore: 0.77,
        isNoise: false,
      },
      {
        id: 'cat_noise',
        name: 'X_id (Random 80 Levels)',
        description: 'Uniform random ID numbers (pure noise, 0 true signal)',
        mdiScore: 0.55, // SEVERE MDI BIAS! Overfitting in-sample splits
        mdaScore: -0.02,
        dropColScore: 0.0,
        condScore: -0.01,
        isNoise: true,
      },
      {
        id: 'cont_noise',
        name: 'X_noise (Continuous Gaussian)',
        description: 'White noise N(0,1) offering many threshold cutpoints',
        mdiScore: 0.1,
        mdaScore: 0.01,
        dropColScore: 0.0,
        condScore: 0.0,
        isNoise: true,
      },
    ],
  },
  collinear: {
    title: 'Collinear Masking & Redundancy',
    description:
      'Features X1 and X2 are collinear duplicates (r = 0.95). X3 is an independent secondary predictor. Observe how drop-column conceals collinear drivers.',
    features: [
      {
        id: 'x1',
        name: 'X1 (Primary Driver)',
        description: 'Core linear signal; r = 0.95 correlated with X2',
        mdiScore: 0.42,
        mdaScore: 0.28,
        dropColScore: 0.03, // Drop-column masked by X2!
        condScore: 0.38,
        isNoise: false,
      },
      {
        id: 'x2',
        name: 'X2 (Collinear Duplicate)',
        description: 'Duplicate readout of X1 + 5% sensor noise',
        mdiScore: 0.38,
        mdaScore: 0.24,
        dropColScore: 0.02, // Masked by X1!
        condScore: 0.34,
        isNoise: false,
      },
      {
        id: 'x3',
        name: 'X3 (Independent Signal)',
        description: 'Orthogonal secondary feature (r = 0.40 with Y)',
        mdiScore: 0.2,
        mdaScore: 0.48,
        dropColScore: 0.45,
        condScore: 0.47,
        isNoise: false,
      },
    ],
  },
}

const MEASURE_INFO: Record<
  MeasureKey,
  {
    name: string
    subtitle: string
    computation: string
    collinearBehavior: string
  }
> = {
  mdi: {
    name: 'MDI (Impurity / Gini)',
    subtitle: 'In-Sample Split Impurity Reduction',
    computation:
      'Calculated entirely on training data during tree fitting. Accumulates weighted variance or Gini drops for all splits on the feature.',
    collinearBehavior:
      'Severely biased toward features with many cutpoints (continuous or high-cardinality discrete). Overfits training data.',
  },
  mda: {
    name: 'MDA (Permutation)',
    subtitle: 'Out-of-Bag / Validation Accuracy Drop',
    computation:
      'Evaluated on unseen validation data. Shuffles values of feature j across instances and computes performance degradation ΔLoss.',
    collinearBehavior:
      'Unbiased by cardinality. However, permuting collinear features generates off-manifold impossible points, which can distort rankings.',
  },
  dropcol: {
    name: 'Drop-Column',
    subtitle: 'Model Retrained Without Column',
    computation:
      'The gold standard for marginal necessity: removes column j completely and trains a brand new model from scratch.',
    collinearBehavior:
      'Exhibits severe collinear masking! When X1 is dropped, X2 effortlessly substitutes for it, making both appear uninformative (score ≈ 0).',
  },
  conditional: {
    name: 'Conditional Permutation',
    subtitle: 'Stratified Permutation within Subspaces',
    computation:
      'Permutes feature j only among samples that share similar values in correlated conditioning variables Z.',
    collinearBehavior:
      'Preserves joint distribution geometry. Disentangles redundant collinear effects without creating impossible off-manifold samples.',
  },
}

export default function ImportanceComparisonExplorer(): React.ReactElement {
  const [scenarioKey, setScenarioKey] = useState<Scenario>('cardinality')
  const [selectedMeasure, setSelectedMeasure] = useState<MeasureKey>('mdi')

  const currentScenario = SCENARIO_DATA[scenarioKey]
  const measureDetails = MEASURE_INFO[selectedMeasure]

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
            Feature Importance Pathology & Benchmark Explorer
          </h4>
          <span
            style={{
              fontSize: '0.74rem',
              padding: '0.2rem 0.55rem',
              borderRadius: '999px',
              background: 'rgba(56, 189, 248, 0.15)',
              color: '#38bdf8',
              fontWeight: 500,
            }}
          >
            {currentScenario.title}
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
          {currentScenario.description}
        </p>
      </div>

      {/* Scenario Selector */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.1rem' }}>
        <button
          type="button"
          onClick={() => setScenarioKey('cardinality')}
          style={{
            flex: 1,
            padding: '0.45rem 0.75rem',
            fontSize: '0.78rem',
            borderRadius: '6px',
            border: scenarioKey === 'cardinality' ? '1px solid #38bdf8' : '1px solid #444',
            background: scenarioKey === 'cardinality' ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
            color: scenarioKey === 'cardinality' ? '#38bdf8' : '#aaa',
            cursor: 'pointer',
            fontWeight: scenarioKey === 'cardinality' ? 600 : 400,
          }}
        >
          1. High-Cardinality Bias Trap
        </button>
        <button
          type="button"
          onClick={() => setScenarioKey('collinear')}
          style={{
            flex: 1,
            padding: '0.45rem 0.75rem',
            fontSize: '0.78rem',
            borderRadius: '6px',
            border: scenarioKey === 'collinear' ? '1px solid #38bdf8' : '1px solid #444',
            background: scenarioKey === 'collinear' ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
            color: scenarioKey === 'collinear' ? '#38bdf8' : '#aaa',
            cursor: 'pointer',
            fontWeight: scenarioKey === 'collinear' ? 600 : 400,
          }}
        >
          2. Collinear Masking & Redundancy
        </button>
      </div>

      {/* Measure Toggle Row */}
      <div style={{ marginBottom: '1.25rem' }}>
        <label
          style={{
            display: 'block',
            fontSize: '0.72rem',
            fontWeight: 600,
            textTransform: 'uppercase',
            color: '#94a3b8',
            marginBottom: '0.4rem',
          }}
        >
          Select Importance Measure to Inspect
        </label>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '0.4rem' }}>
          {(
            [
              ['mdi', 'MDI (Impurity)'],
              ['mda', 'MDA (Permutation)'],
              ['dropcol', 'Drop-Column'],
              ['conditional', 'Conditional MDA'],
            ] as const
          ).map(([mKey, label]) => (
            <button
              key={mKey}
              type="button"
              onClick={() => setSelectedMeasure(mKey)}
              style={{
                padding: '0.4rem 0.5rem',
                fontSize: '0.75rem',
                borderRadius: '5px',
                border: selectedMeasure === mKey ? '1px solid #34d399' : '1px solid #444',
                background: selectedMeasure === mKey ? 'rgba(52, 211, 153, 0.2)' : 'transparent',
                color: selectedMeasure === mKey ? '#34d399' : '#94a3b8',
                cursor: 'pointer',
                fontWeight: selectedMeasure === mKey ? 600 : 400,
              }}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Main Comparative Chart */}
      <div
        style={{
          padding: '1rem',
          borderRadius: '8px',
          background: 'rgba(0, 0, 0, 0.28)',
          border: '1px solid rgba(255, 255, 255, 0.06)',
          marginBottom: '1.25rem',
        }}
      >
        <div
          style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}
        >
          <div>
            <strong style={{ fontSize: '0.85rem', color: '#f8fafc' }}>{measureDetails.name}</strong>
            <span style={{ fontSize: '0.74rem', color: '#94a3b8', marginLeft: '0.5rem' }}>
              &mdash; {measureDetails.subtitle}
            </span>
          </div>
          <span style={{ fontSize: '0.72rem', color: '#64748b' }}>Normalized Score [0, 1]</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {currentScenario.features.map((feat) => {
            let score = 0
            if (selectedMeasure === 'mdi') score = feat.mdiScore
            else if (selectedMeasure === 'mda') score = feat.mdaScore
            else if (selectedMeasure === 'dropcol') score = feat.dropColScore
            else score = feat.condScore

            const isPathological =
              (selectedMeasure === 'mdi' && feat.id === 'cat_noise' && score > 0.4) ||
              (selectedMeasure === 'dropcol' && (feat.id === 'x1' || feat.id === 'x2') && score < 0.1)

            const barWidth = Math.min(100, Math.max(3, Math.round(score * 100)))
            const barColor = feat.isNoise
              ? isPathological
                ? '#f43f5e'
                : '#64748b'
              : isPathological
                ? '#f59e0b'
                : '#38bdf8'

            return (
              <div key={feat.id}>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '0.25rem',
                    fontSize: '0.78rem',
                  }}
                >
                  <div>
                    <span style={{ fontWeight: 600, color: feat.isNoise ? '#fda4af' : '#e2e8f0' }}>{feat.name}</span>
                    <span style={{ fontSize: '0.7rem', color: '#94a3b8', marginLeft: '0.4rem' }}>
                      ({feat.description})
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    {isPathological && (
                      <span
                        style={{
                          fontSize: '0.64rem',
                          fontWeight: 700,
                          padding: '0.1rem 0.35rem',
                          borderRadius: '3px',
                          background: feat.isNoise ? 'rgba(244, 63, 94, 0.25)' : 'rgba(245, 158, 11, 0.25)',
                          color: feat.isNoise ? '#f43f5e' : '#f59e0b',
                        }}
                      >
                        {feat.isNoise ? 'FALSE DISCOVERY' : 'MASKED TO ~0'}
                      </span>
                    )}
                    <strong style={{ color: barColor, fontSize: '0.85rem' }}>{score.toFixed(2)}</strong>
                  </div>
                </div>

                {/* Progress bar */}
                <div
                  style={{
                    width: '100%',
                    height: '8px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    borderRadius: '4px',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      width: `${barWidth}%`,
                      height: '100%',
                      background: barColor,
                      borderRadius: '4px',
                      transition: 'width 0.25s ease',
                    }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Explanatory callouts */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
          gap: '0.75rem',
          fontSize: '0.78rem',
          lineHeight: 1.45,
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
          <strong style={{ display: 'block', color: '#38bdf8', marginBottom: '0.2rem' }}>How It Works:</strong>
          <span style={{ color: '#cbd5e1' }}>{measureDetails.computation}</span>
        </div>
        <div
          style={{
            padding: '0.75rem',
            borderRadius: '6px',
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid rgba(255, 255, 255, 0.05)',
          }}
        >
          <strong style={{ display: 'block', color: '#f59e0b', marginBottom: '0.2rem' }}>Pathology & Pitfalls:</strong>
          <span style={{ color: '#cbd5e1' }}>{measureDetails.collinearBehavior}</span>
        </div>
      </div>
    </div>
  )
}
