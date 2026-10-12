import React, { useMemo, useState } from 'react'

type SearchMethod = 'sfs' | 'sbs' | 'rfe'
type ScenarioKey = 'synergy' | 'noisy'

interface FeatureMeta {
  id: string
  name: string
  nature: string
  trueRelevance: 'primary' | 'synergistic' | 'redundant' | 'noise'
}

interface StepState {
  step: number
  actionDesc: string
  activeFeatures: string[]
  candidateEvaluated?: string
  eliminated?: string
  score: number // CV metric e.g. R^2
  cumulativeFits: number
}

const SCENARIOS: Record<
  ScenarioKey,
  {
    name: string
    description: string
    features: FeatureMeta[]
  }
> = {
  synergy: {
    name: 'Synergistic Pair & Collinear Redundancy',
    description:
      'Features X3 and X4 form an interacting XOR-like pair: weak on their own, but highly predictive together. X2 is collinear with X1. X5, X6 are pure noise.',
    features: [
      { id: 'x1', name: 'X1 (Strong Main)', nature: 'Linear signal (r = 0.65)', trueRelevance: 'primary' },
      { id: 'x2', name: 'X2 (Collinear)', nature: 'Duplicate of X1 + 5% noise', trueRelevance: 'redundant' },
      { id: 'x3', name: 'X3 (Pair Part A)', nature: 'Marginal r = 0.08', trueRelevance: 'synergistic' },
      { id: 'x4', name: 'X4 (Pair Part B)', nature: 'Marginal r = 0.06', trueRelevance: 'synergistic' },
      { id: 'x5', name: 'X5 (White Noise)', nature: 'Gaussian N(0,1)', trueRelevance: 'noise' },
      { id: 'x6', name: 'X6 (White Noise)', nature: 'Gaussian N(0,1)', trueRelevance: 'noise' },
    ],
  },
  noisy: {
    name: 'High-Noise Screening (8 Features)',
    description:
      'Sparse true signal: only X1, X2, and X3 drive the target. The remaining five features are spurious uninformative distractors.',
    features: [
      { id: 'x1', name: 'X1 (Core Factor)', nature: 'Linear weight beta = 2.4', trueRelevance: 'primary' },
      { id: 'x2', name: 'X2 (Core Factor)', nature: 'Linear weight beta = 1.8', trueRelevance: 'primary' },
      { id: 'x3', name: 'X3 (Core Factor)', nature: 'Linear weight beta = 1.2', trueRelevance: 'primary' },
      { id: 'x4', name: 'X4 (Spurious 1)', nature: 'Uncorrelated noise', trueRelevance: 'noise' },
      { id: 'x5', name: 'X5 (Spurious 2)', nature: 'Uncorrelated noise', trueRelevance: 'noise' },
      { id: 'x6', name: 'X6 (Spurious 3)', nature: 'Uncorrelated noise', trueRelevance: 'noise' },
      { id: 'x7', name: 'X7 (Spurious 4)', nature: 'Uncorrelated noise', trueRelevance: 'noise' },
      { id: 'x8', name: 'X8 (Spurious 5)', nature: 'Uncorrelated noise', trueRelevance: 'noise' },
    ],
  },
}

// Simulation trajectories for SFS, SBS, RFE across scenarios
function computeTrajectory(scenarioKey: ScenarioKey, method: SearchMethod): StepState[] {
  if (scenarioKey === 'synergy') {
    if (method === 'sfs') {
      // SFS starts empty.
      // Step 0: Empty
      // Step 1: Tests all 6 -> picks X1 (best marginal score 0.52) - 6 fits
      // Step 2: Tests remaining 5 -> picks X2 (0.55) because X3 & X4 individually look weak - 5 fits
      // Step 3: Tests remaining 4 -> picks X3 (0.57) - 4 fits
      // Step 4: Tests remaining 3 -> picks X4 (0.86! Now synergy activates) - 3 fits
      // Step 5: Tests remaining 2 -> picks X5 (0.84 drops due to noise) - 2 fits
      // Step 6: Adds X6 (0.81 drops) - 1 fit
      return [
        {
          step: 0,
          actionDesc: 'Baseline: empty feature set (predict mean).',
          activeFeatures: [],
          score: 0.0,
          cumulativeFits: 0,
        },
        {
          step: 1,
          actionDesc: 'Screened 6 candidates. Added X1 (highest individual CV R²).',
          activeFeatures: ['x1'],
          score: 0.52,
          cumulativeFits: 6,
        },
        {
          step: 2,
          actionDesc: 'Screened 5 candidates. Added collinear X2 (marginal gain; missed X3-X4 pair!).',
          activeFeatures: ['x1', 'x2'],
          score: 0.55,
          cumulativeFits: 11,
        },
        {
          step: 3,
          actionDesc: 'Screened 4 candidates. Added X3.',
          activeFeatures: ['x1', 'x2', 'x3'],
          score: 0.57,
          cumulativeFits: 15,
        },
        {
          step: 4,
          actionDesc: 'Screened 3 candidates. Added X4 (unlocks synergistic interaction with X3!).',
          activeFeatures: ['x1', 'x2', 'x3', 'x4'],
          score: 0.86,
          cumulativeFits: 18,
        },
        {
          step: 5,
          actionDesc: 'Screened 2 candidates. Added X5 (overfitting to CV fold; performance drops).',
          activeFeatures: ['x1', 'x2', 'x3', 'x4', 'x5'],
          score: 0.83,
          cumulativeFits: 20,
        },
        {
          step: 6,
          actionDesc: 'Added final remaining feature X6.',
          activeFeatures: ['x1', 'x2', 'x3', 'x4', 'x5', 'x6'],
          score: 0.79,
          cumulativeFits: 21,
        },
      ]
    } else if (method === 'sbs') {
      // SBS starts full: {x1, x2, x3, x4, x5, x6}
      // Step 0: Full set -> R2 = 0.79
      // Step 1: Test dropping each of 6 -> drops X6 (pure noise) -> R2 = 0.83 (6 fits)
      // Step 2: Test dropping each of 5 -> drops X5 (noise) -> R2 = 0.86 (5 fits)
      // Step 3: Test dropping each of 4 -> drops X2 (redundant with X1) -> R2 = 0.89 (PEAK! synergistic pair preserved) (4 fits)
      // Step 4: Drops X1 -> R2 = 0.62 (3 fits)
      // Step 5: Drops X4 -> breaks pair -> R2 = 0.04 (2 fits)
      // Step 6: Drops X3 -> R2 = 0.0 (1 fit)
      return [
        {
          step: 0,
          actionDesc: 'Baseline: all 6 features included in full model.',
          activeFeatures: ['x1', 'x2', 'x3', 'x4', 'x5', 'x6'],
          score: 0.79,
          cumulativeFits: 1,
        },
        {
          step: 1,
          actionDesc: 'Evaluated dropping each feature (6 fits). Pruned noise feature X6.',
          activeFeatures: ['x1', 'x2', 'x3', 'x4', 'x5'],
          eliminated: 'x6',
          score: 0.83,
          cumulativeFits: 7,
        },
        {
          step: 2,
          actionDesc: 'Evaluated dropping each feature (5 fits). Pruned noise feature X5.',
          activeFeatures: ['x1', 'x2', 'x3', 'x4'],
          eliminated: 'x5',
          score: 0.86,
          cumulativeFits: 12,
        },
        {
          step: 3,
          actionDesc: 'Evaluated dropping each feature (4 fits). Pruned collinear duplicate X2. Peak R² reached!',
          activeFeatures: ['x1', 'x3', 'x4'],
          eliminated: 'x2',
          score: 0.89,
          cumulativeFits: 16,
        },
        {
          step: 4,
          actionDesc: 'Evaluated dropping each feature (3 fits). Pruned primary feature X1. Severe score drop.',
          activeFeatures: ['x3', 'x4'],
          eliminated: 'x1',
          score: 0.62,
          cumulativeFits: 19,
        },
        {
          step: 5,
          actionDesc: 'Pruned X4, destroying synergistic interaction with X3.',
          activeFeatures: ['x3'],
          eliminated: 'x4',
          score: 0.05,
          cumulativeFits: 21,
        },
        {
          step: 6,
          actionDesc: 'Pruned final feature X3. Empty model.',
          activeFeatures: [],
          eliminated: 'x3',
          score: 0.0,
          cumulativeFits: 22,
        },
      ]
    } else {
      // RFE: 1 model fit per step, ranks by coefficient magnitude
      // Step 0: Full set -> R2 = 0.79 (1 fit)
      // Step 1: Prunes lowest |w| (X6) -> R2 = 0.83 (1 fit)
      // Step 2: Prunes lowest |w| (X5) -> R2 = 0.86 (1 fit)
      // Step 3: Prunes lowest |w| (X2) -> R2 = 0.89 (1 fit)
      // Step 4: Prunes lowest |w| (X1) -> R2 = 0.62 (1 fit)
      // Step 5: Prunes X4 -> R2 = 0.05 (1 fit)
      // Step 6: Empty -> R2 = 0.0
      return [
        {
          step: 0,
          actionDesc: 'Fit full linear/SVM model (1 fit). Ranks features by weight magnitude |w_j|.',
          activeFeatures: ['x1', 'x2', 'x3', 'x4', 'x5', 'x6'],
          score: 0.79,
          cumulativeFits: 1,
        },
        {
          step: 1,
          actionDesc: 'Eliminated feature with smallest weight: X6. Retrained model (1 fit).',
          activeFeatures: ['x1', 'x2', 'x3', 'x4', 'x5'],
          eliminated: 'x6',
          score: 0.83,
          cumulativeFits: 2,
        },
        {
          step: 2,
          actionDesc: 'Eliminated feature with smallest weight: X5. Retrained model (1 fit).',
          activeFeatures: ['x1', 'x2', 'x3', 'x4'],
          eliminated: 'x5',
          score: 0.86,
          cumulativeFits: 3,
        },
        {
          step: 3,
          actionDesc: 'Eliminated collinear X2 (|w_2| attenuated by colinearity). Retrained (1 fit). Peak score!',
          activeFeatures: ['x1', 'x3', 'x4'],
          eliminated: 'x2',
          score: 0.89,
          cumulativeFits: 4,
        },
        {
          step: 4,
          actionDesc: 'Eliminated X1. Retrained (1 fit). Score drops.',
          activeFeatures: ['x3', 'x4'],
          eliminated: 'x1',
          score: 0.62,
          cumulativeFits: 5,
        },
        {
          step: 5,
          actionDesc: 'Eliminated X4. Remaining isolated feature X3 carries negligible signal.',
          activeFeatures: ['x3'],
          eliminated: 'x4',
          score: 0.05,
          cumulativeFits: 6,
        },
        {
          step: 6,
          actionDesc: 'Eliminated last feature. Empty model.',
          activeFeatures: [],
          eliminated: 'x3',
          score: 0.0,
          cumulativeFits: 6,
        },
      ]
    }
  } else {
    // 8-feature noisy scenario
    if (method === 'sfs') {
      return [
        { step: 0, actionDesc: 'Baseline: empty set.', activeFeatures: [], score: 0.0, cumulativeFits: 0 },
        {
          step: 1,
          actionDesc: 'Screened 8 features. Added primary X1 (best R²).',
          activeFeatures: ['x1'],
          score: 0.44,
          cumulativeFits: 8,
        },
        {
          step: 2,
          actionDesc: 'Screened 7 features. Added primary X2.',
          activeFeatures: ['x1', 'x2'],
          score: 0.72,
          cumulativeFits: 15,
        },
        {
          step: 3,
          actionDesc: 'Screened 6 features. Added primary X3. True optimal subset reached!',
          activeFeatures: ['x1', 'x2', 'x3'],
          score: 0.91,
          cumulativeFits: 21,
        },
        {
          step: 4,
          actionDesc: 'Screened 5 candidates. Spurious correlation in CV folds admitted noise X4.',
          activeFeatures: ['x1', 'x2', 'x3', 'x4'],
          score: 0.88,
          cumulativeFits: 26,
        },
        {
          step: 5,
          actionDesc: 'Added spurious noise X7 due to selection optimism.',
          activeFeatures: ['x1', 'x2', 'x3', 'x4', 'x7'],
          score: 0.84,
          cumulativeFits: 30,
        },
        {
          step: 6,
          actionDesc: 'Added spurious noise X5.',
          activeFeatures: ['x1', 'x2', 'x3', 'x4', 'x7', 'x5'],
          score: 0.81,
          cumulativeFits: 33,
        },
        {
          step: 7,
          actionDesc: 'Added spurious noise X8.',
          activeFeatures: ['x1', 'x2', 'x3', 'x4', 'x7', 'x5', 'x8'],
          score: 0.77,
          cumulativeFits: 35,
        },
        {
          step: 8,
          actionDesc: 'All 8 features incorporated.',
          activeFeatures: ['x1', 'x2', 'x3', 'x4', 'x7', 'x5', 'x8', 'x6'],
          score: 0.73,
          cumulativeFits: 36,
        },
      ]
    } else if (method === 'sbs') {
      return [
        {
          step: 0,
          actionDesc: 'Baseline: all 8 features in full model.',
          activeFeatures: ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'],
          score: 0.73,
          cumulativeFits: 1,
        },
        {
          step: 1,
          actionDesc: 'Tested removing each of 8 features (8 fits). Pruned noise X8.',
          activeFeatures: ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'],
          eliminated: 'x8',
          score: 0.77,
          cumulativeFits: 9,
        },
        {
          step: 2,
          actionDesc: 'Tested removing each of 7 features (7 fits). Pruned noise X5.',
          activeFeatures: ['x1', 'x2', 'x3', 'x4', 'x6', 'x7'],
          eliminated: 'x5',
          score: 0.81,
          cumulativeFits: 16,
        },
        {
          step: 3,
          actionDesc: 'Tested removing each of 6 features (6 fits). Pruned noise X6.',
          activeFeatures: ['x1', 'x2', 'x3', 'x4', 'x7'],
          eliminated: 'x6',
          score: 0.84,
          cumulativeFits: 22,
        },
        {
          step: 4,
          actionDesc: 'Tested removing each of 5 features (5 fits). Pruned noise X7.',
          activeFeatures: ['x1', 'x2', 'x3', 'x4'],
          eliminated: 'x7',
          score: 0.88,
          cumulativeFits: 27,
        },
        {
          step: 5,
          actionDesc:
            'Tested removing each of 4 features (4 fits). Pruned noise X4. Reached pure signal set {X1, X2, X3}!',
          activeFeatures: ['x1', 'x2', 'x3'],
          eliminated: 'x4',
          score: 0.91,
          cumulativeFits: 31,
        },
        {
          step: 6,
          actionDesc: 'Pruned informative feature X3.',
          activeFeatures: ['x1', 'x2'],
          eliminated: 'x3',
          score: 0.72,
          cumulativeFits: 34,
        },
        {
          step: 7,
          actionDesc: 'Pruned informative feature X2.',
          activeFeatures: ['x1'],
          eliminated: 'x2',
          score: 0.44,
          cumulativeFits: 36,
        },
        {
          step: 8,
          actionDesc: 'Pruned informative feature X1. Empty model.',
          activeFeatures: [],
          eliminated: 'x1',
          score: 0.0,
          cumulativeFits: 37,
        },
      ]
    } else {
      // RFE
      return [
        {
          step: 0,
          actionDesc: 'Fit full 8-variable model (1 fit). Rank by coefficient weights |w_j|.',
          activeFeatures: ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'],
          score: 0.73,
          cumulativeFits: 1,
        },
        {
          step: 1,
          actionDesc: 'Eliminated noise X6 with lowest weight (1 fit).',
          activeFeatures: ['x1', 'x2', 'x3', 'x4', 'x5', 'x7', 'x8'],
          eliminated: 'x6',
          score: 0.76,
          cumulativeFits: 2,
        },
        {
          step: 2,
          actionDesc: 'Eliminated noise X8 (1 fit).',
          activeFeatures: ['x1', 'x2', 'x3', 'x4', 'x5', 'x7'],
          eliminated: 'x8',
          score: 0.8,
          cumulativeFits: 3,
        },
        {
          step: 3,
          actionDesc: 'Eliminated noise X5 (1 fit).',
          activeFeatures: ['x1', 'x2', 'x3', 'x4', 'x7'],
          eliminated: 'x5',
          score: 0.84,
          cumulativeFits: 4,
        },
        {
          step: 4,
          actionDesc: 'Eliminated noise X7 (1 fit).',
          activeFeatures: ['x1', 'x2', 'x3', 'x4'],
          eliminated: 'x7',
          score: 0.88,
          cumulativeFits: 5,
        },
        {
          step: 5,
          actionDesc: 'Eliminated noise X4 (1 fit). True signal trio {X1, X2, X3} isolated! Peak CV score.',
          activeFeatures: ['x1', 'x2', 'x3'],
          eliminated: 'x4',
          score: 0.91,
          cumulativeFits: 6,
        },
        {
          step: 6,
          actionDesc: 'Eliminated X3 (1 fit). Score drops to 0.72.',
          activeFeatures: ['x1', 'x2'],
          eliminated: 'x3',
          score: 0.72,
          cumulativeFits: 7,
        },
        {
          step: 7,
          actionDesc: 'Eliminated X2 (1 fit). Score drops to 0.44.',
          activeFeatures: ['x1'],
          eliminated: 'x2',
          score: 0.44,
          cumulativeFits: 8,
        },
        {
          step: 8,
          actionDesc: 'Eliminated X1. Empty model.',
          activeFeatures: [],
          eliminated: 'x1',
          score: 0.0,
          cumulativeFits: 8,
        },
      ]
    }
  }
}

export default function WrapperSelectionExplorer(): React.ReactElement {
  const [scenarioKey, setScenarioKey] = useState<ScenarioKey>('synergy')
  const [method, setMethod] = useState<SearchMethod>('sfs')
  const [currentStep, setCurrentStep] = useState<number>(0)

  const scenario = SCENARIOS[scenarioKey]
  const trajectory = useMemo(() => computeTrajectory(scenarioKey, method), [scenarioKey, method])

  // Clamp step if scenario length changed
  const maxStep = trajectory.length - 1
  const safeStep = Math.min(currentStep, maxStep)
  const activeState = trajectory[safeStep] || trajectory[0]

  // Find peak score step
  const peakStepIndex = useMemo(() => {
    let bestIdx = 0
    let bestScore = -Infinity
    trajectory.forEach((s, idx) => {
      if (s.score > bestScore) {
        bestScore = s.score
        bestIdx = idx
      }
    })
    return bestIdx
  }, [trajectory])

  const bestScore = trajectory[peakStepIndex].score

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
      <div style={{ marginBottom: '1rem' }}>
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
            Interactive Wrapper Search Simulator
          </h4>
          <span
            style={{
              fontSize: '0.75rem',
              padding: '0.2rem 0.55rem',
              borderRadius: '999px',
              background: 'rgba(56, 189, 248, 0.15)',
              color: '#38bdf8',
              fontWeight: 500,
            }}
          >
            {scenario.features.length} Features &middot; {method.toUpperCase()}
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
          {scenario.description}
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
              fontSize: '0.75rem',
              fontWeight: 600,
              textTransform: 'uppercase',
              color: '#94a3b8',
              marginBottom: '0.3rem',
            }}
          >
            Problem Scenario
          </label>
          <div style={{ display: 'flex', gap: '0.4rem' }}>
            <button
              type="button"
              onClick={() => {
                setScenarioKey('synergy')
                setCurrentStep(0)
              }}
              style={{
                flex: 1,
                padding: '0.35rem 0.6rem',
                fontSize: '0.78rem',
                borderRadius: '5px',
                border: scenarioKey === 'synergy' ? '1px solid #38bdf8' : '1px solid #444',
                background: scenarioKey === 'synergy' ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
                color: scenarioKey === 'synergy' ? '#38bdf8' : '#aaa',
                cursor: 'pointer',
              }}
            >
              Synergy & Redundancy
            </button>
            <button
              type="button"
              onClick={() => {
                setScenarioKey('noisy')
                setCurrentStep(0)
              }}
              style={{
                flex: 1,
                padding: '0.35rem 0.6rem',
                fontSize: '0.78rem',
                borderRadius: '5px',
                border: scenarioKey === 'noisy' ? '1px solid #38bdf8' : '1px solid #444',
                background: scenarioKey === 'noisy' ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
                color: scenarioKey === 'noisy' ? '#38bdf8' : '#aaa',
                cursor: 'pointer',
              }}
            >
              High-Noise (8 vars)
            </button>
          </div>
        </div>

        <div>
          <label
            style={{
              display: 'block',
              fontSize: '0.75rem',
              fontWeight: 600,
              textTransform: 'uppercase',
              color: '#94a3b8',
              marginBottom: '0.3rem',
            }}
          >
            Wrapper Search Strategy
          </label>
          <div style={{ display: 'flex', gap: '0.35rem' }}>
            {(
              [
                ['sfs', 'SFS (Forward)'],
                ['sbs', 'SBS (Backward)'],
                ['rfe', 'RFE (Recursive)'],
              ] as const
            ).map(([mKey, label]) => (
              <button
                key={mKey}
                type="button"
                onClick={() => {
                  setMethod(mKey)
                  setCurrentStep(0)
                }}
                style={{
                  flex: 1,
                  padding: '0.35rem 0.4rem',
                  fontSize: '0.75rem',
                  borderRadius: '5px',
                  border: method === mKey ? '1px solid #34d399' : '1px solid #444',
                  background: method === mKey ? 'rgba(52, 211, 153, 0.2)' : 'transparent',
                  color: method === mKey ? '#34d399' : '#aaa',
                  cursor: 'pointer',
                }}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Step navigation slider and buttons */}
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
        <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#f8fafc', minWidth: '70px' }}>
          Step {safeStep} / {maxStep}
        </span>
        <input
          type="range"
          min={0}
          max={maxStep}
          value={safeStep}
          onChange={(e) => setCurrentStep(parseInt(e.target.value, 10))}
          style={{ flex: 1, cursor: 'pointer', accentColor: '#38bdf8' }}
        />
        <div style={{ display: 'flex', gap: '0.35rem' }}>
          <button
            type="button"
            disabled={safeStep === 0}
            onClick={() => setCurrentStep((s) => Math.max(0, s - 1))}
            style={{
              padding: '0.3rem 0.6rem',
              fontSize: '0.78rem',
              borderRadius: '4px',
              border: '1px solid #444',
              background: '#222',
              color: safeStep === 0 ? '#666' : '#fff',
              cursor: safeStep === 0 ? 'not-allowed' : 'pointer',
            }}
          >
            &larr; Prev
          </button>
          <button
            type="button"
            disabled={safeStep === maxStep}
            onClick={() => setCurrentStep((s) => Math.min(maxStep, s + 1))}
            style={{
              padding: '0.3rem 0.6rem',
              fontSize: '0.78rem',
              borderRadius: '4px',
              border: '1px solid #444',
              background: '#222',
              color: safeStep === maxStep ? '#666' : '#fff',
              cursor: safeStep === maxStep ? 'not-allowed' : 'pointer',
            }}
          >
            Next &rarr;
          </button>
        </div>
      </div>

      {/* Step description callout */}
      <div
        style={{
          padding: '0.7rem 0.9rem',
          borderRadius: '6px',
          background: 'rgba(56, 189, 248, 0.08)',
          borderLeft: '3px solid #38bdf8',
          marginBottom: '1.25rem',
          fontSize: '0.82rem',
          color: '#e2e8f0',
        }}
      >
        <strong>Step {safeStep}:</strong> {activeState.actionDesc}
      </div>

      {/* Visualization Grid: Feature Status vs Trajectory Chart */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '1.2rem',
          marginBottom: '1.25rem',
        }}
      >
        {/* Left: Feature Chips */}
        <div>
          <div
            style={{
              fontSize: '0.75rem',
              fontWeight: 600,
              textTransform: 'uppercase',
              color: '#94a3b8',
              marginBottom: '0.5rem',
            }}
          >
            Feature State at Step {safeStep} ({activeState.activeFeatures.length} of {scenario.features.length} active)
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
            {scenario.features.map((feat) => {
              const isActive = activeState.activeFeatures.includes(feat.id)
              const isEliminated = activeState.eliminated === feat.id

              let badgeBg = 'rgba(255, 255, 255, 0.04)'
              let badgeBorder = '1px solid rgba(255, 255, 255, 0.08)'
              let badgeColor = '#64748b'

              if (isActive) {
                badgeBg = feat.trueRelevance === 'noise' ? 'rgba(244, 63, 94, 0.15)' : 'rgba(52, 211, 153, 0.15)'
                badgeBorder = feat.trueRelevance === 'noise' ? '1px solid #f43f5e' : '1px solid #34d399'
                badgeColor = feat.trueRelevance === 'noise' ? '#fda4af' : '#6ee7b7'
              } else if (isEliminated) {
                badgeBg = 'rgba(234, 179, 8, 0.1)'
                badgeBorder = '1px solid #eab308'
                badgeColor = '#fde047'
              }

              return (
                <div
                  key={feat.id}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '0.45rem 0.65rem',
                    borderRadius: '6px',
                    background: badgeBg,
                    border: badgeBorder,
                    fontSize: '0.78rem',
                    transition: 'all 0.2s ease',
                  }}
                >
                  <div>
                    <span style={{ fontWeight: 600, color: badgeColor, marginRight: '0.5rem' }}>{feat.name}</span>
                    <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>({feat.nature})</span>
                  </div>
                  <span
                    style={{
                      fontSize: '0.68rem',
                      fontWeight: 600,
                      padding: '0.15rem 0.45rem',
                      borderRadius: '4px',
                      background: isActive
                        ? feat.trueRelevance === 'noise'
                          ? 'rgba(244,63,94,0.3)'
                          : 'rgba(52,211,153,0.3)'
                        : 'rgba(0,0,0,0.3)',
                      color: isActive ? '#fff' : '#64748b',
                    }}
                  >
                    {isActive ? 'IN SUBSET' : isEliminated ? 'JUST DROPPED' : 'EXCLUDED'}
                  </span>
                </div>
              )
            })}
          </div>
        </div>

        {/* Right: Validation Score Curve & Compute Metrics */}
        <div>
          <div
            style={{
              fontSize: '0.75rem',
              fontWeight: 600,
              textTransform: 'uppercase',
              color: '#94a3b8',
              marginBottom: '0.5rem',
            }}
          >
            Cross-Validated Performance Trajectory
          </div>

          {/* Mini step graph */}
          <div
            style={{
              padding: '0.75rem',
              borderRadius: '7px',
              background: 'rgba(0, 0, 0, 0.3)',
              border: '1px solid rgba(255, 255, 255, 0.05)',
              marginBottom: '0.85rem',
            }}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: '0.72rem',
                color: '#94a3b8',
                marginBottom: '0.4rem',
              }}
            >
              <span>Step-by-Step Validation R²</span>
              <span>
                Best: <strong style={{ color: '#34d399' }}>{bestScore.toFixed(2)}</strong> (Step {peakStepIndex})
              </span>
            </div>

            {/* Bar/Curve representation */}
            <div
              style={{ display: 'flex', alignItems: 'flex-end', gap: '0.35rem', height: '100px', paddingTop: '10px' }}
            >
              {trajectory.map((st, idx) => {
                const heightPct = Math.max(8, Math.round(st.score * 95))
                const isSelected = idx === safeStep
                const isPeak = idx === peakStepIndex

                let barColor = 'rgba(255, 255, 255, 0.15)'
                if (isPeak) barColor = '#34d399'
                if (isSelected) barColor = '#38bdf8'

                return (
                  <div
                    key={st.step}
                    onClick={() => setCurrentStep(idx)}
                    style={{
                      flex: 1,
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      gap: '0.25rem',
                      height: '100%',
                      justifyContent: 'flex-end',
                      cursor: 'pointer',
                    }}
                    title={`Step ${idx}: R² = ${st.score.toFixed(2)}, Fits = ${st.cumulativeFits}`}
                  >
                    <span style={{ fontSize: '0.62rem', color: isSelected ? '#38bdf8' : '#64748b' }}>
                      {st.score.toFixed(2)}
                    </span>
                    <div
                      style={{
                        width: '100%',
                        height: `${heightPct}%`,
                        background: barColor,
                        borderRadius: '3px 3px 0 0',
                        border: isSelected ? '1px solid #38bdf8' : 'none',
                        transition: 'height 0.2s ease',
                      }}
                    />
                    <span
                      style={{
                        fontSize: '0.65rem',
                        color: isSelected ? '#fff' : '#64748b',
                        fontWeight: isSelected ? 700 : 400,
                      }}
                    >
                      s{idx}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Stats Bar */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: '0.5rem',
              fontSize: '0.78rem',
            }}
          >
            <div style={{ padding: '0.5rem 0.65rem', borderRadius: '5px', background: 'rgba(255, 255, 255, 0.03)' }}>
              <span style={{ color: '#94a3b8', display: 'block', fontSize: '0.68rem' }}>Cumulative Model Fits</span>
              <strong style={{ fontSize: '1rem', color: '#f59e0b' }}>{activeState.cumulativeFits}</strong> fits
            </div>
            <div style={{ padding: '0.5rem 0.65rem', borderRadius: '5px', background: 'rgba(255, 255, 255, 0.03)' }}>
              <span style={{ color: '#94a3b8', display: 'block', fontSize: '0.68rem' }}>Current Score (R²)</span>
              <strong style={{ fontSize: '1rem', color: safeStep === peakStepIndex ? '#34d399' : '#38bdf8' }}>
                {activeState.score.toFixed(2)}
              </strong>
            </div>
          </div>
        </div>
      </div>

      {/* Analytical Takeaways */}
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
        <span style={{ fontWeight: 600, color: '#f8fafc' }}>Algorithmic Insight: </span>
        {method === 'sfs' &&
          'SFS builds subsets greedily from scratch. Notice how in the synergy scenario, X3 and X4 have weak marginal scores, so SFS selects redundant X2 first. Forward selection struggles with high-order feature synergy unless step sizes or lookaheads are expanded.'}
        {method === 'sbs' &&
          'SBS starts with all features in the model, allowing full interaction effects to be expressed. It readily drops redundant X2 and preserves the synergistic pair {X3, X4}. However, SBS requires fitting models on the full feature space, making it impossible when p > n.'}
        {method === 'rfe' &&
          'RFE uses the internal weight rankings of a single model fit per step to prune features, requiring far fewer total fits (O(p) vs O(p²)) while retaining near-optimal selection quality.'}
      </div>
    </div>
  )
}
