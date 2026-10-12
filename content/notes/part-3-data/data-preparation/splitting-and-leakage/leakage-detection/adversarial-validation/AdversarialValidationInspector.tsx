import React, { useMemo, useState } from 'react'

type ShiftScenario = 'temporal' | 'group' | 'iid'

interface FeatureDrift {
  id: string
  name: string
  type: string
  driftStrength: number // 0 (no drift) to 1 (massive distribution difference)
  description: string
}

const SCENARIOS: Record<
  ShiftScenario,
  {
    title: string
    description: string
    features: FeatureDrift[]
  }
> = {
  temporal: {
    title: 'Temporal Drift (3-Month Future Test Set)',
    description:
      'Test instances were recorded 3 months after training data. Date-derived features, macro interest rates, and seasonal volume have shifted dramatically.',
    features: [
      {
        id: 'date_seq',
        name: 'transaction_timestamp',
        type: 'Temporal',
        driftStrength: 0.92,
        description: 'Monotonically increasing chronological timestamp',
      },
      {
        id: 'macro_rate',
        name: 'benchmark_interest_rate',
        type: 'Macro',
        driftStrength: 0.78,
        description: 'Central bank rate drifted from 2.5% to 4.2%',
      },
      {
        id: 'weekly_vol',
        name: 'weekly_app_volume',
        type: 'Volume',
        driftStrength: 0.45,
        description: 'User traffic growth over quarter',
      },
      {
        id: 'user_age',
        name: 'account_holder_age',
        type: 'Demographic',
        driftStrength: 0.05,
        description: 'Stable user demographic distribution',
      },
      {
        id: 'credit_score',
        name: 'bureau_credit_score',
        type: 'Risk',
        driftStrength: 0.06,
        description: 'Stable credit scoring distribution',
      },
      {
        id: 'device_os',
        name: 'operating_system_id',
        type: 'Technical',
        driftStrength: 0.12,
        description: 'Slight iOS update version shift',
      },
    ],
  },
  group: {
    title: 'Geographic / Entity Clustered Shift',
    description:
      'Train set contains European enterprise clients; test set evaluates newly onboarded North American mid-market customers.',
    features: [
      {
        id: 'client_geo',
        name: 'datacenter_region_code',
        type: 'Entity',
        driftStrength: 0.95,
        description: 'Completely disjoint regional identifier',
      },
      {
        id: 'currency',
        name: 'billing_currency',
        type: 'Billing',
        driftStrength: 0.88,
        description: 'EUR vs USD billing concentration',
      },
      {
        id: 'firm_size',
        name: 'enterprise_employee_count',
        type: 'Firmographic',
        driftStrength: 0.55,
        description: 'Enterprise (train) vs Mid-market (test)',
      },
      {
        id: 'session_len',
        name: 'avg_session_duration_sec',
        type: 'Behavioral',
        driftStrength: 0.08,
        description: 'Core user workflow interaction speed',
      },
      {
        id: 'error_rate',
        name: 'api_error_response_rate',
        type: 'Reliability',
        driftStrength: 0.04,
        description: 'Underlying backend system error rate',
      },
    ],
  },
  iid: {
    title: 'Uniform Stratified Random Split (i.i.d.)',
    description:
      'Train and test sets were created via stratified random splitting across identical sampling frames. Marginal and joint distributions match.',
    features: [
      {
        id: 'feat_1',
        name: 'user_engagement_score',
        type: 'Behavioral',
        driftStrength: 0.03,
        description: 'Uniformly sampled continuous engagement',
      },
      {
        id: 'feat_2',
        name: 'transaction_amount_log',
        type: 'Financial',
        driftStrength: 0.04,
        description: 'Identical monetary value distribution',
      },
      {
        id: 'feat_3',
        name: 'account_tenure_months',
        type: 'Tenure',
        driftStrength: 0.02,
        description: 'Identical tenure spread',
      },
      {
        id: 'feat_4',
        name: 'platform_category',
        type: 'Categorical',
        driftStrength: 0.03,
        description: 'Matched category proportions',
      },
    ],
  },
}

export default function AdversarialValidationInspector(): React.ReactElement {
  const [scenarioKey, setScenarioKey] = useState<ShiftScenario>('temporal')
  const [activeFeatures, setActiveFeatures] = useState<Record<string, boolean>>({})

  const scenario = SCENARIOS[scenarioKey]

  // Initialize features if scenario changes
  const isFeatureActive = (id: string) => {
    return activeFeatures[id] !== undefined ? activeFeatures[id] : true
  }

  const toggleFeature = (id: string) => {
    setActiveFeatures((prev) => ({
      ...prev,
      [id]: prev[id] !== undefined ? !prev[id] : false,
    }))
  }

  // Compute adversarial validation AUC dynamically based on active features
  const { auc } = useMemo(() => {
    const included = scenario.features.filter((f) => isFeatureActive(f.id))
    if (included.length === 0) {
      return { auc: 0.5 }
    }

    // Maximum drift feature dominates adversarial separation
    const maxDrift = Math.max(...included.map((f) => f.driftStrength))
    const meanDrift = included.reduce((acc, f) => acc + f.driftStrength, 0) / included.length

    // Synthetic AUC model: 0.50 (random) up to 0.99
    const computedAuc = Math.min(0.99, 0.5 + 0.49 * (0.75 * maxDrift + 0.25 * meanDrift))

    return { auc: computedAuc }
  }, [scenario, activeFeatures])

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
            Adversarial Validation & Distribution Drift Inspector
          </h4>
          <span
            style={{
              fontSize: '0.75rem',
              padding: '0.2rem 0.55rem',
              borderRadius: '999px',
              background: auc > 0.7 ? 'rgba(244, 63, 94, 0.15)' : 'rgba(52, 211, 153, 0.15)',
              color: auc > 0.7 ? '#f43f5e' : '#34d399',
              fontWeight: 600,
            }}
          >
            Adversarial AUC: {auc.toFixed(3)}
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

      {/* Scenario Buttons */}
      <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '1.25rem', flexWrap: 'wrap' }}>
        {(
          [
            ['temporal', '1. Temporal Future Shift'],
            ['group', '2. Clustered Entity Shift'],
            ['iid', '3. Uniform i.i.d. Split'],
          ] as const
        ).map(([sKey, label]) => (
          <button
            key={sKey}
            type="button"
            onClick={() => {
              setScenarioKey(sKey)
              setActiveFeatures({})
            }}
            style={{
              flex: 1,
              minWidth: '150px',
              padding: '0.4rem 0.6rem',
              fontSize: '0.76rem',
              borderRadius: '5px',
              border: scenarioKey === sKey ? '1px solid #38bdf8' : '1px solid #444',
              background: scenarioKey === sKey ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
              color: scenarioKey === sKey ? '#38bdf8' : '#aaa',
              cursor: 'pointer',
              fontWeight: scenarioKey === sKey ? 600 : 400,
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Diagnostic Gauge & Interpretation */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
          gap: '1rem',
          marginBottom: '1.25rem',
        }}
      >
        {/* Left: AUC Dial */}
        <div
          style={{
            padding: '1rem',
            borderRadius: '8px',
            background: 'rgba(0, 0, 0, 0.3)',
            border: '1px solid rgba(255, 255, 255, 0.05)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <span style={{ fontSize: '0.72rem', color: '#94a3b8', textTransform: 'uppercase', marginBottom: '0.4rem' }}>
            Train vs Test Classifier Separability
          </span>
          <div
            style={{
              fontSize: '2.2rem',
              fontWeight: 700,
              color: auc > 0.75 ? '#f43f5e' : auc > 0.6 ? '#f59e0b' : '#34d399',
            }}
          >
            {auc.toFixed(3)}
          </div>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#e2e8f0', marginTop: '0.2rem' }}>
            {auc < 0.58
              ? 'Indistinguishable (i.i.d. safe)'
              : auc < 0.75
                ? 'Moderate Distribution Drift'
                : 'Severe Train-Test Separation!'}
          </span>
          <span style={{ fontSize: '0.68rem', color: '#64748b', textAlign: 'center', marginTop: '0.3rem' }}>
            Target: 0.500 (Random guess = identical train/test)
          </span>
        </div>

        {/* Right: Actionable Strategy Guidance */}
        <div
          style={{
            padding: '1rem',
            borderRadius: '8px',
            background: 'rgba(0, 0, 0, 0.3)',
            border: '1px solid rgba(255, 255, 255, 0.05)',
            fontSize: '0.78rem',
            lineHeight: 1.5,
            color: '#cbd5e1',
          }}
        >
          <strong style={{ color: '#f8fafc', display: 'block', marginBottom: '0.3rem' }}>
            Strategic Recommendation:
          </strong>
          {auc > 0.75 ? (
            <>
              Standard random cross-validation will drastically overstate deployment accuracy.
              <ul style={{ margin: '0.4rem 0 0', paddingLeft: '1.2rem', color: '#fda4af' }}>
                <li>
                  Drop top-ranking adversarial features below that represent raw temporal coordinates or leaked entity
                  IDs.
                </li>
                <li>Weight training samples by propensity ratio w_i = P(test|x_i) / P(train|x_i).</li>
                <li>Construct validation folds by sampling training instances most similar to test.</li>
              </ul>
            </>
          ) : (
            <>
              The train and test distributions are well aligned. Random or stratified cross-validation will faithfully
              reflect generalization performance.
            </>
          )}
        </div>
      </div>

      {/* Feature Culprits & Interactive Pruning Table */}
      <div style={{ marginBottom: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
          <label style={{ fontSize: '0.74rem', fontWeight: 600, textTransform: 'uppercase', color: '#94a3b8' }}>
            Adversarial Classifier Feature Reliance (Click feature to toggle / drop)
          </label>
          <span style={{ fontSize: '0.7rem', color: '#64748b' }}>
            Active: {scenario.features.filter((f) => isFeatureActive(f.id)).length} / {scenario.features.length}
          </span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
          {scenario.features.map((feat) => {
            const active = isFeatureActive(feat.id)
            const barWidth = Math.min(100, Math.round(feat.driftStrength * 100))

            return (
              <div
                key={feat.id}
                onClick={() => toggleFeature(feat.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '0.5rem 0.75rem',
                  borderRadius: '6px',
                  background: active ? 'rgba(255, 255, 255, 0.04)' : 'rgba(255, 255, 255, 0.01)',
                  border: active ? '1px solid rgba(255, 255, 255, 0.08)' : '1px dashed #333',
                  opacity: active ? 1 : 0.45,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
                title="Click to include or exclude from model"
              >
                <div style={{ minWidth: '180px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span style={{ fontSize: '0.8rem', fontWeight: 600, color: active ? '#f8fafc' : '#64748b' }}>
                      {feat.name}
                    </span>
                    <span
                      style={{
                        fontSize: '0.68rem',
                        padding: '0.05rem 0.3rem',
                        borderRadius: '3px',
                        background: 'rgba(255,255,255,0.08)',
                        color: '#94a3b8',
                      }}
                    >
                      {feat.type}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '0.15rem' }}>{feat.description}</div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', minWidth: '160px' }}>
                  <div
                    style={{
                      flex: 1,
                      height: '6px',
                      background: 'rgba(0,0,0,0.4)',
                      borderRadius: '3px',
                      overflow: 'hidden',
                    }}
                  >
                    <div
                      style={{
                        width: `${barWidth}%`,
                        height: '100%',
                        background:
                          feat.driftStrength > 0.6 ? '#f43f5e' : feat.driftStrength > 0.3 ? '#f59e0b' : '#34d399',
                        borderRadius: '3px',
                      }}
                    />
                  </div>
                  <span
                    style={{
                      fontSize: '0.72rem',
                      fontWeight: 700,
                      color: feat.driftStrength > 0.6 ? '#fda4af' : '#94a3b8',
                      minWidth: '40px',
                      textAlign: 'right',
                    }}
                  >
                    {(feat.driftStrength * 100).toFixed(0)}%
                  </span>
                  <span
                    style={{
                      fontSize: '0.65rem',
                      fontWeight: 600,
                      padding: '0.12rem 0.4rem',
                      borderRadius: '3px',
                      background: active ? 'rgba(56, 189, 248, 0.2)' : 'rgba(244, 63, 94, 0.2)',
                      color: active ? '#38bdf8' : '#fda4af',
                    }}
                  >
                    {active ? 'IN MODEL' : 'DROPPED'}
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Explanatory callout */}
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
        <span style={{ fontWeight: 600, color: '#f8fafc' }}>Core Principle: </span>
        Adversarial validation transforms an intractable multivariate distribution test into a standard binary
        supervised learning task. Try clicking on high-drift features like <code>transaction_timestamp</code> or{' '}
        <code>datacenter_region_code</code> to drop them, and watch the adversarial AUC plummet back down toward 0.50!
      </div>
    </div>
  )
}
