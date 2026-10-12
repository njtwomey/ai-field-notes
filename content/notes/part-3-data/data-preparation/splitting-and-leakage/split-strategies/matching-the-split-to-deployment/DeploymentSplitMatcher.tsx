import React, { useState } from 'react'

type DeploymentDomain = 'clinical' | 'churn' | 'sensors'
type SplitStrategy = 'random' | 'aligned'

interface DomainCase {
  title: string
  deploymentReality: string
  unitOfNovelty: string
  flawedName: string
  flawedDescription: string
  flawedReportedScore: number
  flawedProdScore: number
  alignedName: string
  alignedDescription: string
  alignedReportedScore: number
  alignedProdScore: number
  mechanisms: string
}

const CASES: Record<DeploymentDomain, DomainCase> = {
  clinical: {
    title: 'Clinical Chest X-Ray Diagnosis',
    deploymentReality: 'Model will be deployed to new clinics to evaluate brand-new patients.',
    unitOfNovelty: 'New Patient ID & New Hospital Site',
    flawedName: 'Random Stratified Split',
    flawedDescription:
      'Images from the same patient and hospital land in both train and validation folds. The model learns hospital-specific metal markers and patient baseline anatomy.',
    flawedReportedScore: 0.94,
    flawedProdScore: 0.65,
    alignedName: 'Grouped Split (by Patient & Site)',
    alignedDescription:
      'All radiographs for any patient remain strictly in a single fold, and entire hospital sites are reserved for validation/test partitions.',
    alignedReportedScore: 0.67,
    alignedProdScore: 0.66,
    mechanisms:
      'Zech et al. (2018) discovered that chest X-ray models achieved high apparent AUC by identifying hospital-specific image acquisition artifacts (e.g. portable X-ray markers) rather than pathology.',
  },
  churn: {
    title: 'Subscription Churn Prediction',
    deploymentReality: 'Model will predict at month-end which current subscribers will cancel in the upcoming month.',
    unitOfNovelty: 'Future Time Window (t > T_train)',
    flawedName: 'Random Row Split across Time',
    flawedDescription:
      'Randomly partitions historical events across all dates. Predictions for March are trained on events from April and May.',
    flawedReportedScore: 0.92,
    flawedProdScore: 0.71,
    alignedName: 'Temporal Cut-off Split',
    alignedDescription:
      'Train on months 1–10; validate on month 11; test on month 12. Strict chronological ordering with zero future lookahead.',
    alignedReportedScore: 0.73,
    alignedProdScore: 0.72,
    mechanisms:
      'Random splitting allows future interaction trends and macro pricing shifts to bleed backwards into training rows, concealing temporal drift.',
  },
  sensors: {
    title: 'Industrial Machine Fleet Telemetry',
    deploymentReality: 'Model will monitor a fleet of 500 fixed factory turbines over their operational lifetimes.',
    unitOfNovelty: 'Future Time (Within-Entity)',
    flawedName: 'Group Split by Turbine ID',
    flawedDescription:
      'Splitting by turbine ID trains on turbines 1–400 and tests on turbines 401–500, failing to utilize historical calibration baselines for the specific machines being monitored.',
    flawedReportedScore: 0.72,
    flawedProdScore: 0.86,
    alignedName: 'Within-Entity Temporal Split',
    alignedDescription:
      'When the deployment task is explicitly personalized monitoring of known entities, each turbine is split chronologically: past days train, future days test.',
    alignedReportedScore: 0.87,
    alignedProdScore: 0.86,
    mechanisms:
      'Group splitting across entities is appropriate only when deployment encounters new entities. When deployment monitors known persistent units, within-entity temporal splitting is optimal.',
  },
}

export default function DeploymentSplitMatcher(): React.ReactElement {
  const [domainKey, setDomainKey] = useState<DeploymentDomain>('clinical')
  const [strategy, setStrategy] = useState<SplitStrategy>('random')

  const currentCase = CASES[domainKey]
  const isAligned = strategy === 'aligned'

  const reportedScore = isAligned ? currentCase.alignedReportedScore : currentCase.flawedReportedScore
  const prodScore = isAligned ? currentCase.alignedProdScore : currentCase.flawedProdScore
  const surpriseGap = reportedScore - prodScore

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
            Deployment-Aligned Data Splitting Simulator
          </h4>
          <span
            style={{
              fontSize: '0.74rem',
              padding: '0.2rem 0.55rem',
              borderRadius: '999px',
              background: isAligned ? 'rgba(52, 211, 153, 0.15)' : 'rgba(244, 63, 94, 0.15)',
              color: isAligned ? '#34d399' : '#f43f5e',
              fontWeight: 600,
            }}
          >
            {isAligned ? 'ALIGNED WITH DEPLOYMENT' : 'MISMATCHED SPLIT'}
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
          Deployment Reality: <strong>{currentCase.deploymentReality}</strong> &middot; Unit of Novelty:{' '}
          <strong style={{ color: '#38bdf8' }}>{currentCase.unitOfNovelty}</strong>
        </p>
      </div>

      {/* Domain Selection Tabs */}
      <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '1.1rem', flexWrap: 'wrap' }}>
        {(
          [
            ['clinical', 'Clinical / Medical AI'],
            ['churn', 'Temporal Churn Forecasting'],
            ['sensors', 'Personalized Fleet Monitoring'],
          ] as const
        ).map(([dKey, label]) => (
          <button
            key={dKey}
            type="button"
            onClick={() => setDomainKey(dKey)}
            style={{
              flex: 1,
              minWidth: '160px',
              padding: '0.4rem 0.6rem',
              fontSize: '0.76rem',
              borderRadius: '5px',
              border: domainKey === dKey ? '1px solid #38bdf8' : '1px solid #444',
              background: domainKey === dKey ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
              color: domainKey === dKey ? '#38bdf8' : '#aaa',
              cursor: 'pointer',
              fontWeight: domainKey === dKey ? 600 : 400,
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Strategy Toggle */}
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
            Candidate Splitting Protocol
          </label>
          <div style={{ display: 'flex', gap: '0.4rem' }}>
            <button
              type="button"
              onClick={() => setStrategy('random')}
              style={{
                flex: 1,
                padding: '0.35rem 0.6rem',
                fontSize: '0.76rem',
                borderRadius: '5px',
                border: strategy === 'random' ? '1px solid #f43f5e' : '1px solid #444',
                background: strategy === 'random' ? 'rgba(244, 63, 94, 0.2)' : 'transparent',
                color: strategy === 'random' ? '#f43f5e' : '#aaa',
                cursor: 'pointer',
                fontWeight: strategy === 'random' ? 600 : 400,
              }}
            >
              {currentCase.flawedName}
            </button>
            <button
              type="button"
              onClick={() => setStrategy('aligned')}
              style={{
                flex: 1,
                padding: '0.35rem 0.6rem',
                fontSize: '0.76rem',
                borderRadius: '5px',
                border: strategy === 'aligned' ? '1px solid #34d399' : '1px solid #444',
                background: strategy === 'aligned' ? 'rgba(52, 211, 153, 0.2)' : 'transparent',
                color: strategy === 'aligned' ? '#34d399' : '#aaa',
                cursor: 'pointer',
                fontWeight: strategy === 'aligned' ? 600 : 400,
              }}
            >
              {currentCase.alignedName}
            </button>
          </div>
        </div>

        <div style={{ fontSize: '0.78rem', color: '#cbd5e1', lineHeight: 1.45 }}>
          <strong style={{ color: isAligned ? '#34d399' : '#f43f5e', display: 'block', marginBottom: '0.2rem' }}>
            {isAligned ? currentCase.alignedName : currentCase.flawedName}
          </strong>
          {isAligned ? currentCase.alignedDescription : currentCase.flawedDescription}
        </div>
      </div>

      {/* Outcome Score Metrics Comparison */}
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
            Reported Validation Metric (AUC / R²)
          </span>
          <strong style={{ fontSize: '1.35rem', color: isAligned ? '#38bdf8' : '#f59e0b' }}>
            {reportedScore.toFixed(2)}
          </strong>
          <span style={{ display: 'block', fontSize: '0.7rem', color: '#64748b', marginTop: '0.2rem' }}>
            What the team reports internally
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
            True Production Deployment Metric
          </span>
          <strong style={{ fontSize: '1.35rem', color: '#34d399' }}>{prodScore.toFixed(2)}</strong>
          <span style={{ display: 'block', fontSize: '0.7rem', color: '#64748b', marginTop: '0.2rem' }}>
            What real users experience in deployment
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
            Deployment Discrepancy (Surprise Gap)
          </span>
          <strong style={{ fontSize: '1.35rem', color: surpriseGap > 0.1 ? '#f43f5e' : '#34d399' }}>
            {surpriseGap > 0 ? `+${(surpriseGap * 100).toFixed(0)}%` : `${(surpriseGap * 100).toFixed(0)}%`}
          </strong>
          <span style={{ display: 'block', fontSize: '0.7rem', color: '#64748b', marginTop: '0.2rem' }}>
            {surpriseGap > 0.1 ? 'Catastrophic Overconfidence!' : 'Reliably Aligned'}
          </span>
        </div>
      </div>

      {/* Case Study Takeaway */}
      <div
        style={{
          padding: '0.75rem 0.9rem',
          borderRadius: '7px',
          background: 'rgba(0, 0, 0, 0.25)',
          fontSize: '0.78rem',
          color: '#cbd5e1',
          lineHeight: 1.5,
        }}
      >
        <strong style={{ color: '#f8fafc' }}>Real-World Case Study: </strong> {currentCase.mechanisms}
      </div>
    </div>
  )
}
