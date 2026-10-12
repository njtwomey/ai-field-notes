import React, { useState } from 'react'

type CaseStudyId = 'patient' | 'scanner' | 'code' | 'kg' | 'timeseries'

interface CaseStudy {
  id: CaseStudyId
  title: string
  domain: string
  leakageType: string
  reportedScore: string
  externalScore: string
  rootCause: string
  shortcutLearned: string
  fix: string
  diagnosticCheck: string
}

const CASE_STUDIES: Record<CaseStudyId, CaseStudy> = {
  patient: {
    id: 'patient',
    title: 'Patient Overlap in COVID-19 Radiographs',
    domain: 'Clinical Radiology',
    leakageType: 'Entity / Group Leakage',
    reportedScore: '99.2% Accuracy (Random Split)',
    externalScore: '61.4% Accuracy (External Hospital)',
    rootCause:
      'Multiple chest X-rays taken from the same patient across longitudinal treatment visits were partitioned randomly between train and test sets.',
    shortcutLearned:
      'The deep CNN memorized patient-specific anatomical features (e.g. rib spacing, cardiac silhouette, pre-existing chronic implants) instead of acute COVID pulmonary opacities.',
    fix: 'Strict GroupKFold partitioning on hashed patient UUID. Ensure all images for any single human stay atomically in one split.',
    diagnosticCheck:
      'Run a Group Identifier Predictor: if an auxiliary model can predict which patient a scan belongs to with high accuracy, patient identity is leaking.',
  },
  scanner: {
    id: 'scanner',
    title: 'Hospital Scanner & Marker Confounding (Zech et al. 2018)',
    domain: 'Medical Imaging',
    leakageType: 'Site / Instrument Confounding',
    reportedScore: '0.93 AUC (Pooled In-Sample CV)',
    externalScore: '0.65 AUC (External Hospital Test)',
    rootCause:
      'Hospitals used different radiopaque text stickers (e.g., "PORTABLE" or department initials) burned into images. Bedside portable scanners were used for bedridden ICU patients who had high pneumonia incidence.',
    shortcutLearned:
      'The model detected the visual font of the portable X-ray machine sticker in the image margin, predicting severe pneumonia whenever the portable machine font was present.',
    fix: 'Leave-One-Site-Out (LOGO) cross-validation across hospital institutions. Crop and mask non-anatomical image margins.',
    diagnosticCheck:
      'Adversarial validation across hospital sites: train a model to predict hospital origin from images. High AUC indicates heavy site-specific confounding.',
  },
  code: {
    id: 'code',
    title: 'Code Clones & Fork Duplication (Allamanis 2019)',
    domain: 'Software Engineering / LLMs',
    leakageType: 'Near-Duplicate Benchmark Contamination',
    reportedScore: '84.5% Exact Match (Raw GitHub Scrape)',
    externalScore: '47.2% Exact Match (Deduplicated)',
    rootCause:
      'Popular utility functions, configuration files, and forked repositories were duplicated thousands of times across public GitHub repositories.',
    shortcutLearned:
      'The neural code model memorized common verbatim function bodies rather than learning general program synthesis and semantic logic.',
    fix: 'Syntactic near-duplicate deduplication using Abstract Syntax Tree (AST) token MinHash and Locality-Sensitive Hashing (LSH) prior to splitting.',
    diagnosticCheck:
      'Compute n-gram syntax shingle overlap between training and test sets. High overlap (> 5%) indicates code clone leakage.',
  },
  kg: {
    id: 'kg',
    title: 'Inverse Relations in Knowledge Graph Embeddings (WN18 / FB15k)',
    domain: 'Graph Machine Learning',
    leakageType: 'Symmetric Relation Shortcut',
    reportedScore: '94.0% Hits@10 on WN18',
    externalScore: '53.0% Hits@10 on WN18RR (Fixed)',
    rootCause:
      'For training triple (h, hypernym, t), the test set contained the exact inverse triple (t, hyponym, h). The two relations are exact semantic opposites.',
    shortcutLearned:
      'The embedding model did not learn generalized relation composition; it merely memorized the rule: if (h, r, t) exists in train, predict (t, r_inv, h) in test.',
    fix: 'Filtered benchmarks (WN18RR and FB15k-237) that explicitly detect and prune all reciprocal and inverse relation pairs from evaluation sets.',
    diagnosticCheck:
      'Graph isomorphism checks: test whether flipping relation edges produces a duplicate triple in the opposing partition.',
  },
  timeseries: {
    id: 'timeseries',
    title: 'Random Splitting of Autocorrelated Time Series',
    domain: 'Financial Market Forecasting',
    leakageType: 'Temporal Lookahead / Interpolation',
    reportedScore: '98.5% Directional Accuracy (Random CV)',
    externalScore: '50.2% Accuracy (Live Trading)',
    rootCause: 'Asset prices and momentum indicators were split via uniform random sampling across time rows.',
    shortcutLearned:
      'High autocorrelation between time t and time t+1 meant the model was interpolating between known bookends rather than forecasting the forward trajectory.',
    fix: 'Walk-forward rolling origin temporal splits with forward label horizon gaps, purging, and post-test embargo buffers.',
    diagnosticCheck:
      'Lagged Cross-Correlation: check if test errors spike immediately when adjacent chronological neighbors are withheld.',
  },
}

export default function LeakageTaxonomyExplorer(): React.ReactElement {
  const [selectedCaseId, setSelectedCaseId] = useState<CaseStudyId>('patient')
  const activeCase = CASE_STUDIES[selectedCaseId]

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
            Empirical Leakage Case Studies Explorer
          </h4>
          <span
            style={{
              fontSize: '0.74rem',
              padding: '0.2rem 0.55rem',
              borderRadius: '999px',
              background: 'rgba(244, 63, 94, 0.15)',
              color: '#fda4af',
              fontWeight: 600,
            }}
          >
            {activeCase.leakageType}
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
          Investigate how subtle data partitioning errors produced legendary "too good to be true" results that
          collapsed upon real-world deployment.
        </p>
      </div>

      {/* Case Studies Selector Tabs */}
      <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '1.25rem', flexWrap: 'wrap' }}>
        {(
          [
            ['patient', '1. Patient Overlap'],
            ['scanner', '2. Scanner / Site Artifacts'],
            ['code', '3. Duplicated Code Clones'],
            ['kg', '4. KG Inverse Relations'],
            ['timeseries', '5. Time-Series Shuffling'],
          ] as const
        ).map(([cId, label]) => (
          <button
            key={cId}
            type="button"
            onClick={() => setSelectedCaseId(cId)}
            style={{
              flex: 1,
              minWidth: '150px',
              padding: '0.4rem 0.6rem',
              fontSize: '0.75rem',
              borderRadius: '5px',
              border: selectedCaseId === cId ? '1px solid #38bdf8' : '1px solid #444',
              background: selectedCaseId === cId ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
              color: selectedCaseId === cId ? '#38bdf8' : '#aaa',
              cursor: 'pointer',
              fontWeight: selectedCaseId === cId ? 600 : 400,
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Case Details Card */}
      <div
        style={{
          padding: '1rem',
          borderRadius: '8px',
          background: 'rgba(0, 0, 0, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.06)',
          marginBottom: '1.25rem',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
          <strong style={{ fontSize: '0.95rem', color: '#f8fafc' }}>{activeCase.title}</strong>
          <span style={{ fontSize: '0.72rem', color: '#64748b' }}>Domain: {activeCase.domain}</span>
        </div>

        {/* The Performance Cliff Banner */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '0.75rem',
            margin: '0.75rem 0',
          }}
        >
          <div
            style={{
              padding: '0.65rem 0.8rem',
              borderRadius: '6px',
              background: 'rgba(244, 63, 94, 0.1)',
              border: '1px solid rgba(244, 63, 94, 0.25)',
            }}
          >
            <span style={{ display: 'block', fontSize: '0.66rem', color: '#fda4af', textTransform: 'uppercase' }}>
              Reported Leaked Metric
            </span>
            <strong style={{ fontSize: '1.15rem', color: '#fda4af' }}>{activeCase.reportedScore}</strong>
            <span style={{ display: 'block', fontSize: '0.68rem', color: '#94a3b8', marginTop: '0.15rem' }}>
              Apparent near-flawless performance
            </span>
          </div>

          <div
            style={{
              padding: '0.65rem 0.8rem',
              borderRadius: '6px',
              background: 'rgba(52, 211, 153, 0.1)',
              border: '1px solid rgba(52, 211, 153, 0.25)',
            }}
          >
            <span style={{ display: 'block', fontSize: '0.66rem', color: '#6ee7b7', textTransform: 'uppercase' }}>
              True Out-of-Distribution Metric
            </span>
            <strong style={{ fontSize: '1.15rem', color: '#6ee7b7' }}>{activeCase.externalScore}</strong>
            <span style={{ display: 'block', fontSize: '0.68rem', color: '#94a3b8', marginTop: '0.15rem' }}>
              Catastrophic drop upon live rollout
            </span>
          </div>
        </div>

        {/* Root Cause & Shortcut Mechanism */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
            gap: '0.85rem',
            fontSize: '0.78rem',
            lineHeight: 1.45,
            marginTop: '0.85rem',
          }}
        >
          <div
            style={{
              padding: '0.7rem',
              borderRadius: '6px',
              background: 'rgba(255,255,255,0.02)',
              border: '1px solid rgba(255,255,255,0.05)',
            }}
          >
            <strong style={{ display: 'block', color: '#f59e0b', marginBottom: '0.25rem' }}>What Leaked:</strong>
            <span style={{ color: '#cbd5e1' }}>{activeCase.rootCause}</span>
          </div>

          <div
            style={{
              padding: '0.7rem',
              borderRadius: '6px',
              background: 'rgba(255,255,255,0.02)',
              border: '1px solid rgba(255,255,255,0.05)',
            }}
          >
            <strong style={{ display: 'block', color: '#f43f5e', marginBottom: '0.25rem' }}>
              The Opportunistic Shortcut:
            </strong>
            <span style={{ color: '#cbd5e1' }}>{activeCase.shortcutLearned}</span>
          </div>
        </div>
      </div>

      {/* Fix & Diagnostic Check */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
          gap: '0.75rem',
          fontSize: '0.78rem',
        }}
      >
        <div
          style={{
            padding: '0.75rem',
            borderRadius: '6px',
            background: 'rgba(52, 211, 153, 0.05)',
            border: '1px solid rgba(52, 211, 153, 0.15)',
          }}
        >
          <strong style={{ display: 'block', color: '#34d399', marginBottom: '0.2rem' }}>
            Architectural Solution:
          </strong>
          <span style={{ color: '#e2e8f0', lineHeight: 1.45 }}>{activeCase.fix}</span>
        </div>

        <div
          style={{
            padding: '0.75rem',
            borderRadius: '6px',
            background: 'rgba(56, 189, 248, 0.05)',
            border: '1px solid rgba(56, 189, 248, 0.15)',
          }}
        >
          <strong style={{ display: 'block', color: '#38bdf8', marginBottom: '0.2rem' }}>
            How to Detect It (Audit Test):
          </strong>
          <span style={{ color: '#e2e8f0', lineHeight: 1.45 }}>{activeCase.diagnosticCheck}</span>
        </div>
      </div>
    </div>
  )
}
