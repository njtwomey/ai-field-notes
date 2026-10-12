import React, { useMemo, useState } from 'react'

type OodSuite = 'corruptions' | 'wilds' | 'adversarial'

interface ModelRobustnessData {
  modelName: string
  family: string
  idAccuracy: number
  // Dynamic score based on severity/scenario
  computeOod: (severity: number) => number
  effectiveRobustness: 'baseline' | 'robust' | 'brittle'
}

const MODELS: ModelRobustnessData[] = [
  {
    modelName: 'ResNet-50 (Standard ERM)',
    family: 'CNN (Supervised)',
    idAccuracy: 76.5,
    computeOod: (s) => Math.max(12, 76.5 - s * 11.5),
    effectiveRobustness: 'baseline',
  },
  {
    modelName: 'DenseNet-121 (Standard ERM)',
    family: 'CNN (Supervised)',
    idAccuracy: 74.2,
    computeOod: (s) => Math.max(10, 74.2 - s * 12.2),
    effectiveRobustness: 'brittle',
  },
  {
    modelName: 'ViT-B/16 (Vision Transformer)',
    family: 'Transformer',
    idAccuracy: 81.8,
    computeOod: (s) => Math.max(28, 81.8 - s * 8.5),
    effectiveRobustness: 'robust',
  },
  {
    modelName: 'CLIP-ViT-L (Multimodal Pretrained)',
    family: 'Foundation Model',
    idAccuracy: 85.2,
    computeOod: (s) => Math.max(45, 85.2 - s * 5.8),
    effectiveRobustness: 'robust',
  },
]

const WILDS_BENCHMARKS = [
  {
    name: 'Camelyon17 (Hospital Shift)',
    domain: 'Histopathology',
    shift: 'Cross-institution transfer across 5 hospital scanner centers',
    idAcc: '94.2%',
    oodAcc: '68.5%',
    gap: '-25.7%',
  },
  {
    name: 'FMoW (Global Satellite Shift)',
    domain: 'Geospatial Remote Sensing',
    shift: 'Temporal shift across years (2002–2018) and continental regions',
    idAcc: '74.8%',
    oodAcc: '58.3%',
    gap: '-16.5%',
  },
  {
    name: 'CivilComments (Demographic Shift)',
    domain: 'Toxicity Detection',
    shift: 'Worst-group accuracy across protected demographic identities',
    idAcc: '92.1% (Avg)',
    oodAcc: '66.4% (Worst)',
    gap: '-25.7%',
  },
]

export default function OodEvaluationExplorer(): React.ReactElement {
  const [suite, setSuite] = useState<OodSuite>('corruptions')
  const [severityLevel, setSeverityLevel] = useState<number>(3) // 1 to 5

  // Compute current model scores for corruptions suite
  const scoredModels = useMemo(() => {
    return MODELS.map((m) => {
      const oodAcc = m.computeOod(severityLevel)
      const drop = m.idAccuracy - oodAcc
      return {
        ...m,
        oodAccuracy: oodAcc,
        drop,
      }
    })
  }, [severityLevel])

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
            Out-of-Distribution (OOD) Evaluation Benchmarking
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
            {suite === 'corruptions'
              ? `ImageNet-C Severity ${severityLevel}/5`
              : suite === 'wilds'
                ? 'WILDS Real-World Shifts'
                : 'Natural Adversarial Examples'}
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
          In-distribution validation scores measure expected accuracy under static environments. OOD evaluation suites
          stress-test generalization under physical noise, domain transfer, and spurious shortcuts.
        </p>
      </div>

      {/* Benchmark Suite Tabs */}
      <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '1.25rem', flexWrap: 'wrap' }}>
        {(
          [
            ['corruptions', 'ImageNet-C (Synthetic Perturbations)'],
            ['wilds', 'WILDS (Real-World Institutional Shift)'],
            ['adversarial', 'ImageNet-A (Spurious Cue Challenges)'],
          ] as const
        ).map(([sKey, label]) => (
          <button
            key={sKey}
            type="button"
            onClick={() => setSuite(sKey)}
            style={{
              flex: 1,
              minWidth: '170px',
              padding: '0.4rem 0.6rem',
              fontSize: '0.75rem',
              borderRadius: '5px',
              border: suite === sKey ? '1px solid #38bdf8' : '1px solid #444',
              background: suite === sKey ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
              color: suite === sKey ? '#38bdf8' : '#aaa',
              cursor: 'pointer',
              fontWeight: suite === sKey ? 600 : 400,
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Main Suite Content */}
      {suite === 'corruptions' && (
        <>
          {/* Severity Slider */}
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
            <label style={{ fontSize: '0.76rem', fontWeight: 600, color: '#f8fafc', minWidth: '220px' }}>
              Corruption Severity Level: <strong style={{ color: '#f59e0b' }}>Level {severityLevel} of 5</strong>
            </label>
            <input
              type="range"
              min={1}
              max={5}
              step={1}
              value={severityLevel}
              onChange={(e) => setSeverityLevel(parseInt(e.target.value, 10))}
              style={{ flex: 1, cursor: 'pointer', accentColor: '#f59e0b' }}
            />
            <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>
              {severityLevel === 1
                ? 'Mild Gaussian blur'
                : severityLevel === 3
                  ? 'Medium snow/fog/noise'
                  : 'Extreme severe corruption'}
            </span>
          </div>

          {/* Comparative Model Performance Table */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.55rem', marginBottom: '1.25rem' }}>
            {scoredModels.map((m) => {
              const oodPct = Math.min(100, Math.round(m.oodAccuracy))
              const idPct = Math.min(100, Math.round(m.idAccuracy))

              return (
                <div
                  key={m.modelName}
                  style={{
                    padding: '0.65rem 0.85rem',
                    borderRadius: '6px',
                    background: 'rgba(255, 255, 255, 0.03)',
                    border: '1px solid rgba(255, 255, 255, 0.06)',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      marginBottom: '0.35rem',
                    }}
                  >
                    <div>
                      <strong style={{ fontSize: '0.82rem', color: '#f8fafc' }}>{m.modelName}</strong>
                      <span style={{ fontSize: '0.7rem', color: '#94a3b8', marginLeft: '0.5rem' }}>({m.family})</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                      <span style={{ fontSize: '0.74rem', color: '#94a3b8' }}>ID: {m.idAccuracy.toFixed(1)}%</span>
                      <strong
                        style={{
                          fontSize: '0.84rem',
                          color: m.oodAccuracy > 60 ? '#34d399' : m.oodAccuracy > 40 ? '#f59e0b' : '#f43f5e',
                        }}
                      >
                        OOD: {m.oodAccuracy.toFixed(1)}% (-{m.drop.toFixed(1)}%)
                      </strong>
                      <span
                        style={{
                          fontSize: '0.65rem',
                          fontWeight: 600,
                          padding: '0.1rem 0.35rem',
                          borderRadius: '3px',
                          background:
                            m.effectiveRobustness === 'robust'
                              ? 'rgba(52, 211, 153, 0.2)'
                              : 'rgba(255, 255, 255, 0.08)',
                          color: m.effectiveRobustness === 'robust' ? '#6ee7b7' : '#94a3b8',
                        }}
                      >
                        {m.effectiveRobustness === 'robust' ? 'ROBUST' : 'STANDARD ERM'}
                      </span>
                    </div>
                  </div>

                  {/* Dual comparison bar: ID vs OOD */}
                  <div
                    style={{
                      position: 'relative',
                      width: '100%',
                      height: '8px',
                      background: 'rgba(0,0,0,0.3)',
                      borderRadius: '4px',
                      overflow: 'hidden',
                    }}
                  >
                    {/* Ghost bar for ID */}
                    <div
                      style={{
                        position: 'absolute',
                        width: `${idPct}%`,
                        height: '100%',
                        background: 'rgba(56, 189, 248, 0.25)',
                        borderRadius: '4px',
                      }}
                    />
                    {/* Active OOD bar */}
                    <div
                      style={{
                        position: 'absolute',
                        width: `${oodPct}%`,
                        height: '100%',
                        background: m.effectiveRobustness === 'robust' ? '#34d399' : '#f43f5e',
                        borderRadius: '4px',
                        transition: 'width 0.2s ease',
                      }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </>
      )}

      {suite === 'wilds' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem', marginBottom: '1.25rem' }}>
          {WILDS_BENCHMARKS.map((b) => (
            <div
              key={b.name}
              style={{
                padding: '0.85rem',
                borderRadius: '7px',
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid rgba(255, 255, 255, 0.06)',
                fontSize: '0.78rem',
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
                <strong style={{ color: '#38bdf8', fontSize: '0.85rem' }}>{b.name}</strong>
                <span style={{ fontSize: '0.7rem', color: '#64748b' }}>{b.domain}</span>
              </div>
              <p style={{ margin: '0 0 0.5rem', color: '#94a3b8', fontSize: '0.74rem' }}>{b.shift}</p>
              <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                <span style={{ color: '#cbd5e1' }}>
                  In-Distribution Val: <strong style={{ color: '#fff' }}>{b.idAcc}</strong>
                </span>
                <span style={{ color: '#cbd5e1' }}>
                  Out-of-Distribution Test: <strong style={{ color: '#f43f5e' }}>{b.oodAcc}</strong>
                </span>
                <span style={{ color: '#f43f5e', fontWeight: 600 }}>OOD Performance Gap: {b.gap}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {suite === 'adversarial' && (
        <div
          style={{
            padding: '1rem',
            borderRadius: '8px',
            background: 'rgba(0, 0, 0, 0.3)',
            border: '1px solid rgba(255, 255, 255, 0.06)',
            fontSize: '0.78rem',
            lineHeight: 1.5,
            color: '#cbd5e1',
            marginBottom: '1.25rem',
          }}
        >
          <strong style={{ color: '#f8fafc', fontSize: '0.85rem', display: 'block', marginBottom: '0.4rem' }}>
            ImageNet-A Natural Adversarial Examples (Hendrycks et al. 2021)
          </strong>
          Standard ImageNet test sets suffer from spurious background correlations (e.g. monkeys always in jungle trees,
          whales always in blue oceans).
          <div
            style={{
              margin: '0.6rem 0',
              padding: '0.65rem',
              borderRadius: '5px',
              background: 'rgba(244, 63, 94, 0.1)',
              border: '1px solid rgba(244, 63, 94, 0.25)',
              color: '#fda4af',
            }}
          >
            On 7,500 unmodified, natural photographic examples with unexpected backgrounds (e.g. a dragonfly resting on
            a wooden chair), a standard ResNet-50 top-1 accuracy crashes from <strong>76.5% down to 2.2%</strong>!
          </div>
          Natural adversarial challenge sets prove that high ID test accuracy is easily achieved by exploiting
          non-causal background textures rather than learning the object itself.
        </div>
      )}

      {/* Effective Robustness Callout */}
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
        <span style={{ fontWeight: 600, color: '#f8fafc' }}>Concept of Effective Robustness (Taori et al. 2020): </span>
        Most model improvements only show "relative robustness" (a model with higher ID accuracy has slightly higher OOD
        accuracy along a fixed linear slope). True <em>effective robustness</em> occurs when an architectural family
        (such as Vision Transformers or foundation models with massive diverse pretraining) achieves OOD accuracy
        strictly above the standard empirical trendline!
      </div>
    </div>
  )
}
