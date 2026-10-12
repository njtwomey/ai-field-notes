import { useMemo } from 'react'
import { choice, Figure, Readout, slider, useFigureState } from 'aifn-render'

const DOMAIN_PRESETS = {
  clinical: {
    label: 'clinical trial biostatistics',
    accuracy: 98,
    completeness: 95,
    consistency: 96,
    validity: 98,
    uniqueness: 99,
    timeliness: 50,
  },
  fraud: {
    label: 'real-time fraud scoring',
    accuracy: 92,
    completeness: 72,
    consistency: 85,
    validity: 90,
    uniqueness: 90,
    timeliness: 98,
  },
  llm_pretraining: {
    label: 'web corpus LLM pre-training',
    accuracy: 82,
    completeness: 60,
    consistency: 70,
    validity: 78,
    uniqueness: 96,
    timeliness: 65,
  },
  accounting: {
    label: 'financial ledger auditing',
    accuracy: 99,
    completeness: 99,
    consistency: 100,
    validity: 99,
    uniqueness: 100,
    timeliness: 80,
  },
} as const

type PresetKey = keyof typeof DOMAIN_PRESETS

export function DataQualityDimensionsRadar() {
  const state = useFigureState({
    preset: choice<PresetKey>(
      [
        { value: 'clinical', label: 'clinical trial biostatistics' },
        { value: 'fraud', label: 'real-time fraud scoring' },
        { value: 'llm_pretraining', label: 'web corpus LLM pre-training' },
        { value: 'accounting', label: 'financial ledger auditing' },
      ],
      'clinical',
      { label: 'workload profile' },
    ),
    cleaningEffort: slider(0, 100, 60, { step: 5, label: 'allocated cleaning budget' }),
  })

  const results = useMemo(() => {
    const p = DOMAIN_PRESETS[state.preset]
    const boost = (state.cleaningEffort - 50) * 0.15

    const dims = [
      {
        name: 'Accuracy',
        score: Math.min(100, Math.max(20, p.accuracy + boost * 0.8)),
        desc: 'Fidelity to ground truth reality',
      },
      {
        name: 'Completeness',
        score: Math.min(100, Math.max(20, p.completeness + boost * 1.2)),
        desc: 'Proportion of non-missing values',
      },
      {
        name: 'Consistency',
        score: Math.min(100, Math.max(20, p.consistency + boost * 1.0)),
        desc: 'Absence of logical contradiction',
      },
      {
        name: 'Validity',
        score: Math.min(100, Math.max(20, p.validity + boost * 0.9)),
        desc: 'Adherence to syntax and domain bounds',
      },
      {
        name: 'Uniqueness',
        score: Math.min(100, Math.max(20, p.uniqueness + boost * 1.1)),
        desc: 'Absence of duplicate entity records',
      },
      {
        name: 'Timeliness',
        score: Math.min(100, Math.max(20, p.timeliness + boost * 0.3)),
        desc: 'Freshness at inference time',
      },
    ]

    const avgScore = dims.reduce((a, b) => a + b.score, 0) / dims.length
    const sorted = [...dims].sort((a, b) => a.score - b.score)
    const bottleneck = sorted[0]

    return {
      dims,
      avgScore,
      bottleneck,
    }
  }, [state.preset, state.cleaningEffort])

  return (
    <Figure
      title="The six dimensions of data quality across machine learning workloads"
      state={state}
      hoverReadout={false}
      caption="Evaluating the six fundamental dimensions of data quality across workload contexts. In clinical trials, completeness and accuracy are non-negotiable. In real-time fraud scoring, timeliness (sub-second latency) supersedes completeness. In web-scale LLM pre-training, uniqueness (aggressive near-deduplication) is paramount. Data cleaning is an inductive modeling decision calibrated to specific workload failure modes."
      readouts={
        <>
          <Readout label="composite quality" value={`${results.avgScore.toFixed(1)} / 100`} />
          <Readout
            label="workload bottleneck"
            value={`${results.bottleneck.name} (${results.bottleneck.score.toFixed(0)}%)`}
          />
          <Readout label="critical dimension" value={results.bottleneck.desc} />
        </>
      }
    >
      <div style={{ padding: '14px 16px', fontSize: '13px', lineHeight: 1.5 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
          {results.dims.map((dim) => {
            const isLowest = dim.name === results.bottleneck.name
            const pct = dim.score.toFixed(0)
            return (
              <div
                key={dim.name}
                style={{
                  border: isLowest ? '1px solid rgba(239,68,68,0.5)' : '1px solid rgba(128,128,128,0.2)',
                  borderRadius: '6px',
                  padding: '10px 12px',
                  backgroundColor: isLowest ? 'rgba(239,68,68,0.06)' : 'rgba(128,128,128,0.04)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <span style={{ fontWeight: 600 }}>{dim.name}</span>
                  <span style={{ fontWeight: 600, color: isLowest ? '#ef4444' : '#16a34a' }}>{pct}%</span>
                </div>
                <div
                  style={{
                    height: '6px',
                    width: '100%',
                    backgroundColor: 'rgba(128,128,128,0.2)',
                    borderRadius: '3px',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      height: '100%',
                      width: `${pct}%`,
                      backgroundColor: isLowest ? '#ef4444' : '#16a34a',
                    }}
                  />
                </div>
                <div style={{ marginTop: '6px', fontSize: '11px', color: 'rgba(128,128,128,0.85)' }}>{dim.desc}</div>
              </div>
            )
          })}
        </div>
      </div>
    </Figure>
  )
}
