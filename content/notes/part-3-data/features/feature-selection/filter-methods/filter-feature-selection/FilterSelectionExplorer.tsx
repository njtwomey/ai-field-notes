import { useMemo, useState } from 'react'

interface FeatureProfile {
  id: string
  name: string
  nature: string
  pearsonR: number
  fStat: number
  mutualInfo: number
  redundancyWithX1: number
  mrmrScore: number
  description: string
}

const CANDIDATES: FeatureProfile[] = [
  {
    id: 'x1',
    name: 'X₁ (Linear Signal)',
    nature: 'Strong linear relation with target y',
    pearsonR: 0.84,
    fStat: 142.5,
    mutualInfo: 0.72,
    redundancyWithX1: 1.0,
    mrmrScore: 0.72,
    description: 'High linear correlation, high F-statistic, high mutual information.',
  },
  {
    id: 'x2',
    name: 'X₂ (Non-linear Parabolic)',
    nature: 'Quadratic relation y = (X₂)² + ε',
    pearsonR: 0.04,
    fStat: 0.3,
    mutualInfo: 0.64,
    redundancyWithX1: 0.05,
    mrmrScore: 0.61,
    description: 'Zero linear correlation (r ≈ 0), but high mutual information. Missed by Pearson/ANOVA, caught by MI.',
  },
  {
    id: 'x3',
    name: 'X₃ (Redundant Clone of X₁)',
    nature: 'Collinear duplicate X₃ = X₁ + 0.05 ε',
    pearsonR: 0.82,
    fStat: 135.1,
    mutualInfo: 0.69,
    redundancyWithX1: 0.98,
    mrmrScore: -0.29,
    description:
      'High univariate score, but almost 100% redundant with X₁. Univariate filters select both; mRMR rejects it.',
  },
  {
    id: 'x4',
    name: 'X₄ (XOR Interaction)',
    nature: 'XOR partner with X₅: y = X₄ ⊕ X₅',
    pearsonR: 0.02,
    fStat: 0.1,
    mutualInfo: 0.03,
    redundancyWithX1: 0.02,
    mrmrScore: 0.02,
    description: 'Independent fair bit. Marginal univariate scores are zero; informative only in combination with X₅.',
  },
  {
    id: 'x5',
    name: 'X₅ (Pure Noise)',
    nature: 'Independent Gaussian noise N(0, 1)',
    pearsonR: 0.05,
    fStat: 0.5,
    mutualInfo: 0.04,
    redundancyWithX1: 0.03,
    mrmrScore: 0.02,
    description: 'Spurious random noise. Baseline uninformative column.',
  },
]

type FilterMetric = 'pearson' | 'fstat' | 'mi' | 'mrmr'

export function FilterSelectionExplorer() {
  const [metric, setMetric] = useState<FilterMetric>('pearson')
  const [topK, setTopK] = useState<number>(3)

  const rankedFeatures = useMemo(() => {
    const list = [...CANDIDATES]
    list.sort((a, b) => {
      if (metric === 'pearson') return b.pearsonR - a.pearsonR
      if (metric === 'fstat') return b.fStat - a.fStat
      if (metric === 'mi') return b.mutualInfo - a.mutualInfo
      return b.mrmrScore - a.mrmrScore
    })
    return list
  }, [metric])

  const selectedSet = useMemo(() => {
    return new Set(rankedFeatures.slice(0, topK).map((f) => f.id))
  }, [rankedFeatures, topK])

  return (
    <div className="my-6 rounded-lg border border-border bg-card p-4 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
        <div>
          <h4 className="text-text font-semibold">Univariate Filter Methods & mRMR Explorer</h4>
          <p className="text-text-muted text-xs">
            Compare Pearson correlation, ANOVA F-statistic, Mutual Information, and Minimum Redundancy Maximum
            Relevance.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-text-muted text-xs font-semibold">Top-k Features:</label>
          <div className="flex items-center gap-1 font-mono text-xs">
            <input
              type="range"
              min={1}
              max={5}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-20 accent-primary"
            />
            <span className="font-bold text-primary">{topK}</span>
          </div>
        </div>
      </div>

      {/* Metric Selector Tabs */}
      <div className="mb-4 flex flex-wrap gap-2 text-xs">
        <button
          onClick={() => setMetric('pearson')}
          className={`rounded px-3 py-1 font-medium transition-colors ${
            metric === 'pearson'
              ? 'bg-primary text-primary-foreground'
              : 'bg-surface hover:bg-surface-elevated text-text'
          }`}
        >
          Pearson |r| (Linear)
        </button>
        <button
          onClick={() => setMetric('fstat')}
          className={`rounded px-3 py-1 font-medium transition-colors ${
            metric === 'fstat' ? 'bg-primary text-primary-foreground' : 'bg-surface hover:bg-surface-elevated text-text'
          }`}
        >
          ANOVA F-Statistic
        </button>
        <button
          onClick={() => setMetric('mi')}
          className={`rounded px-3 py-1 font-medium transition-colors ${
            metric === 'mi' ? 'bg-primary text-primary-foreground' : 'bg-surface hover:bg-surface-elevated text-text'
          }`}
        >
          Mutual Information I(X; Y)
        </button>
        <button
          onClick={() => setMetric('mrmr')}
          className={`rounded px-3 py-1 font-medium transition-colors ${
            metric === 'mrmr' ? 'bg-primary text-primary-foreground' : 'bg-surface hover:bg-surface-elevated text-text'
          }`}
        >
          mRMR (Relevance − Redundancy)
        </button>
      </div>

      {/* Feature Rankings Table & Bar Visualiser */}
      <div className="bg-surface space-y-2 rounded border border-border p-3">
        {rankedFeatures.map((f, rankIdx) => {
          const isSelected = selectedSet.has(f.id)
          let scoreLabel = ''
          let normPct = 0

          if (metric === 'pearson') {
            scoreLabel = `|r| = ${f.pearsonR.toFixed(2)}`
            normPct = (f.pearsonR / 1.0) * 100
          } else if (metric === 'fstat') {
            scoreLabel = `F = ${f.fStat.toFixed(1)}`
            normPct = (f.fStat / 150.0) * 100
          } else if (metric === 'mi') {
            scoreLabel = `I(X; Y) = ${f.mutualInfo.toFixed(2)} nats`
            normPct = (f.mutualInfo / 0.8) * 100
          } else {
            scoreLabel = `mRMR = ${f.mrmrScore.toFixed(2)}`
            normPct = Math.max(0, (f.mrmrScore / 0.8) * 100)
          }

          return (
            <div
              key={f.id}
              className={`rounded border p-2.5 transition-colors ${
                isSelected
                  ? 'border-primary/50 bg-primary/5 shadow-sm'
                  : 'bg-surface-elevated/40 border-border/70 opacity-75'
              }`}
            >
              <div className="mb-1 flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <span className="text-text font-mono font-bold">#{rankIdx + 1}</span>
                  <span className="text-text font-semibold">{f.name}</span>
                  {isSelected && (
                    <span className="py-0.2 rounded bg-emerald-500/20 px-1.5 text-[10px] font-bold text-emerald-800 dark:text-emerald-300">
                      SELECTED
                    </span>
                  )}
                </div>
                <span className="font-mono text-xs font-semibold text-primary">{scoreLabel}</span>
              </div>

              {/* Progress bar */}
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-border/60">
                <div
                  className={`h-full transition-all duration-300 ${isSelected ? 'bg-primary' : 'bg-text-muted/40'}`}
                  style={{ width: `${Math.min(100, Math.max(3, normPct))}%` }}
                />
              </div>

              <p className="text-text-muted mt-1.5 text-[11px] leading-tight">{f.description}</p>
            </div>
          )
        })}
      </div>

      {/* Pathology / Insight Callout */}
      <div className="bg-surface/60 text-text-muted mt-4 rounded p-2.5 text-xs">
        {metric === 'pearson' && (
          <p>
            <strong className="text-text">Pathology of Linear Filters:</strong> Non-linear parabolic feature{' '}
            <code className="text-text">X₂</code> has near-zero correlation (<code className="text-text">r = 0.04</code>
            ), causing Pearson filtering to discard it while retaining redundant clone{' '}
            <code className="text-text">X₃</code>.
          </p>
        )}
        {metric === 'fstat' && (
          <p>
            <strong className="text-text">ANOVA Limitation:</strong> Like correlation, ANOVA F-tests measure linear
            variance explanation across groups and completely fail to detect parity or non-linear symmetries.
          </p>
        )}
        {metric === 'mi' && (
          <p>
            <strong className="text-text">Mutual Information Advantage:</strong> Successfully identifies non-linear{' '}
            <code className="text-text">X₂</code> (<code className="text-text">0.64 nats</code>). However, univariate MI
            still selects both <code className="text-text">X₁</code> and redundant duplicate{' '}
            <code className="text-text">X₃</code>.
          </p>
        )}
        {metric === 'mrmr' && (
          <p>
            <strong className="text-text">mRMR Solution:</strong> Subtracting pairwise mutual information{' '}
            <code className="text-text">I(Xᵢ; Xⱼ)</code> penalises redundant duplicates, causing{' '}
            <code className="text-text">X₃</code> to drop to a negative score (<code className="text-text">-0.29</code>)
            and prioritizing orthogonal feature <code className="text-text">X₂</code>.
          </p>
        )}
      </div>
    </div>
  )
}
