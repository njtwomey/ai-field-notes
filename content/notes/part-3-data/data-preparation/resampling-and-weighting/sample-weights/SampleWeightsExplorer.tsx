import React, { useMemo, useState } from 'react'

type WeightingScheme = 'ipw' | 'covariate_shift' | 'recency' | 'survey'

interface SampleWeightsExplorerProps {
  initialScheme?: WeightingScheme
}

export default function SampleWeightsExplorer({
  initialScheme = 'ipw',
}: SampleWeightsExplorerProps): React.ReactElement {
  const [scheme, setScheme] = useState<WeightingScheme>(initialScheme)
  const [skewIntensity, setSkewIntensity] = useState<number>(3) // 1 (mild) to 5 (extreme)
  const [clippingCap, setClippingCap] = useState<number>(15) // max allowed weight
  const [enableClipping, setEnableClipping] = useState<boolean>(true)

  // Generate 30 representative sample weights based on scheme and skew
  const { rawWeights, processedWeights, nominalN, essRaw, essProcessed, deff } = useMemo(() => {
    const n = 1000
    const samplePoints = 25
    const weights: number[] = []

    for (let i = 0; i < samplePoints; i++) {
      let w = 1.0
      const t = i / (samplePoints - 1) // 0 to 1

      if (scheme === 'ipw') {
        // IPW: propensity e(x) close to 0 or 1 creates massive weight spikes 1/e(x)
        const e = 0.01 + 0.98 * (1 - Math.pow(t, skewIntensity))
        w = 1 / Math.max(0.01, e)
      } else if (scheme === 'covariate_shift') {
        // Density ratio: exponential shift
        w = Math.exp((t - 0.5) * skewIntensity * 1.5)
      } else if (scheme === 'recency') {
        // Exponential decay backwards from current time
        const decayRate = 0.3 * skewIntensity
        w = Math.exp(decayRate * (i - (samplePoints - 1)))
      } else {
        // Survey sampling: stratified oversampling of rare strata
        w = i < 3 ? Math.pow(skewIntensity, 2.2) * 2 : 1.0
      }
      weights.push(w)
    }

    // Compute Kish's Effective Sample Size on the simulated distribution
    // Scaling to nominal N = 1000
    const sumRaw = weights.reduce((a, b) => a + b, 0)
    const sumSqRaw = weights.reduce((a, b) => a + b * b, 0)
    const essRawVal = Math.round((n * (sumRaw * sumRaw)) / (weights.length * sumSqRaw))

    // Clipped & normalized weights
    const clipped = weights.map((w) => (enableClipping ? Math.min(clippingCap, w) : w))
    const sumClipped = clipped.reduce((a, b) => a + b, 0)
    const sumSqClipped = clipped.reduce((a, b) => a + b * b, 0)
    const essProcessedVal = Math.round((n * (sumClipped * sumClipped)) / (clipped.length * sumSqClipped))

    const deffVal = n / Math.max(1, essProcessedVal)

    return {
      rawWeights: weights,
      processedWeights: clipped,
      nominalN: n,
      essRaw: essRawVal,
      essProcessed: essProcessedVal,
      deff: deffVal,
    }
  }, [scheme, skewIntensity, clippingCap, enableClipping])

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
            Sample Weighting & Effective Sample Size (Kish) Explorer
          </h4>
          <span
            style={{
              fontSize: '0.74rem',
              padding: '0.2rem 0.55rem',
              borderRadius: '999px',
              background: deff > 2.5 ? 'rgba(244, 63, 94, 0.15)' : 'rgba(52, 211, 153, 0.15)',
              color: deff > 2.5 ? '#f43f5e' : '#34d399',
              fontWeight: 600,
            }}
          >
            Design Effect Deff: {deff.toFixed(2)}x
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
          Sample weights rebalance datasets for covariate shift, survey design, or causal propensity, but extreme
          weights inflate estimator variance and collapse statistical power.
        </p>
      </div>

      {/* Scheme Selection Tabs */}
      <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '1.25rem', flexWrap: 'wrap' }}>
        {(
          [
            ['ipw', 'Inverse Propensity (IPW)'],
            ['covariate_shift', 'Covariate Shift Density Ratio'],
            ['recency', 'Temporal Recency Decay'],
            ['survey', 'Survey Design / Horvitz-Thompson'],
          ] as const
        ).map(([sKey, label]) => (
          <button
            key={sKey}
            type="button"
            onClick={() => setScheme(sKey)}
            style={{
              flex: 1,
              minWidth: '160px',
              padding: '0.4rem 0.6rem',
              fontSize: '0.75rem',
              borderRadius: '5px',
              border: scheme === sKey ? '1px solid #38bdf8' : '1px solid #444',
              background: scheme === sKey ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
              color: scheme === sKey ? '#38bdf8' : '#aaa',
              cursor: 'pointer',
              fontWeight: scheme === sKey ? 600 : 400,
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Sliders Control Row */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '0.85rem',
          marginBottom: '1.25rem',
          padding: '0.75rem',
          borderRadius: '8px',
          background: 'rgba(0, 0, 0, 0.25)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
        }}
      >
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
            <label style={{ fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', color: '#94a3b8' }}>
              Skew / Shift Intensity: <strong style={{ color: '#38bdf8' }}>Level {skewIntensity} of 5</strong>
            </label>
          </div>
          <input
            type="range"
            min={1}
            max={5}
            step={1}
            value={skewIntensity}
            onChange={(e) => setSkewIntensity(parseInt(e.target.value, 10))}
            style={{ width: '100%', cursor: 'pointer', accentColor: '#38bdf8' }}
          />
          <span style={{ fontSize: '0.68rem', color: '#64748b' }}>Controls severity of weight disparity</span>
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
            <label style={{ fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', color: '#94a3b8' }}>
              Weight Clipping Cap (w_max): <strong style={{ color: '#f59e0b' }}>{clippingCap}x</strong>
            </label>
          </div>
          <input
            type="range"
            min={2}
            max={35}
            step={1}
            value={clippingCap}
            onChange={(e) => setClippingCap(parseInt(e.target.value, 10))}
            disabled={!enableClipping}
            style={{ width: '100%', cursor: enableClipping ? 'pointer' : 'not-allowed', accentColor: '#f59e0b' }}
          />
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', marginTop: '0.2rem' }}>
            <input
              type="checkbox"
              id="clip_toggle"
              checked={enableClipping}
              onChange={(e) => setEnableClipping(e.target.checked)}
              style={{ cursor: 'pointer' }}
            />
            <label htmlFor="clip_toggle" style={{ fontSize: '0.68rem', color: '#cbd5e1', cursor: 'pointer' }}>
              Enable Weight Truncation (Winsorization)
            </label>
          </div>
        </div>
      </div>

      {/* Kish Effective Sample Size Banner */}
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
            Nominal Sample Size (N)
          </span>
          <strong style={{ fontSize: '1.35rem', color: '#f8fafc' }}>{nominalN.toLocaleString()}</strong>
          <span style={{ display: 'block', fontSize: '0.7rem', color: '#64748b', marginTop: '0.2rem' }}>
            Total rows in training dataset
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
            Kish Effective Sample Size (n_eff)
          </span>
          <strong
            style={{
              fontSize: '1.35rem',
              color: essProcessed < 350 ? '#f43f5e' : essProcessed < 700 ? '#f59e0b' : '#34d399',
            }}
          >
            {essProcessed.toLocaleString()} rows
          </strong>
          <span style={{ display: 'block', fontSize: '0.7rem', color: '#64748b', marginTop: '0.2rem' }}>
            {enableClipping ? `Unclipped raw ESS: ${essRaw}` : 'Raw unclipped variance'}
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
            Variance Inflation Factor (Deff)
          </span>
          <strong style={{ fontSize: '1.35rem', color: deff > 2.5 ? '#f43f5e' : '#38bdf8' }}>{deff.toFixed(2)}x</strong>
          <span style={{ display: 'block', fontSize: '0.7rem', color: '#64748b', marginTop: '0.2rem' }}>
            {deff > 2.5 ? 'Severe loss of statistical precision' : 'Manageable variance trade-off'}
          </span>
        </div>
      </div>

      {/* Visual Bar Chart of Sample Weight Disparity */}
      <div
        style={{
          padding: '0.85rem',
          borderRadius: '8px',
          background: 'rgba(0, 0, 0, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
          marginBottom: '1rem',
        }}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: '0.72rem',
            color: '#94a3b8',
            marginBottom: '0.5rem',
          }}
        >
          <span>Simulated Sample Weight Profile (25 Exemplar Instances)</span>
          <span style={{ color: '#38bdf8' }}>Max Weight: {Math.max(...processedWeights).toFixed(1)}x</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'flex-end', gap: '3px', height: '80px', paddingTop: '10px' }}>
          {processedWeights.map((w, idx) => {
            const maxVal = Math.max(10, Math.max(...processedWeights))
            const heightPct = Math.max(6, Math.min(100, Math.round((w / maxVal) * 100)))
            const isClipped = enableClipping && rawWeights[idx] > clippingCap

            return (
              <div
                key={idx}
                style={{
                  flex: 1,
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'flex-end',
                  alignItems: 'center',
                }}
                title={`Sample #${idx + 1}: weight = ${w.toFixed(2)}${isClipped ? ' (CLIPPED from ' + rawWeights[idx].toFixed(1) + ')' : ''}`}
              >
                <div
                  style={{
                    width: '100%',
                    height: `${heightPct}%`,
                    background: isClipped ? '#f59e0b' : w > 5 ? '#f43f5e' : '#38bdf8',
                    borderRadius: '2px 2px 0 0',
                    transition: 'height 0.2s ease',
                  }}
                />
              </div>
            )
          })}
        </div>
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
        <span style={{ fontWeight: 600, color: '#f8fafc' }}>The Bias-Variance Trade-off in Weighting: </span>
        Sample weights eliminate bias (e.g. confounding in IPW or distribution mismatch in covariate shift). However, as
        weights diverge, Kish's effective sample size collapses from {nominalN} to {essProcessed}, multiplying model
        variance. <strong>Weight clipping (Winsorization)</strong> limits the maximum weight, accepting a small amount
        of residual bias to rescue statistical power.
      </div>
    </div>
  )
}
