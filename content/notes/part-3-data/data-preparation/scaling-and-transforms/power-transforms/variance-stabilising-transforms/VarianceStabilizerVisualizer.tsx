import React, { useState, useMemo } from 'react'

type DataType = 'poisson' | 'binomial'
type TransformMode = 'raw' | 'sqrt' | 'anscombe' | 'arcsin'

export const VarianceStabilizerVisualizer: React.FC = () => {
  const [dataType, setDataType] = useState<DataType>('poisson')
  const [transformMode, setTransformMode] = useState<TransformMode>('raw')
  const [sampleScale, setSampleScale] = useState<number>(1.0)

  // Generate synthetic groups with varying means
  // Group i has mean mu_i. Each group has 40 observations drawn from Poisson or Binomial.
  const groupStats = useMemo(() => {
    const numGroups = 8
    const nObs = 35
    let seed = 777
    const rand = () => {
      seed = (seed * 16807) % 2147483647
      return seed / 2147483647
    }

    const normalRand = () => {
      const u1 = Math.max(1e-5, rand())
      const u2 = rand()
      return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2)
    }

    const groups: {
      groupId: number
      targetMean: number
      rawValues: number[]
      transformedValues: number[]
      calcMean: number
      calcVar: number
    }[] = []

    for (let g = 0; g < numGroups; g++) {
      let targetMean = 0
      const rawVals: number[] = []

      if (dataType === 'poisson') {
        // Means from 2 to 40
        targetMean = (3 + g * 5) * sampleScale
        for (let i = 0; i < nObs; i++) {
          // Approximate Poisson using Gaussian with mean=targetMean, var=targetMean
          const val = Math.max(0, Math.round(targetMean + normalRand() * Math.sqrt(targetMean)))
          rawVals.push(val)
        }
      } else {
        // Binomial proportions: p from 0.05 to 0.95, n = 50
        const nTrials = 50
        const p = 0.05 + g * 0.12
        targetMean = p
        for (let i = 0; i < nObs; i++) {
          const successes = Math.max(
            0,
            Math.min(nTrials, Math.round(nTrials * p + normalRand() * Math.sqrt(nTrials * p * (1 - p)))),
          )
          rawVals.push(successes / nTrials)
        }
      }

      // Apply transform
      const transVals = rawVals.map((y) => {
        if (transformMode === 'raw') return y
        if (transformMode === 'sqrt') return Math.sqrt(y)
        if (transformMode === 'anscombe') return 2 * Math.sqrt(y + 3 / 8)
        if (transformMode === 'arcsin') {
          const clamped = Math.max(0, Math.min(1, y))
          return 2 * Math.asin(Math.sqrt(clamped))
        }
        return y
      })

      // Compute sample mean and sample variance of transformed values
      const sum = transVals.reduce((acc, v) => acc + v, 0)
      const m = sum / nObs
      const sqDiff = transVals.reduce((acc, v) => acc + Math.pow(v - m, 2), 0)
      const v = sqDiff / (nObs - 1)

      groups.push({
        groupId: g + 1,
        targetMean: Math.round(targetMean * 100) / 100,
        rawValues: rawVals,
        transformedValues: transVals,
        calcMean: Math.round(m * 100) / 100,
        calcVar: Math.round(v * 1000) / 1000,
      })
    }

    return groups
  }, [dataType, transformMode, sampleScale])

  // Dispersion metric: maxVar / minVar
  const dispersionMetric = useMemo(() => {
    const vars = groupStats.map((g) => g.calcVar).filter((v) => v > 0)
    const minV = Math.min(...vars)
    const maxV = Math.max(...vars)
    const ratio = minV > 0 ? maxV / minV : 1
    return {
      minV: minV.toFixed(3),
      maxV: maxV.toFixed(3),
      ratio: ratio.toFixed(2),
    }
  }, [groupStats])

  const maxMean = Math.max(...groupStats.map((g) => g.calcMean), 1)
  const maxVar = Math.max(...groupStats.map((g) => g.calcVar), 1)

  return (
    <div className="not-prose my-6 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-col gap-1 border-b border-border pb-4">
        <h3 className="text-base font-semibold text-foreground">Variance-Stabilising Transformation (VST) Explorer</h3>
        <p className="text-xs text-muted-foreground">
          Observe how the Delta Method derives functions $g(y) = \int 1/\sqrt{'{'}v(\mu){'}'} d\mu$ that neutralize
          mean-variance dependencies into constant homoscedastic variance.
        </p>
      </div>

      {/* Control Bar */}
      <div className="mb-5 grid grid-cols-1 gap-4 text-xs md:grid-cols-3">
        <div>
          <label className="mb-1 block font-medium text-foreground">Observation Data Type:</label>
          <div className="flex gap-1.5">
            {[
              { id: 'poisson', label: 'Poisson Counts (Var &prop; &mu;)' },
              { id: 'binomial', label: 'Proportions (Var &prop; p(1-p))' },
            ].map((opt) => (
              <button
                key={opt.id}
                onClick={() => {
                  setDataType(opt.id as DataType)
                  setTransformMode('raw')
                }}
                className={`flex-1 rounded border px-2 py-1.5 text-xs font-medium transition-colors ${
                  dataType === opt.id
                    ? 'border-primary bg-primary text-primary-foreground'
                    : 'border-transparent bg-muted/40 text-muted-foreground hover:bg-muted'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="mb-1 block font-medium text-foreground">Transform Function:</label>
          <div className="flex flex-wrap gap-1.5">
            {dataType === 'poisson' ? (
              <>
                {[
                  { id: 'raw', label: 'Raw y' },
                  { id: 'sqrt', label: '&radic;y' },
                  { id: 'anscombe', label: 'Anscombe: 2&radic;(y + 3/8)' },
                ].map((opt) => (
                  <button
                    key={opt.id}
                    onClick={() => setTransformMode(opt.id as TransformMode)}
                    className={`rounded border px-2.5 py-1 text-xs font-medium transition-colors ${
                      transformMode === opt.id
                        ? 'border-primary bg-primary text-primary-foreground'
                        : 'border-transparent bg-muted/40 text-muted-foreground hover:bg-muted'
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </>
            ) : (
              <>
                {[
                  { id: 'raw', label: 'Raw Proportion p' },
                  { id: 'arcsin', label: '2 &middot; arcsin(&radic;p)' },
                ].map((opt) => (
                  <button
                    key={opt.id}
                    onClick={() => setTransformMode(opt.id as TransformMode)}
                    className={`rounded border px-2.5 py-1 text-xs font-medium transition-colors ${
                      transformMode === opt.id
                        ? 'border-primary bg-primary text-primary-foreground'
                        : 'border-transparent bg-muted/40 text-muted-foreground hover:bg-muted'
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </>
            )}
          </div>
        </div>

        <div>
          <label className="mb-1 block font-medium text-foreground">
            Mean Scaling Factor: <span className="font-mono text-primary">{sampleScale.toFixed(1)}x</span>
          </label>
          <input
            type="range"
            min={0.5}
            max={2.0}
            step={0.1}
            value={sampleScale}
            onChange={(e) => setSampleScale(Number(e.target.value))}
            className="w-full accent-primary"
          />
          <span className="mt-0.5 block text-[10px] text-muted-foreground">
            Scales baseline arrival intensity or trial size.
          </span>
        </div>
      </div>

      {/* Main Plot: Mean vs Variance Relationship */}
      <div className="mb-5 grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-border bg-card/50 p-3 lg:col-span-2">
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="font-semibold text-foreground">
              Mean vs. Variance Across Groups ($\mu$ vs. $\sigma^2$)
            </span>
            <span className="text-[11px] text-muted-foreground">
              {transformMode === 'raw'
                ? 'Severe Heteroscedasticity (Variance expands with mean)'
                : 'Stabilized Homoscedasticity (Flat constant variance)'}
            </span>
          </div>

          <svg viewBox="0 0 460 210" className="h-auto w-full rounded bg-muted/10">
            {/* Guide grid */}
            <line x1="40" y1="180" x2="430" y2="180" stroke="currentColor" strokeOpacity={0.2} />
            <line x1="40" y1="20" x2="40" y2="180" stroke="currentColor" strokeOpacity={0.2} />

            <text x="430" y="195" textAnchor="end" className="fill-muted-foreground text-[9px]">
              Group Mean E[g(Y)]
            </text>
            <text x="35" y="25" textAnchor="end" className="fill-muted-foreground text-[9px]">
              Var[g(Y)]
            </text>

            {/* Target theoretical flat line when transformed */}
            {transformMode !== 'raw' && (
              <g>
                <line
                  x1="40"
                  y1={180 - (dataType === 'poisson' ? (1 / maxVar) * 150 : (0.02 / maxVar) * 150)}
                  x2="430"
                  y2={180 - (dataType === 'poisson' ? (1 / maxVar) * 150 : (0.02 / maxVar) * 150)}
                  stroke="#10b981"
                  strokeWidth="1.5"
                  strokeDasharray="4 3"
                />
                <text
                  x="425"
                  y={175 - (dataType === 'poisson' ? (1 / maxVar) * 150 : (0.02 / maxVar) * 150)}
                  textAnchor="end"
                  className="fill-emerald-500 text-[9px] font-semibold"
                >
                  Theoretical Target Var = {dataType === 'poisson' ? '1.0 (Anscombe)' : '1/n'}
                </text>
              </g>
            )}

            {/* Plot group points */}
            {groupStats.map((g) => {
              const xPos = 40 + (g.calcMean / maxMean) * 370
              const yPos = 180 - (g.calcVar / maxVar) * 150
              return (
                <g key={g.groupId}>
                  <circle
                    cx={xPos}
                    cy={yPos}
                    r={5}
                    fill={transformMode === 'raw' ? '#f43f5e' : '#10b981'}
                    fillOpacity={0.8}
                    stroke="#0f172a"
                    strokeWidth="1"
                  />
                  <text x={xPos} y={yPos - 8} textAnchor="middle" className="fill-foreground font-mono text-[8px]">
                    G{g.groupId}
                  </text>
                </g>
              )
            })}
          </svg>
        </div>

        {/* Diagnostic Dispersion Card */}
        <div className="flex flex-col justify-between rounded-lg border border-border bg-card/50 p-4">
          <div>
            <div className="mb-2 text-xs font-bold tracking-wider text-foreground uppercase">
              Homoscedasticity Ratio
            </div>
            <div className="mb-3 rounded-lg border border-border bg-muted/20 p-3">
              <span className="block text-[11px] text-muted-foreground">
                Variance Dispersion Ratio (&sigma;<sub>max</sub>
                <sup>2</sup> / &sigma;<sub>min</sub>
                <sup>2</sup>):
              </span>
              <div
                className={`mt-1 font-mono text-2xl font-bold ${
                  Number(dispersionMetric.ratio) < 2.5 ? 'text-emerald-500' : 'text-rose-500'
                }`}
              >
                {dispersionMetric.ratio}x
              </div>
              <p className="mt-1 text-[10px] text-muted-foreground">
                {Number(dispersionMetric.ratio) > 5
                  ? 'Severe variance explosion: OLS assumption of homoscedasticity is violated.'
                  : 'Near 1.0x: Variance is constant across the entire operating range!'}
              </p>
            </div>

            <div className="space-y-1.5 text-xs">
              <div className="flex justify-between border-b border-border/50 py-1">
                <span className="text-muted-foreground">Min Group Variance:</span>
                <span className="font-mono font-medium text-foreground">{dispersionMetric.minV}</span>
              </div>
              <div className="flex justify-between border-b border-border/50 py-1">
                <span className="text-muted-foreground">Max Group Variance:</span>
                <span className="font-mono font-medium text-foreground">{dispersionMetric.maxV}</span>
              </div>
            </div>
          </div>

          <div className="mt-3 border-t border-border pt-3 text-[11px] text-muted-foreground">
            <span className="mb-0.5 block font-semibold text-primary">Delta Method Derivation:</span>
            <code className="block font-mono text-foreground">
              g(&mu;) = &int; [v(&mu;)]<sup>-1/2</sup> d&mu;
            </code>
          </div>
        </div>
      </div>
    </div>
  )
}
