import React, { useState, useMemo } from 'react'

interface DataPoint {
  id: number
  x: number
  y: number
  isRare: boolean
  isSynthetic?: boolean
  weight: number
}

export const ImbalancedRegressionExplorer: React.FC = () => {
  const [strategy, setStrategy] = useState<'standard' | 'weighted' | 'smoter' | 'stratified'>('standard')
  const [tailThreshold, setTailThreshold] = useState<number>(65)
  const [weightPower, setWeightPower] = useState<number>(0.6)
  const [smoterRatio, setSmoterRatio] = useState<number>(1.5)

  // Generate synthetic imbalanced continuous dataset:
  // Feature x in [0, 10]. True latent function has a steep rise for high x,
  // but high x/y values are exponentially rare.
  const rawData: DataPoint[] = useMemo(() => {
    const points: DataPoint[] = []
    const count = 120
    // Deterministic PRNG
    let seed = 42
    const random = () => {
      seed = (seed * 9301 + 49297) % 233280
      return seed / 233280
    }

    for (let i = 0; i < count; i++) {
      // Skewed x distribution towards lower values: x in [0, 10]
      const u = random()
      const x = u < 0.85 ? u * 6 : 6 + (u - 0.85) * 26.6 // 85% below 6, 15% in [6, 10]
      // True relation: quadratic non-linear growth with noise
      const cleanY = 15 + 0.6 * x + 0.75 * Math.pow(x, 2)
      const noise = (random() - 0.5) * 10
      const y = Math.max(5, Math.min(100, cleanY + noise))
      points.push({
        id: i,
        x: Math.round(x * 100) / 100,
        y: Math.round(y * 10) / 10,
        isRare: y >= tailThreshold,
        weight: 1.0,
      })
    }
    return points
  }, [tailThreshold])

  // Relevance function phi(y): smooth sigmoid transitioning at tailThreshold
  const getRelevance = (y: number, thresh: number): number => {
    const k = 0.2
    return 1 / (1 + Math.exp(-k * (y - thresh)))
  }

  // Empirical target density KDE approximation over [0, 100]
  const densityValues = useMemo(() => {
    const bins = 40
    const step = 100 / bins
    const counts = new Array(bins).fill(0)
    rawData.forEach((p) => {
      const idx = Math.min(bins - 1, Math.max(0, Math.floor(p.y / step)))
      counts[idx]++
    })
    const maxCount = Math.max(...counts, 1)
    return counts.map((c, i) => ({
      y: (i + 0.5) * step,
      density: c / (rawData.length * step),
      relHeight: c / maxCount,
    }))
  }, [rawData])

  // Augmented/Weighted data depending on strategy
  const processedData = useMemo(() => {
    if (strategy === 'standard' || strategy === 'stratified') {
      return rawData.map((p) => ({
        ...p,
        isRare: p.y >= tailThreshold,
        weight: 1.0,
      }))
    }

    if (strategy === 'weighted') {
      // Inverse density weighting tempered by weightPower
      return rawData.map((p) => {
        const phi = getRelevance(p.y, tailThreshold)
        // Find local density
        const binIdx = Math.min(densityValues.length - 1, Math.max(0, Math.floor(p.y / (100 / densityValues.length))))
        const localDens = Math.max(0.005, densityValues[binIdx]?.density || 0.01)
        const invDensWeight = Math.pow(1 / localDens, weightPower)
        const w = (1 - phi) * 0.5 + phi * invDensWeight * 0.15
        return {
          ...p,
          isRare: p.y >= tailThreshold,
          weight: Math.max(0.2, Math.min(10, Math.round(w * 100) / 100)),
        }
      })
    }

    if (strategy === 'smoter') {
      // SMOTER: interpolate rare points in tail
      const rarePoints = rawData.filter((p) => p.y >= tailThreshold)
      const synthCount = Math.round(rarePoints.length * smoterRatio)
      const syntheticPoints: DataPoint[] = []

      let prng = 1337
      const pseudoRand = () => {
        prng = (prng * 16807) % 2147483647
        return prng / 2147483647
      }

      for (let s = 0; s < synthCount; s++) {
        if (rarePoints.length >= 2) {
          const idxA = Math.floor(pseudoRand() * rarePoints.length)
          let idxB = Math.floor(pseudoRand() * rarePoints.length)
          if (idxA === idxB) idxB = (idxA + 1) % rarePoints.length
          const pA = rarePoints[idxA]
          const pB = rarePoints[idxB]
          const alpha = 0.2 + pseudoRand() * 0.6 // between 0.2 and 0.8
          const newX = pA.x + alpha * (pB.x - pA.x)
          const newY = pA.y + alpha * (pB.y - pA.y)
          syntheticPoints.push({
            id: 1000 + s,
            x: Math.round(newX * 100) / 100,
            y: Math.round(newY * 10) / 10,
            isRare: true,
            isSynthetic: true,
            weight: 1.0,
          })
        }
      }

      return [...rawData.map((p) => ({ ...p, isRare: p.y >= tailThreshold, weight: 1.0 })), ...syntheticPoints]
    }

    return rawData
  }, [rawData, strategy, tailThreshold, weightPower, smoterRatio, densityValues])

  // Fit quadratic polynomial model y ~ w0 + w1*x + w2*x^2 via weighted least squares
  const modelCoeffs = useMemo(() => {
    // Weighted least squares for [1, x, x^2]
    let s00 = 0,
      s01 = 0,
      s02 = 0
    let s11 = 0,
      s12 = 0,
      s22 = 0
    let sy0 = 0,
      sy1 = 0,
      sy2 = 0

    processedData.forEach((p) => {
      const w = p.weight
      const x1 = p.x
      const x2 = p.x * p.x
      s00 += w
      s01 += w * x1
      s02 += w * x2
      s11 += w * x1 * x1
      s12 += w * x1 * x2
      s22 += w * x2 * x2
      sy0 += w * p.y
      sy1 += w * p.y * x1
      sy2 += w * p.y * x2
    })

    // 3x3 linear system solver (Gaussian elimination)
    const A = [
      [s00, s01, s02, sy0],
      [s01, s11, s12, sy1],
      [s02, s12, s22, sy2],
    ]

    for (let i = 0; i < 3; i++) {
      let maxRow = i
      for (let k = i + 1; k < 3; k++) {
        if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) maxRow = k
      }
      const tmp = A[i]
      A[i] = A[maxRow]
      A[maxRow] = tmp

      const pivot = A[i][i] || 1e-6
      for (let j = i; j <= 3; j++) A[i][j] /= pivot
      for (let k = 0; k < 3; k++) {
        if (k !== i) {
          const factor = A[k][i]
          for (let j = i; j <= 3; j++) {
            A[k][j] -= factor * A[i][j]
          }
        }
      }
    }

    return {
      b0: A[0][3],
      b1: A[1][3],
      b2: A[2][3],
    }
  }, [processedData])

  // Calculate evaluation metrics on ground truth original rawData (un-augmented)
  const metrics = useMemo(() => {
    let overallSqErr = 0
    let tailSqErr = 0
    let tailCount = 0
    let bulkSqErr = 0
    let bulkCount = 0

    rawData.forEach((p) => {
      const pred = modelCoeffs.b0 + modelCoeffs.b1 * p.x + modelCoeffs.b2 * Math.pow(p.x, 2)
      const errSq = Math.pow(p.y - pred, 2)
      overallSqErr += errSq
      if (p.isRare) {
        tailSqErr += errSq
        tailCount++
      } else {
        bulkSqErr += errSq
        bulkCount++
      }
    })

    const rmseOverall = Math.sqrt(overallSqErr / rawData.length)
    const rmseTail = tailCount > 0 ? Math.sqrt(tailSqErr / tailCount) : 0
    const rmseBulk = bulkCount > 0 ? Math.sqrt(bulkSqErr / bulkCount) : 0

    return {
      rmseOverall: rmseOverall.toFixed(2),
      rmseTail: rmseTail.toFixed(2),
      rmseBulk: rmseBulk.toFixed(2),
      tailCount,
      bulkCount,
    }
  }, [rawData, modelCoeffs])

  // Curve points for rendering fitted curve
  const curvePoints = useMemo(() => {
    const pts: string[] = []
    const steps = 40
    for (let i = 0; i <= steps; i++) {
      const xVal = (i / steps) * 10
      const yVal = modelCoeffs.b0 + modelCoeffs.b1 * xVal + modelCoeffs.b2 * Math.pow(xVal, 2)
      const clampedY = Math.max(0, Math.min(105, yVal))
      // Map to SVG coordinates: W=520, H=220, padding=35
      const svgX = 35 + (xVal / 10) * 460
      const svgY = 205 - (clampedY / 105) * 180
      pts.push(`${svgX.toFixed(1)},${svgY.toFixed(1)}`)
    }
    return pts.join(' ')
  }, [modelCoeffs])

  return (
    <div className="not-prose my-6 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-col gap-1 border-b border-border pb-4">
        <h3 className="text-base font-semibold text-foreground">Interactive Imbalanced Regression Laboratory</h3>
        <p className="text-xs text-muted-foreground">
          Observe how rare continuous target values cause standard MSE to underfit extreme tails, and how relevance
          weighting &amp; SMOTER restore tail fidelity.
        </p>
      </div>

      {/* Control Bar */}
      <div className="mb-5 grid grid-cols-1 gap-4 text-xs md:grid-cols-4">
        <div>
          <label className="mb-1 block font-medium text-foreground">Strategy:</label>
          <div className="flex flex-col gap-1.5">
            {[
              { id: 'standard', label: 'Standard OLS / MSE' },
              { id: 'weighted', label: 'Relevance-Weighted (KDE)' },
              { id: 'smoter', label: 'SMOTER (Interpolation)' },
              { id: 'stratified', label: 'Binned Stratified Split' },
            ].map((opt) => (
              <button
                key={opt.id}
                onClick={() => setStrategy(opt.id as any)}
                className={`rounded border px-2.5 py-1.5 text-left text-xs transition-colors ${
                  strategy === opt.id
                    ? 'border-primary bg-primary font-medium text-primary-foreground'
                    : 'border-transparent bg-muted/40 text-muted-foreground hover:bg-muted'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="mb-1 block font-medium text-foreground">
            Tail Threshold ($t_R$): <span className="font-mono text-primary">{tailThreshold}</span>
          </label>
          <input
            type="range"
            min={45}
            max={85}
            step={1}
            value={tailThreshold}
            onChange={(e) => setTailThreshold(Number(e.target.value))}
            className="w-full accent-primary"
          />
          <span className="mt-1 block text-[11px] text-muted-foreground">
            Defines the rare domain where relevance &phi;(y) &asymp; 1.
          </span>
        </div>

        {strategy === 'weighted' && (
          <div>
            <label className="mb-1 block font-medium text-foreground">
              Inverse Weight Power ($\alpha$): <span className="font-mono text-primary">{weightPower.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0.1}
              max={1.0}
              step={0.05}
              value={weightPower}
              onChange={(e) => setWeightPower(Number(e.target.value))}
              className="w-full accent-primary"
            />
            <span className="mt-1 block text-[11px] text-muted-foreground">
              Controls penalty strength w(y) &prop; (f&#770;(y))<sup>-&alpha;</sup>.
            </span>
          </div>
        )}

        {strategy === 'smoter' && (
          <div>
            <label className="mb-1 block font-medium text-foreground">
              SMOTER Oversampling Ratio: <span className="font-mono text-primary">{smoterRatio}x</span>
            </label>
            <input
              type="range"
              min={0.5}
              max={3.0}
              step={0.5}
              value={smoterRatio}
              onChange={(e) => setSmoterRatio(Number(e.target.value))}
              className="w-full accent-primary"
            />
            <span className="mt-1 block text-[11px] text-muted-foreground">
              Synthetic tail cases created along feature vectors.
            </span>
          </div>
        )}

        {strategy === 'stratified' && (
          <div className="rounded border border-border bg-muted/30 p-2.5">
            <span className="mb-1 block font-semibold text-foreground">Continuous Stratification</span>
            <p className="text-[11px] leading-relaxed text-muted-foreground">
              Splits target distribution into 5 quantile bins ($Q_1$ to $Q_5$). Guarantees test and training splits hold
              an exact {((metrics.tailCount / rawData.length) * 100).toFixed(0)}% tail quota.
            </p>
          </div>
        )}

        {strategy === 'standard' && (
          <div className="rounded border border-border bg-muted/30 p-2.5">
            <span className="mb-1 block font-semibold text-foreground">Baseline Uniform MSE</span>
            <p className="text-[11px] leading-relaxed text-muted-foreground">
              Every data point receives uniform weight $w_i = 1$. The 85% bulk points pull the regression fit toward the
              low-y regime.
            </p>
          </div>
        )}
      </div>

      {/* Main Scatter & Regression Plot */}
      <div className="mb-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-border bg-card/50 p-3 lg:col-span-2">
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="font-medium text-foreground">Feature Space Fit ($x \to y$)</span>
            <div className="flex items-center gap-3 text-[11px]">
              <span className="flex items-center gap-1">
                <span className="inline-block h-2.5 w-2.5 rounded-full bg-blue-500"></span> Common Bulk
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block h-2.5 w-2.5 rounded-full bg-amber-500"></span> Rare Tail ($y \ge{' '}
                {tailThreshold}$)
              </span>
              {strategy === 'smoter' && (
                <span className="flex items-center gap-1">
                  <span className="inline-block h-2.5 w-2.5 rounded-full border border-dashed border-emerald-500 bg-emerald-500/30"></span>{' '}
                  SMOTER Synth
                </span>
              )}
              <span className="flex items-center gap-1">
                <span className="inline-block h-0.5 w-4 bg-rose-500"></span> Fitted Model
              </span>
            </div>
          </div>

          <svg viewBox="0 0 520 230" className="h-auto w-full rounded bg-muted/10">
            {/* Grid Lines */}
            {[0, 25, 50, 75, 100].map((val) => {
              const y = 205 - (val / 105) * 180
              return (
                <g key={`y-${val}`}>
                  <line x1="35" y1={y} x2="495" y2={y} stroke="currentColor" strokeOpacity={0.1} />
                  <text x="28" y={y + 3} textAnchor="end" className="fill-muted-foreground text-[9px]">
                    {val}
                  </text>
                </g>
              )
            })}
            {[0, 2.5, 5, 7.5, 10].map((val) => {
              const x = 35 + (val / 10) * 460
              return (
                <g key={`x-${val}`}>
                  <line x1={x} y1="25" x2={x} y2="205" stroke="currentColor" strokeOpacity={0.1} />
                  <text x={x} y="218" textAnchor="middle" className="fill-muted-foreground text-[9px]">
                    {val}
                  </text>
                </g>
              )
            })}

            {/* Threshold Line */}
            {(() => {
              const threshY = 205 - (tailThreshold / 105) * 180
              return (
                <g>
                  <line
                    x1="35"
                    y1={threshY}
                    x2="495"
                    y2={threshY}
                    stroke="rgba(245, 158, 11, 0.4)"
                    strokeDasharray="4 3"
                    strokeWidth="1.5"
                  />
                  <text x="490" y={threshY - 4} textAnchor="end" className="fill-amber-500 text-[9px] font-medium">
                    Rare Threshold $t_R = {tailThreshold}$
                  </text>
                </g>
              )
            })()}

            {/* Scatter Data Points */}
            {processedData.map((p) => {
              const cx = 35 + (p.x / 10) * 460
              const cy = 205 - (Math.min(105, p.y) / 105) * 180
              const radius = strategy === 'weighted' ? Math.max(2.5, Math.min(6, Math.sqrt(p.weight) * 2.5)) : 3

              if (p.isSynthetic) {
                return (
                  <circle
                    key={`synth-${p.id}`}
                    cx={cx}
                    cy={cy}
                    r={3.5}
                    fill="#10b981"
                    fillOpacity={0.6}
                    stroke="#047857"
                    strokeWidth="1"
                    strokeDasharray="2 1"
                  />
                )
              }

              return (
                <circle
                  key={`pt-${p.id}`}
                  cx={cx}
                  cy={cy}
                  r={radius}
                  fill={p.isRare ? '#f59e0b' : '#3b82f6'}
                  fillOpacity={0.7}
                  stroke={p.isRare ? '#b45309' : '#1d4ed8'}
                  strokeWidth="0.8"
                />
              )
            })}

            {/* Fitted Regression Curve */}
            <polyline points={curvePoints} fill="none" stroke="#f43f5e" strokeWidth="2.5" />
          </svg>
        </div>

        {/* Target Density & Relevance Profile */}
        <div className="flex flex-col justify-between rounded-lg border border-border bg-card/50 p-3">
          <div>
            <div className="mb-2 flex items-center justify-between text-xs">
              <span className="font-medium text-foreground">Target Density &amp; Relevance $\phi(y)$</span>
            </div>
            <p className="mb-3 text-[11px] leading-tight text-muted-foreground">
              Distribution of $y$. Bars represent sample counts; the purple curve shows the continuous relevance
              function $\phi(y)$.
            </p>

            <svg viewBox="0 0 240 140" className="h-auto w-full rounded bg-muted/10">
              {/* Density Bars */}
              {densityValues.map((d, i) => {
                const barWidth = 200 / densityValues.length
                const barHeight = d.relHeight * 100
                const x = 20 + i * barWidth
                const y = 115 - barHeight
                const isTailBin = d.y >= tailThreshold
                return (
                  <rect
                    key={`bar-${i}`}
                    x={x}
                    y={y}
                    width={barWidth - 1}
                    height={barHeight}
                    fill={isTailBin ? 'rgba(245, 158, 11, 0.55)' : 'rgba(59, 130, 246, 0.45)'}
                  />
                )
              })}

              {/* Relevance Curve Overlay */}
              {(() => {
                const pts: string[] = []
                for (let yVal = 0; yVal <= 100; yVal += 2) {
                  const rel = getRelevance(yVal, tailThreshold)
                  const x = 20 + (yVal / 100) * 200
                  const y = 115 - rel * 100
                  pts.push(`${x.toFixed(1)},${y.toFixed(1)}`)
                }
                return <polyline points={pts.join(' ')} fill="none" stroke="#a855f7" strokeWidth="2" />
              })()}

              {/* Baseline axis */}
              <line x1="20" y1="115" x2="220" y2="115" stroke="currentColor" strokeOpacity={0.2} />
              <text x="20" y="128" textAnchor="middle" className="fill-muted-foreground text-[9px]">
                0
              </text>
              <text x="120" y="128" textAnchor="middle" className="fill-muted-foreground text-[9px]">
                50
              </text>
              <text x="220" y="128" textAnchor="middle" className="fill-muted-foreground text-[9px]">
                100
              </text>
              <text x="195" y="25" textAnchor="end" className="fill-purple-400 text-[9px] font-semibold">
                $\phi(y)=1$
              </text>
            </svg>
          </div>

          <div className="mt-3 grid grid-cols-2 gap-2 border-t border-border pt-3 text-center text-xs">
            <div className="rounded bg-muted/40 p-1.5">
              <span className="block text-[10px] text-muted-foreground">Bulk Samples ($y &lt; t_R$)</span>
              <span className="font-bold text-blue-500">{metrics.bulkCount}</span>
            </div>
            <div className="rounded bg-muted/40 p-1.5">
              <span className="block text-[10px] text-muted-foreground">Rare Tail ($y \ge t_R$)</span>
              <span className="font-bold text-amber-500">{metrics.tailCount}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Evaluation Metric Comparison */}
      <div className="rounded-lg border border-border bg-card/60 p-4">
        <h4 className="mb-2 text-xs font-semibold tracking-wider text-foreground uppercase">
          Diagnostic Evaluation on True Targets (Unmasked)
        </h4>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div className="rounded-lg border border-border bg-muted/20 p-3">
            <span className="block text-xs text-muted-foreground">Global RMSE (Bulk Dominated)</span>
            <div className="mt-1 font-mono text-xl font-bold text-foreground">{metrics.rmseOverall}</div>
            <p className="mt-1 text-[11px] text-muted-foreground">
              {strategy === 'standard'
                ? 'Standard MSE optimizes exclusively for this global metric.'
                : 'Slightly higher because rare tail loss is prioritized over bulk.'}
            </p>
          </div>

          <div className="rounded-lg border border-amber-500/20 bg-amber-500/10 p-3">
            <span className="block text-xs font-semibold text-amber-600 dark:text-amber-400">
              Rare Tail RMSE ($y \ge {tailThreshold}$)
            </span>
            <div className="mt-1 font-mono text-xl font-bold text-amber-600 dark:text-amber-400">
              {metrics.rmseTail}
            </div>
            <p className="mt-1 text-[11px] text-muted-foreground">
              {Number(metrics.rmseTail) > 15
                ? 'High error: standard fit flattens out and severely underpredicts extreme values.'
                : 'Substantially improved tail precision; model climbs towards extreme targets.'}
            </p>
          </div>

          <div className="rounded-lg border border-blue-500/20 bg-blue-500/10 p-3">
            <span className="block text-xs font-semibold text-blue-600 dark:text-blue-400">
              Common Bulk RMSE ($y &lt; {tailThreshold}$)
            </span>
            <div className="mt-1 font-mono text-xl font-bold text-blue-600 dark:text-blue-400">{metrics.rmseBulk}</div>
            <p className="mt-1 text-[11px] text-muted-foreground">
              Minimal degradation in common cases while unlocking extreme event forecastability.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
