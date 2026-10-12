import React, { useState, useMemo } from 'react'

// Approximation of inverse normal CDF (probit) using Winitzki approximation
function probit(p: number): number {
  if (p <= 0.0001) return -3.719
  if (p >= 0.9999) return 3.719
  // Rational approximation for normal inverse CDF
  const a1 = -3.969683028665376e1,
    a2 = 2.209460984245205e2,
    a3 = -2.759285104469687e2
  const a4 = 1.38357751867269e2,
    a5 = -3.066479806614716e1,
    a6 = 2.506628277459239
  const b1 = -5.447609879822406e1,
    b2 = 1.615858368580409e2,
    b3 = -1.556989798598866e2
  const b4 = 6.680131188771972e1,
    b5 = -1.328068155288572e1
  const c1 = -7.784894002430293e-3,
    c2 = -3.223964580411365e-1,
    c3 = -2.400758277161838
  const c4 = -2.549732539343734,
    c5 = 4.374664141464968,
    c6 = 2.938163982698783
  const d1 = 7.784695709041462e-3,
    d2 = 3.224671290700398e-1,
    d3 = 2.445134137142996
  const d4 = 3.754408661907416

  const q = p - 0.5
  if (Math.abs(q) <= 0.42) {
    const r = q * q
    return (
      (q * (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)) /
      (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    )
  } else {
    const r = q < 0 ? p : 1.0 - p
    const s = Math.log(-Math.log(r))
    let x = c1 + s * (c2 + s * (c3 + s * (c4 + s * (c5 + s * c6))))
    const y = 1.0 + s * (d1 + s * (d2 + s * (d3 + s * d4)))
    x = x / y
    return q < 0 ? -x : x
  }
}

export const QuantileTransformVisualizer: React.FC = () => {
  const [outputDist, setOutputDist] = useState<'uniform' | 'normal'>('normal')
  const [inputSkew, setInputSkew] = useState<'heavy_tail' | 'bimodal'>('heavy_tail')
  const [nQuantiles, setNQuantiles] = useState<number>(50)
  const [probeVal, setProbeVal] = useState<number>(45)

  // Generate synthetic training points
  const rawData: number[] = useMemo(() => {
    const pts: number[] = []
    const count = 200
    let seed = 99
    const rand = () => {
      seed = (seed * 16807) % 2147483647
      return seed / 2147483647
    }

    if (inputSkew === 'heavy_tail') {
      // Exponential distribution with a few extreme outliers
      for (let i = 0; i < count; i++) {
        const u = Math.max(0.001, rand())
        const x = -10 * Math.log(u)
        pts.push(Math.round(x * 10) / 10)
      }
      // Add extreme sensor spikes
      pts.push(85, 95, 120)
    } else {
      // Bimodal mixture: Mode 1 at ~15, Mode 2 at ~65
      for (let i = 0; i < count; i++) {
        const isMode1 = rand() < 0.65
        const mean = isMode1 ? 15 : 65
        const std = isMode1 ? 5 : 8
        const u1 = Math.max(0.001, rand())
        const u2 = rand()
        const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2)
        pts.push(Math.round(Math.max(0, mean + z0 * std) * 10) / 10)
      }
    }

    return pts.sort((a, b) => a - b)
  }, [inputSkew])

  // Compute empirical quantiles on rawData
  const quantileReferences = useMemo(() => {
    const N = rawData.length
    const quantiles: { q: number; x: number }[] = []
    for (let i = 0; i < nQuantiles; i++) {
      const q = i / (nQuantiles - 1)
      const idx = q * (N - 1)
      const low = Math.floor(idx)
      const high = Math.ceil(idx)
      const weight = idx - low
      const x = rawData[low] * (1 - weight) + rawData[high] * weight
      quantiles.push({ q, x })
    }
    return quantiles
  }, [rawData, nQuantiles])

  // Transform function using linear interpolation across quantiles
  const transform = useMemo(() => {
    return (x: number): { u: number; z: number } => {
      const qs = quantileReferences
      if (x <= qs[0].x) {
        return { u: 0.0001, z: probit(0.0001) }
      }
      if (x >= qs[qs.length - 1].x) {
        return { u: 0.9999, z: probit(0.9999) }
      }
      // Binary search segment
      let low = 0
      let high = qs.length - 1
      while (high - low > 1) {
        const mid = Math.floor((low + high) / 2)
        if (qs[mid].x <= x) low = mid
        else high = mid
      }
      const span = qs[high].x - qs[low].x || 1e-5
      const alpha = (x - qs[low].x) / span
      const u = Math.min(0.9999, Math.max(0.0001, qs[low].q + alpha * (qs[high].q - qs[low].q)))
      return { u, z: probit(u) }
    }
  }, [quantileReferences])

  // Transform all points
  const transformedPoints = useMemo(() => {
    return rawData.map((x) => {
      const res = transform(x)
      return outputDist === 'uniform' ? res.u : res.z
    })
  }, [rawData, transform, outputDist])

  const probeResult = useMemo(() => {
    return transform(probeVal)
  }, [probeVal, transform])

  // Histogram helper
  const inputBins = useMemo(() => {
    const bins = 25
    const maxVal = Math.max(...rawData, 100)
    const step = maxVal / bins
    const counts = new Array(bins).fill(0)
    rawData.forEach((x) => {
      const idx = Math.min(bins - 1, Math.max(0, Math.floor(x / step)))
      counts[idx]++
    })
    const maxCount = Math.max(...counts, 1)
    return counts.map((c, i) => ({
      x: (i + 0.5) * step,
      count: c,
      height: c / maxCount,
    }))
  }, [rawData])

  const outputBins = useMemo(() => {
    const bins = 25
    const isUniform = outputDist === 'uniform'
    const minVal = isUniform ? 0 : -3.5
    const maxVal = isUniform ? 1 : 3.5
    const step = (maxVal - minVal) / bins
    const counts = new Array(bins).fill(0)
    transformedPoints.forEach((val) => {
      const idx = Math.min(bins - 1, Math.max(0, Math.floor((val - minVal) / step)))
      counts[idx]++
    })
    const maxCount = Math.max(...counts, 1)
    return counts.map((c, i) => ({
      val: minVal + (i + 0.5) * step,
      count: c,
      height: c / maxCount,
    }))
  }, [transformedPoints, outputDist])

  return (
    <div className="not-prose my-6 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-col gap-1 border-b border-border pb-4">
        <h3 className="text-base font-semibold text-foreground">Quantile &amp; Rank Transformation Explorer</h3>
        <p className="text-xs text-muted-foreground">
          Quantile transformation maps arbitrary empirical distributions to exact uniform or standard normal targets by
          computing ranks and applying the inverse target CDF.
        </p>
      </div>

      {/* Control Bar */}
      <div className="mb-5 grid grid-cols-1 gap-4 text-xs md:grid-cols-4">
        <div>
          <label className="mb-1 block font-medium text-foreground">Output Distribution:</label>
          <div className="flex gap-1.5">
            {[
              { id: 'normal', label: 'Gaussian N(0, 1)' },
              { id: 'uniform', label: 'Uniform U(0, 1)' },
            ].map((opt) => (
              <button
                key={opt.id}
                onClick={() => setOutputDist(opt.id as any)}
                className={`flex-1 rounded border px-2 py-1.5 text-xs font-medium transition-colors ${
                  outputDist === opt.id
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
          <label className="mb-1 block font-medium text-foreground">Input Data Profile:</label>
          <div className="flex gap-1.5">
            {[
              { id: 'heavy_tail', label: 'Skewed + Outliers' },
              { id: 'bimodal', label: 'Bimodal Mixture' },
            ].map((opt) => (
              <button
                key={opt.id}
                onClick={() => setInputSkew(opt.id as any)}
                className={`flex-1 rounded border px-2 py-1.5 text-xs font-medium transition-colors ${
                  inputSkew === opt.id
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
          <label className="mb-1 block font-medium text-foreground">
            Quantile Knots (n_quantiles): <span className="font-mono text-primary">{nQuantiles}</span>
          </label>
          <input
            type="range"
            min={10}
            max={150}
            step={10}
            value={nQuantiles}
            onChange={(e) => setNQuantiles(Number(e.target.value))}
            className="w-full accent-primary"
          />
          <span className="mt-0.5 block text-[10px] text-muted-foreground">
            Fewer knots smooth out local sampling noise.
          </span>
        </div>

        <div>
          <label className="mb-1 block font-medium text-foreground">
            Test Probe ($x$): <span className="font-mono text-primary">{probeVal}</span>
          </label>
          <input
            type="range"
            min={0}
            max={130}
            step={1}
            value={probeVal}
            onChange={(e) => setProbeVal(Number(e.target.value))}
            className="w-full accent-primary"
          />
          <span className="mt-0.5 block text-[10px] text-muted-foreground">
            Test point mapped through training ECDF.
          </span>
        </div>
      </div>

      {/* Side-by-Side Histograms */}
      <div className="mb-5 grid grid-cols-1 gap-4 md:grid-cols-2">
        {/* Input Space */}
        <div className="rounded-lg border border-border bg-card/50 p-3">
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="font-semibold text-foreground">Original Distribution ($X$)</span>
            <span className="font-mono text-[11px] text-muted-foreground">
              Min: {rawData[0]} | Max: {rawData[rawData.length - 1]}
            </span>
          </div>

          <svg viewBox="0 0 300 130" className="h-auto w-full rounded bg-muted/10">
            {/* Guide lines */}
            <line x1="20" y1="110" x2="285" y2="110" stroke="currentColor" strokeOpacity={0.2} />

            {/* Input bars */}
            {inputBins.map((b, i) => {
              const barWidth = 265 / inputBins.length
              const barHeight = b.height * 95
              const x = 20 + i * barWidth
              const y = 110 - barHeight
              return (
                <rect
                  key={`in-${i}`}
                  x={x}
                  y={y}
                  width={barWidth - 1}
                  height={Math.max(1, barHeight)}
                  fill="#3b82f6"
                  fillOpacity={0.7}
                  rx={1}
                />
              )
            })}

            {/* Probe line */}
            {(() => {
              const maxVal = Math.max(...rawData, 100)
              const px = 20 + Math.min(1, probeVal / maxVal) * 265
              return (
                <g>
                  <line x1={px} y1="15" x2={px} y2="110" stroke="#f43f5e" strokeWidth="2" strokeDasharray="3 2" />
                  <circle cx={px} cy={110} r="3" fill="#f43f5e" />
                  <text x={px} y="12" textAnchor="middle" className="fill-rose-500 text-[8px] font-bold">
                    Probe: {probeVal}
                  </text>
                </g>
              )
            })()}

            <text x="20" y="122" className="fill-muted-foreground text-[8px]">
              0
            </text>
            <text x="285" y="122" textAnchor="end" className="fill-muted-foreground text-[8px]">
              {Math.max(...rawData, 100)}
            </text>
          </svg>
        </div>

        {/* Output Space */}
        <div className="rounded-lg border border-border bg-card/50 p-3">
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="font-semibold text-foreground">
              Quantile Transformed ($Z = \Phi^{'{'}-1{'}'}(F(X))$)
            </span>
            <span className="font-mono text-[11px] font-medium text-primary">
              {outputDist === 'normal' ? 'Gaussian N(0, 1)' : 'Uniform U(0, 1)'}
            </span>
          </div>

          <svg viewBox="0 0 300 130" className="h-auto w-full rounded bg-muted/10">
            <line x1="20" y1="110" x2="285" y2="110" stroke="currentColor" strokeOpacity={0.2} />

            {/* Output bars */}
            {outputBins.map((b, i) => {
              const barWidth = 265 / outputBins.length
              const barHeight = b.height * 95
              const x = 20 + i * barWidth
              const y = 110 - barHeight
              return (
                <rect
                  key={`out-${i}`}
                  x={x}
                  y={y}
                  width={barWidth - 1}
                  height={Math.max(1, barHeight)}
                  fill="#10b981"
                  fillOpacity={0.7}
                  rx={1}
                />
              )
            })}

            {/* Probe line in output space */}
            {(() => {
              const isUniform = outputDist === 'uniform'
              const targetVal = isUniform ? probeResult.u : probeResult.z
              const minVal = isUniform ? 0 : -3.5
              const maxVal = isUniform ? 1 : 3.5
              const fraction = Math.max(0, Math.min(1, (targetVal - minVal) / (maxVal - minVal)))
              const px = 20 + fraction * 265
              return (
                <g>
                  <line x1={px} y1="15" x2={px} y2="110" stroke="#f43f5e" strokeWidth="2" strokeDasharray="3 2" />
                  <circle cx={px} cy={110} r="3" fill="#f43f5e" />
                  <text x={px} y="12" textAnchor="middle" className="fill-rose-500 text-[8px] font-bold">
                    Z: {targetVal.toFixed(2)}
                  </text>
                </g>
              )
            })()}

            <text x="20" y="122" className="fill-muted-foreground text-[8px]">
              {outputDist === 'uniform' ? '0.0' : '-3.5'}
            </text>
            <text x="152" y="122" textAnchor="middle" className="fill-muted-foreground text-[8px]">
              {outputDist === 'uniform' ? '0.5' : '0.0'}
            </text>
            <text x="285" y="122" textAnchor="end" className="fill-muted-foreground text-[8px]">
              {outputDist === 'uniform' ? '1.0' : '+3.5'}
            </text>
          </svg>
        </div>
      </div>

      {/* Diagnostic Probe Card */}
      <div className="grid grid-cols-1 gap-3 rounded-lg border border-border bg-muted/30 p-3.5 text-xs sm:grid-cols-3">
        <div>
          <span className="block text-[11px] text-muted-foreground">Empirical Rank Quantile:</span>
          <div className="mt-0.5 font-mono text-base font-bold text-primary">
            {(probeResult.u * 100).toFixed(1)} percentile
          </div>
          <span className="text-[10px] text-muted-foreground">
            {probeVal >= rawData[rawData.length - 1]
              ? 'At or beyond training max (clipped to top boundary).'
              : probeVal <= rawData[0]
                ? 'At or below training min (clipped to bottom boundary).'
                : 'Interpolated smoothly across knot boundaries.'}
          </span>
        </div>

        <div>
          <span className="block text-[11px] text-muted-foreground">Gaussian Mapping ($Z$):</span>
          <div className="mt-0.5 font-mono text-base font-bold text-foreground">{probeResult.z.toFixed(3)} &sigma;</div>
          <span className="text-[10px] text-muted-foreground">
            Standard normal score ready for neural networks or linear models.
          </span>
        </div>

        <div>
          <span className="block text-[11px] text-muted-foreground">Metric Distortion Trade-off:</span>
          <div className="mt-0.5 text-base font-semibold text-amber-500">Non-linear Rank Warp</div>
          <p className="text-[10px] text-muted-foreground">
            Outliers lose metric distance. Relative ordering is preserved; absolute Euclidean distances are eliminated.
          </p>
        </div>
      </div>
    </div>
  )
}
