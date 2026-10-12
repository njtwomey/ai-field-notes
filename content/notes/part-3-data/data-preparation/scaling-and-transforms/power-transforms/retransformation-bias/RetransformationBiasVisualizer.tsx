import React, { useState, useMemo } from 'react'

type CorrectionType = 'naive' | 'lognormal' | 'duan'

interface Point {
  x: number
  logY: number
  trueY: number
  predLog: number
  resid: number
  predNaive: number
  predLognormal: number
  predDuan: number
}

export const RetransformationBiasVisualizer: React.FC = () => {
  const [sigma, setSigma] = useState<number>(0.7)
  const [correction, setCorrection] = useState<CorrectionType>('naive')

  // Synthetic log-linear model: ln(Y) = 2.0 + 0.3 * X + eps, eps ~ N(0, sigma^2)
  const data = useMemo(() => {
    const points: Point[] = []
    const N = 80
    let seed = 42
    const rand = () => {
      seed = (seed * 16807) % 2147483647
      return seed / 2147483647
    }
    const randn = () => {
      const u1 = Math.max(1e-5, rand())
      const u2 = rand()
      return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2)
    }

    // Generate samples
    let sumExpResid = 0
    const temp: { x: number; logY: number; trueY: number; predLog: number; resid: number }[] = []

    for (let i = 0; i < N; i++) {
      const x = 1.0 + (i / (N - 1)) * 8.0
      const predLog = 2.0 + 0.25 * x
      const eps = randn() * sigma
      const logY = predLog + eps
      const trueY = Math.exp(logY)
      const resid = eps // in this clean setup
      sumExpResid += Math.exp(resid)
      temp.push({ x, logY, trueY, predLog, resid })
    }

    const smearingFactor = sumExpResid / N
    const lognormalFactor = Math.exp((sigma * sigma) / 2)

    temp.forEach((t) => {
      const predNaive = Math.exp(t.predLog)
      const predLognormal = predNaive * lognormalFactor
      const predDuan = predNaive * smearingFactor
      points.push({
        ...t,
        predNaive: Math.round(predNaive * 10) / 10,
        predLognormal: Math.round(predLognormal * 10) / 10,
        predDuan: Math.round(predDuan * 10) / 10,
      })
    })

    return {
      points,
      smearingFactor,
      lognormalFactor,
    }
  }, [sigma])

  // Aggregate sum comparisons
  const totals = useMemo(() => {
    let sumTrue = 0
    let sumNaive = 0
    let sumLognormal = 0
    let sumDuan = 0

    data.points.forEach((p) => {
      sumTrue += p.trueY
      sumNaive += p.predNaive
      sumLognormal += p.predLognormal
      sumDuan += p.predDuan
    })

    return {
      trueTotal: Math.round(sumTrue).toLocaleString(),
      naiveTotal: Math.round(sumNaive).toLocaleString(),
      lognormalTotal: Math.round(sumLognormal).toLocaleString(),
      duanTotal: Math.round(sumDuan).toLocaleString(),
      underestimatePct: Math.round((1 - sumNaive / sumTrue) * 100),
    }
  }, [data])

  // Curve points across x in [1, 9]
  const curves = useMemo(() => {
    const ptsNaive: string[] = []
    const ptsCorrected: string[] = []
    const steps = 30

    for (let i = 0; i <= steps; i++) {
      const xVal = 1.0 + (i / steps) * 8.0
      const predLog = 2.0 + 0.25 * xVal
      const naiveY = Math.exp(predLog)
      const factor =
        correction === 'naive' ? 1.0 : correction === 'lognormal' ? data.lognormalFactor : data.smearingFactor
      const activeY = naiveY * factor

      // Map to SVG: W=480, H=220, max Y approx 180
      const sx = 40 + ((xVal - 1) / 8) * 410
      const syNaive = 195 - (naiveY / 180) * 170
      const syActive = 195 - (activeY / 180) * 170

      ptsNaive.push(`${sx.toFixed(1)},${syNaive.toFixed(1)}`)
      ptsCorrected.push(`${sx.toFixed(1)},${syActive.toFixed(1)}`)
    }

    return {
      naive: ptsNaive.join(' '),
      active: ptsCorrected.join(' '),
    }
  }, [correction, data])

  return (
    <div className="not-prose my-6 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-col gap-1 border-b border-border pb-4">
        <h3 className="text-base font-semibold text-foreground">
          Retransformation Bias &amp; Smearing Correction Laboratory
        </h3>
        <p className="text-xs text-muted-foreground">
          Fitting models to transformed responses like $\ln(Y)$ estimates the conditional median upon exponentiation.
          Observe the Jensen inequality gap and restore the true conditional mean.
        </p>
      </div>

      {/* Control Bar */}
      <div className="mb-5 grid grid-cols-1 gap-4 text-xs md:grid-cols-3">
        <div>
          <label className="mb-1 block font-medium text-foreground">Retransformation Method:</label>
          <div className="flex flex-col gap-1.5">
            {[
              { id: 'naive', label: 'Naive: exp(y_hat) (Median)' },
              { id: 'lognormal', label: 'Log-Normal: exp(y_hat + &sigma;^2/2)' },
              { id: 'duan', label: "Duan's Non-Parametric Smearing (1983)" },
            ].map((opt) => (
              <button
                key={opt.id}
                onClick={() => setCorrection(opt.id as CorrectionType)}
                className={`rounded border px-2.5 py-1.5 text-left text-xs transition-colors ${
                  correction === opt.id
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
            Log Residual Spread (&sigma;): <span className="font-mono text-primary">{sigma.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min={0.2}
            max={1.1}
            step={0.05}
            value={sigma}
            onChange={(e) => setSigma(Number(e.target.value))}
            className="w-full accent-primary"
          />
          <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
            <span>Tight (&sigma;=0.2)</span>
            <span>Typical (&sigma;=0.7)</span>
            <span>Extreme (&sigma;=1.1)</span>
          </div>
        </div>

        <div className="flex flex-col justify-between rounded-lg border border-border bg-muted/30 p-3 text-xs">
          <div>
            <span className="mb-1 block font-semibold text-foreground">Theoretical Correction Factor</span>
            <div className="font-mono text-sm font-bold text-primary">
              {correction === 'naive'
                ? '1.000x (No adjustment)'
                : correction === 'lognormal'
                  ? `${data.lognormalFactor.toFixed(3)}x (exp(sigma^2/2))`
                  : `${data.smearingFactor.toFixed(3)}x (Sample E[exp(e)])`}
            </div>
          </div>
          <span className="mt-2 text-[10px] text-muted-foreground">
            Multiplied against naive predictions to recover unbiased original expectations.
          </span>
        </div>
      </div>

      {/* Main Scatter & Prediction Curves */}
      <div className="mb-5 rounded-lg border border-border bg-card/50 p-4">
        <div className="mb-2 flex items-center justify-between text-xs">
          <span className="font-semibold text-foreground">Original Scale ($Y = \exp(\ln Y)$)</span>
          <div className="flex items-center gap-3 text-[11px]">
            <span className="flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-blue-500"></span> Observations
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-0.5 w-4 border-t border-dashed bg-amber-500"></span> Naive Median:
              $\exp(X\hat{'{'}\beta{'}'})$
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-0.5 w-4 bg-emerald-500"></span> Corrected Mean: $E[Y \mid X]$
            </span>
          </div>
        </div>

        <svg viewBox="0 0 480 220" className="h-auto w-full rounded bg-muted/10">
          {/* Grid lines */}
          {[0, 50, 100, 150].map((val) => {
            const y = 195 - (val / 180) * 170
            return (
              <g key={`y-${val}`}>
                <line x1="40" y1={y} x2="465" y2={y} stroke="currentColor" strokeOpacity={0.1} />
                <text x="32" y={y + 3} textAnchor="end" className="fill-muted-foreground text-[9px]">
                  {val}
                </text>
              </g>
            )
          })}
          {[1, 3, 5, 7, 9].map((val) => {
            const x = 40 + ((val - 1) / 8) * 410
            return (
              <g key={`x-${val}`}>
                <line x1={x} y1="20" x2={x} y2="195" stroke="currentColor" strokeOpacity={0.1} />
                <text x={x} y="208" textAnchor="middle" className="fill-muted-foreground text-[9px]">
                  {val}
                </text>
              </g>
            )
          })}

          {/* Scatter points */}
          {data.points.map((p, i) => {
            const sx = 40 + ((p.x - 1) / 8) * 410
            const sy = 195 - (Math.min(180, p.trueY) / 180) * 170
            return <circle key={`p-${i}`} cx={sx} cy={sy} r={2.5} fill="#3b82f6" fillOpacity={0.65} />
          })}

          {/* Naive median curve (dashed amber) */}
          <polyline points={curves.naive} fill="none" stroke="#f59e0b" strokeWidth="2" strokeDasharray="4 3" />

          {/* Corrected active curve */}
          {correction !== 'naive' && <polyline points={curves.active} fill="none" stroke="#10b981" strokeWidth="2.5" />}
        </svg>
      </div>

      {/* Aggregate Impact Comparison */}
      <div className="rounded-lg border border-border bg-card/60 p-4">
        <h4 className="mb-2 text-xs font-semibold tracking-wider text-foreground uppercase">
          Aggregate Portfolio Sum Test ($\sum_{'{'}i=1{'}'}^N Y_i$)
        </h4>
        <div className="grid grid-cols-1 gap-3 text-xs sm:grid-cols-3">
          <div className="rounded-lg border border-border bg-muted/20 p-3">
            <span className="block text-[11px] text-muted-foreground">True Portfolio Total:</span>
            <div className="mt-0.5 font-mono text-lg font-bold text-foreground">{totals.trueTotal}</div>
            <span className="text-[10px] text-muted-foreground">Sum of actual observations.</span>
          </div>

          <div className="rounded-lg border border-rose-500/20 bg-rose-500/10 p-3">
            <span className="block text-[11px] font-semibold text-rose-600 dark:text-rose-400">
              Naive Retransformed Sum:
            </span>
            <div className="mt-0.5 font-mono text-lg font-bold text-rose-600 dark:text-rose-400">
              {totals.naiveTotal}
            </div>
            <span className="mt-0.5 block text-[10px] font-bold text-rose-500">
              Underestimates total by {totals.underestimatePct}%!
            </span>
          </div>

          <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/10 p-3">
            <span className="block text-[11px] font-semibold text-emerald-600 dark:text-emerald-400">
              Corrected Expectation Sum:
            </span>
            <div className="mt-0.5 font-mono text-lg font-bold text-emerald-600 dark:text-emerald-400">
              {correction === 'lognormal' ? totals.lognormalTotal : totals.duanTotal}
            </div>
            <span className="mt-0.5 block text-[10px] text-emerald-600 dark:text-emerald-400">
              Unbiased recovery of true portfolio cost.
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
