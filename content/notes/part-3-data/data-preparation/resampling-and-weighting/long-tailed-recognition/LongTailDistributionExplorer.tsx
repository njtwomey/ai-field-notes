import React, { useState, useMemo } from 'react'

type MethodType = 'standard' | 'cui' | 'logit_adj' | 'decoupled'

export const LongTailDistributionExplorer: React.FC = () => {
  const [method, setMethod] = useState<MethodType>('standard')
  const [imbalanceRatio, setImbalanceRatio] = useState<number>(100)
  const [tauStrength, setTauStrength] = useState<number>(1.0)
  const [betaEffective, setBetaEffective] = useState<number>(0.999)

  // Generate 25 classes with exponential long-tailed decay
  // n_i = N_max * (1 / imbalanceRatio) ^ (i / (numClasses - 1))
  const numClasses = 25
  const maxSamples = 1000

  const classData = useMemo(() => {
    return Array.from({ length: numClasses }, (_, i) => {
      const fraction = i / (numClasses - 1)
      const count = Math.round(maxSamples * Math.pow(1 / imbalanceRatio, fraction))
      const group: 'many' | 'medium' | 'few' = count > 100 ? 'many' : count >= 20 ? 'medium' : 'few'
      return {
        classId: i + 1,
        count: Math.max(1, count),
        group,
      }
    })
  }, [imbalanceRatio, numClasses])

  const totalSamples = useMemo(() => {
    return classData.reduce((acc, c) => acc + c.count, 0)
  }, [classData])

  // Model weights / margin offsets per class under different methods
  const modelProfiles = useMemo(() => {
    return classData.map((c) => {
      const priorPi = c.count / totalSamples
      // Standard empirical classifier weight norm ~ log(n_i)
      const baseNorm = Math.log(c.count + 1) * 0.8

      let effectiveWeight = baseNorm
      let logitOffset = 0
      let predictedAcc = 0

      if (method === 'standard') {
        effectiveWeight = baseNorm
        logitOffset = 0
        // Standard CE has high many-shot, disastrous few-shot
        predictedAcc =
          c.group === 'many'
            ? 88 - c.classId * 1.5
            : c.group === 'medium'
              ? 62 - (c.classId - 6) * 2.5
              : Math.max(2, 28 - (c.classId - 14) * 2.2)
      } else if (method === 'cui') {
        // Cui et al. 2019: Class-balanced loss based on (1 - beta) / (1 - beta^n)
        const effectiveN = (1 - Math.pow(betaEffective, c.count)) / (1 - betaEffective)
        const weightLoss = 1 / Math.max(1, effectiveN)
        effectiveWeight = baseNorm * Math.pow(weightLoss * 100, 0.25)
        predictedAcc =
          c.group === 'many'
            ? 82 - c.classId * 1.2
            : c.group === 'medium'
              ? 68 - (c.classId - 6) * 1.8
              : Math.max(15, 48 - (c.classId - 14) * 1.9)
      } else if (method === 'logit_adj') {
        // Menon et al. 2021: Logit adjustment tau * log(pi_y)
        logitOffset = tauStrength * Math.log(priorPi)
        effectiveWeight = baseNorm // Feature backbone unchanged
        predictedAcc =
          c.group === 'many'
            ? Math.max(50, 85 - tauStrength * 12 - c.classId * 1.1)
            : c.group === 'medium'
              ? 70 + tauStrength * 4 - (c.classId - 6) * 1.4
              : Math.min(65, Math.max(10, 20 + tauStrength * 36 - (c.classId - 14) * 1.6))
      } else if (method === 'decoupled') {
        // Kang et al. 2020: tau-normalized weights W_i / ||W_i||^tau
        // Flattens classifier weight norms across classes
        const normFactor = Math.pow(baseNorm, 1 - tauStrength * 0.85)
        effectiveWeight = normFactor * 1.2
        predictedAcc =
          c.group === 'many'
            ? Math.max(60, 84 - tauStrength * 7 - c.classId * 1.1)
            : c.group === 'medium'
              ? 72 + tauStrength * 3 - (c.classId - 6) * 1.2
              : Math.min(68, Math.max(15, 25 + tauStrength * 34 - (c.classId - 14) * 1.4))
      }

      return {
        ...c,
        priorPi,
        effectiveWeight: Math.max(0.1, effectiveWeight),
        logitOffset,
        predictedAcc: Math.min(99, Math.max(1, Math.round(predictedAcc))),
      }
    })
  }, [classData, totalSamples, method, tauStrength, betaEffective])

  // Aggregate metrics across Many, Medium, Few shots
  const groupMetrics = useMemo(() => {
    const calcAvg = (grp: 'many' | 'medium' | 'few') => {
      const items = modelProfiles.filter((p) => p.group === grp)
      if (items.length === 0) return 0
      const sum = items.reduce((acc, curr) => acc + curr.predictedAcc, 0)
      return Math.round(sum / items.length)
    }

    const manyAvg = calcAvg('many')
    const medAvg = calcAvg('medium')
    const fewAvg = calcAvg('few')
    // Balanced overall accuracy (macro-average over all classes)
    const balancedAcc = Math.round(
      modelProfiles.reduce((acc, curr) => acc + curr.predictedAcc, 0) / modelProfiles.length,
    )

    return {
      many: manyAvg,
      medium: medAvg,
      few: fewAvg,
      balanced: balancedAcc,
      manyCount: modelProfiles.filter((p) => p.group === 'many').length,
      medCount: modelProfiles.filter((p) => p.group === 'medium').length,
      fewCount: modelProfiles.filter((p) => p.group === 'few').length,
    }
  }, [modelProfiles])

  const maxWeightNorm = Math.max(...modelProfiles.map((p) => p.effectiveWeight), 1)

  return (
    <div className="not-prose my-6 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-col gap-1 border-b border-border pb-4">
        <h3 className="text-base font-semibold text-foreground">Long-Tailed Recognition &amp; Classifier Decoupling</h3>
        <p className="text-xs text-muted-foreground">
          Explore class imbalance across Many-shot, Medium-shot, and Few-shot classes. Contrast standard Cross-Entropy
          with Class-Balanced Loss, Logit Adjustment, and Decoupled Head Calibration.
        </p>
      </div>

      {/* Control Grid */}
      <div className="mb-5 grid grid-cols-1 gap-4 text-xs md:grid-cols-4">
        <div>
          <label className="mb-1 block font-medium text-foreground">Paradigm:</label>
          <div className="flex flex-col gap-1.5">
            {[
              { id: 'standard', label: 'Standard Cross-Entropy' },
              { id: 'cui', label: 'Class-Balanced Loss (Cui 2019)' },
              { id: 'logit_adj', label: 'Logit Adjustment (Menon 2021)' },
              { id: 'decoupled', label: 'Decoupled Head &tau;-Norm (Kang 2020)' },
            ].map((opt) => (
              <button
                key={opt.id}
                onClick={() => setMethod(opt.id as MethodType)}
                className={`rounded border px-2.5 py-1.5 text-left text-xs transition-colors ${
                  method === opt.id
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
            Imbalance Ratio (&rho; = N_max / N_min): <span className="font-mono text-primary">{imbalanceRatio}x</span>
          </label>
          <input
            type="range"
            min={10}
            max={500}
            step={10}
            value={imbalanceRatio}
            onChange={(e) => setImbalanceRatio(Number(e.target.value))}
            className="w-full accent-primary"
          />
          <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
            <span>Moderate (10x)</span>
            <span>Heavy (100x)</span>
            <span>Extreme (500x)</span>
          </div>
        </div>

        {(method === 'logit_adj' || method === 'decoupled') && (
          <div>
            <label className="mb-1 block font-medium text-foreground">
              Adjustment Strength (&tau;): <span className="font-mono text-primary">{tauStrength.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0.2}
              max={1.8}
              step={0.1}
              value={tauStrength}
              onChange={(e) => setTauStrength(Number(e.target.value))}
              className="w-full accent-primary"
            />
            <span className="mt-1 block text-[11px] text-muted-foreground">
              {method === 'logit_adj'
                ? 'Additive shift margin &Delta;y = &tau; &middot; log(&pi;_y).'
                : 'Classifier weight re-scaling exponent W_y / ||W_y||^&tau;.'}
            </span>
          </div>
        )}

        {method === 'cui' && (
          <div>
            <label className="mb-1 block font-medium text-foreground">
              Effective Hyperparameter (&beta;):{' '}
              <span className="font-mono text-primary">{betaEffective.toFixed(4)}</span>
            </label>
            <input
              type="range"
              min={0.99}
              max={0.9999}
              step={0.0009}
              value={betaEffective}
              onChange={(e) => setBetaEffective(Number(e.target.value))}
              className="w-full accent-primary"
            />
            <span className="mt-1 block text-[11px] text-muted-foreground">
              Effective sample count E_n = (1 - &beta;^n) / (1 - &beta;).
            </span>
          </div>
        )}

        {method === 'standard' && (
          <div className="rounded border border-border bg-muted/30 p-2.5">
            <span className="mb-1 block font-semibold text-foreground">Standard Risk</span>
            <p className="text-[11px] leading-relaxed text-muted-foreground">
              Standard Empirical Risk Minimization optimizes average accuracy. Because head classes contain{' '}
              {Math.round(
                (classData.filter((c) => c.group === 'many').reduce((s, c) => s + c.count, 0) / totalSamples) * 100,
              )}
              % of rows, the tail is virtually ignored.
            </p>
          </div>
        )}
      </div>

      {/* Visualisation Grid */}
      <div className="mb-5 grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Panel 1: Class Frequency Distribution */}
        <div className="rounded-lg border border-border bg-card/50 p-3">
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="font-medium text-foreground">Long-Tailed Class Distribution ($N_i$)</span>
            <div className="flex items-center gap-2 text-[10px]">
              <span className="flex items-center gap-1">
                <span className="inline-block h-2.5 w-2.5 rounded-sm bg-blue-500"></span> Many-shot (&gt;100)
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block h-2.5 w-2.5 rounded-sm bg-amber-500"></span> Med (20-100)
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block h-2.5 w-2.5 rounded-sm bg-rose-500"></span> Few (&lt;20)
              </span>
            </div>
          </div>

          <svg viewBox="0 0 380 140" className="h-auto w-full rounded bg-muted/10">
            {/* Y axis guides */}
            {[0, 250, 500, 750, 1000].map((val) => {
              const y = 120 - (val / 1000) * 105
              return (
                <g key={`y-${val}`}>
                  <line x1="30" y1={y} x2="365" y2={y} stroke="currentColor" strokeOpacity={0.1} />
                  <text x="25" y={y + 3} textAnchor="end" className="fill-muted-foreground text-[8px]">
                    {val}
                  </text>
                </g>
              )
            })}

            {/* Bars */}
            {classData.map((c, i) => {
              const barWidth = 320 / numClasses
              const barHeight = (c.count / maxSamples) * 105
              const x = 35 + i * barWidth
              const y = 120 - barHeight
              const fillColor = c.group === 'many' ? '#3b82f6' : c.group === 'medium' ? '#f59e0b' : '#f43f5e'

              return (
                <g key={`bar-${c.classId}`}>
                  <rect
                    x={x}
                    y={y}
                    width={barWidth - 2}
                    height={Math.max(1, barHeight)}
                    fill={fillColor}
                    fillOpacity={0.8}
                    rx={1}
                  />
                </g>
              )
            })}

            {/* Threshold lines */}
            <line
              x1="35"
              y1={120 - (100 / 1000) * 105}
              x2="365"
              y2={120 - (100 / 1000) * 105}
              stroke="#3b82f6"
              strokeDasharray="2 2"
              strokeOpacity={0.5}
            />
            <line
              x1="35"
              y1={120 - (20 / 1000) * 105}
              x2="365"
              y2={120 - (20 / 1000) * 105}
              stroke="#f59e0b"
              strokeDasharray="2 2"
              strokeOpacity={0.5}
            />

            <text x="35" y="132" className="fill-muted-foreground text-[8px]">
              Class 1 (Head)
            </text>
            <text x="365" y="132" textAnchor="end" className="fill-muted-foreground text-[8px]">
              Class 25 (Tail)
            </text>
          </svg>
        </div>

        {/* Panel 2: Classifier Head Norm / Decision Boundary */}
        <div className="rounded-lg border border-border bg-card/50 p-3">
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="font-medium text-foreground">
              {method === 'logit_adj'
                ? 'Logit Adjustment Offset &tau; &middot; log(&pi;_y)'
                : 'Classifier Weight Vector Norm ||W_y||'}
            </span>
            <span className="text-[10px] text-muted-foreground">
              {method === 'decoupled' ? 'Equalized via &tau;-normalization' : 'Per-class capacity'}
            </span>
          </div>

          <svg viewBox="0 0 380 140" className="h-auto w-full rounded bg-muted/10">
            {/* Guide lines */}
            <line x1="35" y1="20" x2="365" y2="20" stroke="currentColor" strokeOpacity={0.1} />
            <line x1="35" y1="70" x2="365" y2="70" stroke="currentColor" strokeOpacity={0.1} />
            <line x1="35" y1="120" x2="365" y2="120" stroke="currentColor" strokeOpacity={0.2} />

            {modelProfiles.map((p, i) => {
              const barWidth = 320 / numClasses
              const x = 35 + i * barWidth

              if (method === 'logit_adj') {
                // Logit offsets are negative: e.g. -6 to 0
                const minOffset = -7 * tauStrength
                const normOffset = Math.abs(p.logitOffset) / Math.abs(minOffset || 1)
                const height = normOffset * 95
                const y = 120 - height
                return (
                  <rect
                    key={`norm-${p.classId}`}
                    x={x}
                    y={y}
                    width={barWidth - 2}
                    height={Math.max(2, height)}
                    fill="#8b5cf6"
                    fillOpacity={0.75}
                    rx={1}
                  />
                )
              }

              // Weight norms
              const barHeight = (p.effectiveWeight / maxWeightNorm) * 95
              const y = 120 - barHeight
              const color = p.group === 'many' ? '#3b82f6' : p.group === 'medium' ? '#f59e0b' : '#f43f5e'

              return (
                <rect
                  key={`norm-${p.classId}`}
                  x={x}
                  y={y}
                  width={barWidth - 2}
                  height={Math.max(2, barHeight)}
                  fill={color}
                  fillOpacity={0.8}
                  rx={1}
                />
              )
            })}

            <text x="35" y="132" className="fill-muted-foreground text-[8px]">
              Class 1
            </text>
            <text x="365" y="132" textAnchor="end" className="fill-muted-foreground text-[8px]">
              Class 25
            </text>
            <text x="30" y="25" textAnchor="end" className="fill-muted-foreground text-[8px]">
              High
            </text>
            <text x="30" y="118" textAnchor="end" className="fill-muted-foreground text-[8px]">
              0
            </text>
          </svg>
        </div>
      </div>

      {/* Accuracy Evaluation Breakdown */}
      <div className="rounded-lg border border-border bg-card/60 p-4">
        <h4 className="mb-2 text-xs font-semibold tracking-wider text-foreground uppercase">
          Balanced Evaluation Breakdown across Splits
        </h4>
        <div className="grid grid-cols-2 gap-3 text-center sm:grid-cols-4">
          <div className="rounded-lg border border-blue-500/20 bg-blue-500/10 p-2.5">
            <span className="block text-[11px] font-semibold text-blue-600 dark:text-blue-400">
              Many-shot Acc ({groupMetrics.manyCount} classes)
            </span>
            <div className="mt-0.5 font-mono text-xl font-bold text-foreground">{groupMetrics.many}%</div>
            <span className="text-[10px] text-muted-foreground">&gt;100 instances</span>
          </div>

          <div className="rounded-lg border border-amber-500/20 bg-amber-500/10 p-2.5">
            <span className="block text-[11px] font-semibold text-amber-600 dark:text-amber-400">
              Medium-shot Acc ({groupMetrics.medCount} classes)
            </span>
            <div className="mt-0.5 font-mono text-xl font-bold text-foreground">{groupMetrics.medium}%</div>
            <span className="text-[10px] text-muted-foreground">20 to 100 instances</span>
          </div>

          <div className="rounded-lg border border-rose-500/20 bg-rose-500/10 p-2.5">
            <span className="block text-[11px] font-semibold text-rose-600 dark:text-rose-400">
              Few-shot Acc ({groupMetrics.fewCount} classes)
            </span>
            <div className="mt-0.5 font-mono text-xl font-bold text-foreground">{groupMetrics.few}%</div>
            <span className="text-[10px] text-muted-foreground">&lt;20 instances</span>
          </div>

          <div className="rounded-lg border border-purple-500/20 bg-purple-500/10 p-2.5">
            <span className="block text-[11px] font-semibold text-purple-600 dark:text-purple-400">
              Balanced Macro Acc
            </span>
            <div className="mt-0.5 font-mono text-xl font-bold text-foreground">{groupMetrics.balanced}%</div>
            <span className="text-[10px] text-muted-foreground">Mean across all 25 classes</span>
          </div>
        </div>
      </div>
    </div>
  )
}
