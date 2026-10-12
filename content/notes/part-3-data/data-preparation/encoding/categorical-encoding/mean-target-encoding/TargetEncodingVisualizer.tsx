import React, { useState, useMemo } from 'react'

type EncodingScheme = 'naive' | 'smoothed' | 'oof' | 'ordered'

interface CategoryRow {
  id: string
  name: string
  count: number
  positives: number
  rawMean: number
}

export const TargetEncodingVisualizer: React.FC = () => {
  const [scheme, setScheme] = useState<EncodingScheme>('smoothed')
  const [smoothingM, setSmoothingM] = useState<number>(10)

  // Global dataset statistics
  const globalTotal = 400
  const globalPositives = 80
  const globalPrior = globalPositives / globalTotal // 0.20 (20%)

  const categories: CategoryRow[] = useMemo(
    () => [
      { id: 'cat1', name: 'Metro Area (Large)', count: 240, positives: 72, rawMean: 0.3 },
      { id: 'cat2', name: 'Suburban (Medium)', count: 60, positives: 6, rawMean: 0.1 },
      { id: 'cat3', name: 'Small Town (Rare)', count: 8, positives: 4, rawMean: 0.5 },
      { id: 'cat4', name: 'Remote Village (Singleton)', count: 1, positives: 1, rawMean: 1.0 },
      { id: 'cat5', name: 'Offshore (Unseen in Train)', count: 0, positives: 0, rawMean: 0.0 },
    ],
    [],
  )

  // Compute encoded values
  const results = useMemo(() => {
    return categories.map((c) => {
      let encoded = globalPrior
      let leakageWarning = false
      let explanation = ''

      if (c.count === 0) {
        // Unseen level
        encoded = globalPrior
        explanation = 'Falls back exactly to global prior (0.200).'
      } else if (scheme === 'naive') {
        encoded = c.rawMean
        if (c.count === 1) {
          leakageWarning = true
          explanation = 'CRITICAL LEAKAGE: Encodes exactly 1.000 (its own target label)!'
        } else if (c.count <= 10) {
          explanation = 'High sample variance: small sample overfits local mean.'
        } else {
          explanation = 'Approaches true category conditional probability.'
        }
      } else if (scheme === 'smoothed') {
        // Empirical Bayes shrinkage: (n * rawMean + m * prior) / (n + m)
        const lambda = c.count / (c.count + smoothingM)
        encoded = lambda * c.rawMean + (1 - lambda) * globalPrior
        explanation = `Shrunk towards prior with weight ${(lambda * 100).toFixed(0)}% category, ${((1 - lambda) * 100).toFixed(0)}% prior.`
      } else if (scheme === 'oof') {
        // 5-fold OOF simulation: for instance i in fold k, target mean computed on remaining 4/5 data
        if (c.count === 1) {
          // Singleton: in 4 folds it has 0 observations -> encodes prior 0.20!
          encoded = globalPrior
          explanation =
            'OOF Shield: In validation fold, training folds contain 0 instances -> smoothly falls back to 0.200!'
        } else {
          const lambda = (c.count * 0.8) / (c.count * 0.8 + smoothingM)
          encoded = lambda * c.rawMean + (1 - lambda) * globalPrior
          explanation = 'Fitted strictly on out-of-fold partitions; test labels never bleed.'
        }
      } else if (scheme === 'ordered') {
        // CatBoost permutation scheme
        if (c.count === 1) {
          // First instance gets prior: (0 + m * prior) / (0 + m) = prior
          encoded = globalPrior
          explanation = 'CatBoost ordered: First appearance only sees historical prior -> perfectly safe from leakage.'
        } else {
          const lambda = c.count / 2 / (c.count / 2 + smoothingM)
          encoded = lambda * c.rawMean + (1 - lambda) * globalPrior
          explanation = 'Computed online along random row permutation. Past instances inform future rows.'
        }
      }

      return {
        ...c,
        encoded: Math.round(encoded * 1000) / 1000,
        leakageWarning,
        explanation,
      }
    })
  }, [categories, scheme, smoothingM, globalPrior])

  return (
    <div className="not-prose my-6 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-col gap-1 border-b border-border pb-4">
        <h3 className="text-base font-semibold text-foreground">
          Mean Target Encoding &amp; Empirical Bayes Shrinkage
        </h3>
        <p className="text-xs text-muted-foreground">
          Replace categorical levels with target expectations while mitigating target leakage through Bayesian
          shrinkage, Out-of-Fold (OOF) cross-validation, and CatBoost ordered permutations.
        </p>
      </div>

      {/* Control Grid */}
      <div className="mb-5 grid grid-cols-1 gap-4 text-xs md:grid-cols-3">
        <div>
          <label className="mb-1 block font-medium text-foreground">Encoding Scheme:</label>
          <div className="flex flex-col gap-1.5">
            {[
              { id: 'naive', label: '1. Naive In-Sample (Leakage Trap)' },
              { id: 'smoothed', label: '2. Empirical Bayes Smoothed' },
              { id: 'oof', label: '3. Out-of-Fold (5-Fold Cross-Val)' },
              { id: 'ordered', label: '4. CatBoost Ordered Permutation' },
            ].map((opt) => (
              <button
                key={opt.id}
                onClick={() => setScheme(opt.id as EncodingScheme)}
                className={`rounded border px-2.5 py-1.5 text-left text-xs transition-colors ${
                  scheme === opt.id
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
            Smoothing Weight ($m$): <span className="font-mono text-primary">{smoothingM}</span>
          </label>
          <input
            type="range"
            min={1}
            max={50}
            step={1}
            value={smoothingM}
            disabled={scheme === 'naive'}
            onChange={(e) => setSmoothingM(Number(e.target.value))}
            className="w-full accent-primary"
          />
          <span className="mt-1 block text-[10px] text-muted-foreground">
            Number of pseudo-observations required to balance category mean vs global prior.
          </span>
        </div>

        <div className="flex flex-col justify-between rounded-lg border border-border bg-muted/20 p-3 text-xs">
          <div>
            <span className="mb-1 block font-semibold text-foreground">Global Dataset Baseline</span>
            <div className="font-mono text-sm font-bold text-primary">
              Prior y&#772;<sub>global</sub> = 0.200 (20%)
            </div>
          </div>
          <span className="mt-2 text-[10px] text-muted-foreground">
            Base rate used for shrinkage anchor and unseen level fallback.
          </span>
        </div>
      </div>

      {/* Table Comparison */}
      <div className="mb-4 overflow-x-auto rounded-lg border border-border bg-card/50 p-4">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-border text-[10px] text-muted-foreground uppercase">
              <th className="px-2.5 py-2">Category Level</th>
              <th className="px-2.5 py-2 font-mono">Sample Size (n)</th>
              <th className="px-2.5 py-2 font-mono">Raw Target Mean</th>
              <th className="px-2.5 py-2 font-mono">Encoded Value S_c</th>
              <th className="px-2.5 py-2">Diagnostic Behavior</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r) => (
              <tr key={r.id} className={`border-b border-border/40 ${r.leakageWarning ? 'bg-rose-500/10' : ''}`}>
                <td className="px-2.5 py-2.5 font-medium text-foreground">{r.name}</td>
                <td className="px-2.5 py-2.5 font-mono text-muted-foreground">{r.count}</td>
                <td className="px-2.5 py-2.5 font-mono text-muted-foreground">
                  {r.count > 0 ? r.rawMean.toFixed(3) : 'N/A'}
                </td>
                <td className="px-2.5 py-2.5 font-mono font-bold text-primary">{r.encoded.toFixed(3)}</td>
                <td className="px-2.5 py-2.5 text-[11px] text-muted-foreground">
                  {r.leakageWarning ? (
                    <span className="font-bold text-rose-600 dark:text-rose-400">{r.explanation}</span>
                  ) : (
                    r.explanation
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Mathematical Foundation Box */}
      <div className="grid grid-cols-1 gap-3 rounded-lg border border-border bg-muted/30 p-3.5 text-xs md:grid-cols-2">
        <div>
          <span className="mb-1 block font-semibold text-foreground">Empirical Bayes Shrinkage Formula:</span>
          <code className="block rounded bg-muted/60 p-1.5 font-mono text-xs text-primary">
            S_c = &lambda;(n_c) &middot; y&#772;_c + (1 - &lambda;(n_c)) &middot; y&#772;_global, &nbsp; where
            &lambda;(n) = n / (n + m)
          </code>
        </div>
        <div>
          <span className="mb-1 block font-semibold text-foreground">Why Target Encoding Excels on Trees:</span>
          <p className="text-[11px] leading-normal text-muted-foreground">
            By ordering categorical levels monotonically by their conditional target rate, a single axis-aligned split
            $S_c \le \theta$ cleanly separates high-risk from low-risk entities without one-hot expansion.
          </p>
        </div>
      </div>
    </div>
  )
}
