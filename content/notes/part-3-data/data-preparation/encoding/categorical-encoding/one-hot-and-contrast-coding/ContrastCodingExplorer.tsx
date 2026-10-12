import React, { useState, useMemo } from 'react'

type CodingScheme = 'treatment' | 'sum' | 'helmert' | 'full_ohe'

export const ContrastCodingExplorer: React.FC = () => {
  const [scheme, setScheme] = useState<CodingScheme>('treatment')
  const [refLevel, setRefLevel] = useState<number>(0) // 0-indexed: Level 1 = Spring

  // 4 categories with true response means:
  // Spring: 40, Summer: 75, Autumn: 55, Winter: 30
  const levels = useMemo(
    () => [
      { name: 'Spring (L1)', mean: 40, color: '#10b981' },
      { name: 'Summer (L2)', mean: 75, color: '#f59e0b' },
      { name: 'Autumn (L3)', mean: 55, color: '#8b5cf6' },
      { name: 'Winter (L4)', mean: 30, color: '#3b82f6' },
    ],
    [],
  )

  const grandMean = useMemo(() => {
    return levels.reduce((s, l) => s + l.mean, 0) / levels.length
  }, [levels])

  // Coding matrix definitions for K=4
  const codingMatrix = useMemo(() => {
    if (scheme === 'treatment') {
      // Treatment with refLevel as reference (all 0s)
      return levels.map((_, i) => {
        if (i === refLevel) return [0, 0, 0]
        const row = [0, 0, 0]
        let colIdx = 0
        for (let j = 0; j < 4; j++) {
          if (j !== refLevel) {
            if (j === i) row[colIdx] = 1
            colIdx++
          }
        }
        return row
      })
    }

    if (scheme === 'sum') {
      // Sum/deviation coding: last level is -1 across all columns
      return [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, -1, -1],
      ]
    }

    if (scheme === 'helmert') {
      // Helmert coding (orthogonal)
      return [
        [-1, -1, -1],
        [1, -1, -1],
        [0, 2, -1],
        [0, 0, 3],
      ]
    }

    // full_ohe: 4 columns
    return [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ]
  }, [scheme, refLevel, levels])

  // Coefficient interpretations
  const interpretations = useMemo(() => {
    if (scheme === 'treatment') {
      const refMean = levels[refLevel].mean
      const nonRefs = levels.filter((_, idx) => idx !== refLevel)
      return {
        beta0: { name: '&beta;0 (Intercept)', value: refMean, formula: `Mean of ${levels[refLevel].name}` },
        slopes: nonRefs.map((l) => ({
          name: `&beta; (${l.name} vs Ref)`,
          value: l.mean - refMean,
          formula: `${l.mean} - ${refMean} = ${l.mean - refMean > 0 ? '+' : ''}${l.mean - refMean}`,
        })),
        isSingular: false,
      }
    }

    if (scheme === 'sum') {
      return {
        beta0: { name: '&beta;0 (Intercept)', value: grandMean, formula: 'Grand Mean (mean of all 4 groups)' },
        slopes: [
          {
            name: '&beta;1 (Spring Dev)',
            value: levels[0].mean - grandMean,
            formula: `${levels[0].mean} - ${grandMean}`,
          },
          {
            name: '&beta;2 (Summer Dev)',
            value: levels[1].mean - grandMean,
            formula: `${levels[1].mean} - ${grandMean}`,
          },
          {
            name: '&beta;3 (Autumn Dev)',
            value: levels[2].mean - grandMean,
            formula: `${levels[2].mean} - ${grandMean}`,
          },
        ],
        isSingular: false,
      }
    }

    if (scheme === 'helmert') {
      return {
        beta0: { name: '&beta;0 (Intercept)', value: grandMean, formula: 'Grand Mean' },
        slopes: [
          {
            name: '&beta;1 (L2 vs L1)',
            value: (levels[1].mean - levels[0].mean) / 2,
            formula: '(Summer - Spring) / 2',
          },
          {
            name: '&beta;2 (L3 vs L1+L2)',
            value: (2 * levels[2].mean - levels[0].mean - levels[1].mean) / 6,
            formula: '(2*Autumn - Spring - Summer) / 6',
          },
          {
            name: '&beta;3 (L4 vs Prior)',
            value: (3 * levels[3].mean - levels[0].mean - levels[1].mean - levels[2].mean) / 12,
            formula: '(3*Winter - Prior Sum) / 12',
          },
        ],
        isSingular: false,
      }
    }

    return {
      beta0: { name: '&beta;0 (Intercept)', value: 0, formula: 'Undefined (Matrix Singular!)' },
      slopes: levels.map((l) => ({
        name: `&beta; (${l.name})`,
        value: 0,
        formula: 'Infinitely many solutions (X^T X is non-invertible)',
      })),
      isSingular: true,
    }
  }, [scheme, refLevel, levels, grandMean])

  return (
    <div className="not-prose my-6 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-col gap-1 border-b border-border pb-4">
        <h3 className="text-base font-semibold text-foreground">One-Hot &amp; Contrast Coding Laboratory</h3>
        <p className="text-xs text-muted-foreground">
          Explore how Treatment, Sum, and Helmert contrasts parameterize discrete factor levels into design matrices,
          resolving the dummy variable trap and controlling coefficient semantics.
        </p>
      </div>

      {/* Scheme Selector */}
      <div className="mb-5 grid grid-cols-1 gap-4 text-xs md:grid-cols-4">
        <div className="md:col-span-2">
          <label className="mb-1 block font-medium text-foreground">Contrast Coding Scheme:</label>
          <div className="grid grid-cols-2 gap-1.5">
            {[
              { id: 'treatment', label: 'Treatment (Dummy / Reference)' },
              { id: 'sum', label: 'Sum (Deviation / Grand Mean)' },
              { id: 'helmert', label: 'Helmert (Orthogonal Incremental)' },
              { id: 'full_ohe', label: 'Full One-Hot (Trap Test)' },
            ].map((opt) => (
              <button
                key={opt.id}
                onClick={() => setScheme(opt.id as CodingScheme)}
                className={`rounded border p-2 text-left text-xs transition-colors ${
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

        {scheme === 'treatment' && (
          <div>
            <label className="mb-1 block font-medium text-foreground">Reference Category:</label>
            <div className="flex flex-col gap-1">
              {levels.map((l, i) => (
                <button
                  key={l.name}
                  onClick={() => setRefLevel(i)}
                  className={`rounded border px-2 py-1 text-left text-[11px] ${
                    refLevel === i
                      ? 'bg-foreground font-semibold text-background'
                      : 'border-border bg-muted/30 text-muted-foreground'
                  }`}
                >
                  {l.name} ({l.mean} pts)
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="flex flex-col justify-between rounded-lg border border-border bg-muted/20 p-3 text-xs">
          <div>
            <span className="mb-1 block font-semibold text-foreground">Collinearity Status</span>
            <span
              className={`inline-block rounded px-2 py-0.5 text-[10px] font-bold ${
                interpretations.isSingular
                  ? 'bg-rose-500/20 text-rose-600 dark:text-rose-400'
                  : 'bg-emerald-500/20 text-emerald-600 dark:text-emerald-400'
              }`}
            >
              {interpretations.isSingular ? 'Dummy Variable Trap (Rank 4 < 5)' : 'Full Rank (Invertible)'}
            </span>
          </div>
          <span className="mt-2 text-[10px] text-muted-foreground">
            {interpretations.isSingular
              ? 'Columns sum to 1.0, creating exact linear dependence with the intercept column.'
              : 'K-1 coded columns permit unique Ordinary Least Squares solution.'}
          </span>
        </div>
      </div>

      {/* Design Matrix Table */}
      <div className="mb-5 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="rounded-lg border border-border bg-card/50 p-3">
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="font-semibold text-foreground">Design Coding Matrix ($C$)</span>
            <span className="font-mono text-[10px] text-muted-foreground">
              Dimension: 4 &times; {codingMatrix[0].length}
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-border text-[10px] text-muted-foreground uppercase">
                  <th className="px-2 py-1.5">Level</th>
                  <th className="px-2 py-1.5">Obs Mean</th>
                  {codingMatrix[0].map((_, cIdx) => (
                    <th key={`col-${cIdx}`} className="px-2 py-1.5 font-mono">
                      x_{cIdx + 1}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {levels.map((l, rIdx) => {
                  const isRef = scheme === 'treatment' && rIdx === refLevel
                  return (
                    <tr
                      key={l.name}
                      className={`border-b border-border/40 font-mono text-[11px] ${
                        isRef ? 'bg-muted/40 font-bold' : ''
                      }`}
                    >
                      <td className="px-2 py-2 font-sans font-medium text-foreground">
                        <span
                          className="mr-1.5 inline-block h-2 w-2 rounded-full"
                          style={{ backgroundColor: l.color }}
                        ></span>
                        {l.name} {isRef && <span className="text-[9px] text-primary">(Ref)</span>}
                      </td>
                      <td className="px-2 py-2 text-foreground">{l.mean}</td>
                      {codingMatrix[rIdx].map((val, cIdx) => (
                        <td
                          key={`c-${cIdx}`}
                          className={`px-2 py-2 ${val !== 0 ? 'font-bold text-primary' : 'text-muted-foreground/60'}`}
                        >
                          {val > 0 ? `+${val}` : val}
                        </td>
                      ))}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Coefficient Interpretation Card */}
        <div className="flex flex-col justify-between rounded-lg border border-border bg-card/50 p-4">
          <div>
            <div className="mb-2 text-xs font-bold tracking-wider text-foreground uppercase">
              Resulting OLS Parameter Semantics
            </div>

            {interpretations.isSingular ? (
              <div className="space-y-2 rounded border border-rose-500/20 bg-rose-500/10 p-3 text-xs text-card-foreground">
                <span className="block font-bold text-rose-600 dark:text-rose-400">
                  Singular Matrix: Determinant = 0
                </span>
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  With 4 one-hot columns and 1 intercept column, the columns are linearly dependent:
                </p>
                <code className="block rounded bg-muted/60 p-1.5 font-mono text-xs">
                  x1 + x2 + x3 + x4 = 1 &times; Intercept
                </code>
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  OLS cannot invert X^T X. In regularized models (L2 Ridge), the penalty &lambda;I makes it invertible,
                  but unregularized models require dropping one level.
                </p>
              </div>
            ) : (
              <div className="space-y-2 text-xs">
                <div className="flex items-center justify-between rounded border border-border bg-muted/30 p-2.5">
                  <div>
                    <span
                      className="block font-bold text-foreground"
                      dangerouslySetInnerHTML={{ __html: interpretations.beta0.name }}
                    ></span>
                    <span className="text-[10px] text-muted-foreground">{interpretations.beta0.formula}</span>
                  </div>
                  <span className="font-mono text-sm font-bold text-primary">
                    {interpretations.beta0.value.toFixed(1)}
                  </span>
                </div>

                {interpretations.slopes.map((s, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between rounded border border-border/70 bg-muted/15 p-2"
                  >
                    <div>
                      <span
                        className="block text-[11px] font-medium text-foreground"
                        dangerouslySetInnerHTML={{ __html: s.name }}
                      ></span>
                      <span className="text-[10px] text-muted-foreground">{s.formula}</span>
                    </div>
                    <span className="font-mono text-xs font-bold text-foreground">
                      {s.value > 0 ? `+${s.value.toFixed(1)}` : s.value.toFixed(1)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="mt-3 border-t border-border pt-2 text-[10px] text-muted-foreground">
            Grand Sample Mean across all groups:{' '}
            <span className="font-mono font-bold text-foreground">{grandMean}</span>
          </div>
        </div>
      </div>
    </div>
  )
}
