import React, { useState, useMemo } from 'react'

type TransformType = 'simplex' | 'alr' | 'clr'

export const CompositionalSimplexExplorer: React.FC = () => {
  const [partA, setPartA] = useState<number>(45)
  const [partB, setPartB] = useState<number>(35)
  const [transformView, setTransformView] = useState<TransformType>('simplex')

  // Third component is uniquely determined on simplex: Part C = 100 - A - B
  const composition = useMemo(() => {
    const rawC = Math.max(1, 100 - partA - partB)
    const sum = partA + partB + rawC
    const a = partA / sum
    const b = partB / sum
    const c = rawC / sum
    const geomMean = Math.cbrt(a * b * c)

    return {
      a,
      b,
      c,
      pctA: Math.round(a * 100),
      pctB: Math.round(b * 100),
      pctC: Math.round(c * 100),
      // Additive log-ratio with C as denominator
      alr1: Math.log(a / c),
      alr2: Math.log(b / c),
      // Centered log-ratio
      clrA: Math.log(a / geomMean),
      clrB: Math.log(b / geomMean),
      clrC: Math.log(c / geomMean),
    }
  }, [partA, partB])

  // Ternary barycentric coordinates to 2D Cartesian (equilateral triangle)
  // Vertex A: top (160, 20), Vertex B: bottom-left (30, 220), Vertex C: bottom-right (290, 220)
  const toTernaryCartesian = (a: number, b: number, c: number) => {
    const xA = 160,
      yA = 25
    const xB = 30,
      yB = 220
    const xC = 290,
      yC = 220
    const px = a * xA + b * xB + c * xC
    const py = a * yA + b * yB + c * yC
    return { px, py }
  }

  const pointCartesian = useMemo(() => {
    return toTernaryCartesian(composition.a, composition.b, composition.c)
  }, [composition])

  return (
    <div className="not-prose my-6 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-col gap-1 border-b border-border pb-4">
        <h3 className="text-base font-semibold text-foreground">Compositional Data &amp; Aitchison Simplex Explorer</h3>
        <p className="text-xs text-muted-foreground">
          Proportions constrained to sum to 1 lie on the simplex <em>S</em>
          <sup>
            <em>D</em>
          </sup>
          . Log-ratio transforms (ALR and CLR) project constrained compositions onto unconstrained Euclidean space.
        </p>
      </div>

      {/* Control Bar */}
      <div className="mb-5 grid grid-cols-1 gap-4 text-xs md:grid-cols-3">
        <div>
          <label className="mb-1 block font-medium text-foreground">Projection Geometry:</label>
          <div className="flex flex-col gap-1.5">
            {[
              { id: 'simplex', label: '1. Ternary Simplex Triangle (S^3)' },
              { id: 'alr', label: '2. Additive Log-Ratio: alr(x) in R^2' },
              { id: 'clr', label: '3. Centered Log-Ratio: clr(x) in R^3' },
            ].map((opt) => (
              <button
                key={opt.id}
                onClick={() => setTransformView(opt.id as TransformType)}
                className={`rounded border px-2.5 py-1.5 text-left text-xs transition-colors ${
                  transformView === opt.id
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
            Component A Share: <span className="font-mono text-primary">{composition.pctA}%</span>
          </label>
          <input
            type="range"
            min={5}
            max={85}
            step={5}
            value={partA}
            onChange={(e) => setPartA(Number(e.target.value))}
            className="w-full accent-primary"
          />
          <span className="mt-1 block text-[10px] text-muted-foreground">
            Top vertex: e.g. Equity allocation, Sand fraction.
          </span>
        </div>

        <div>
          <label className="mb-1 block font-medium text-foreground">
            Component B Share: <span className="font-mono text-primary">{composition.pctB}%</span>
          </label>
          <input
            type="range"
            min={5}
            max={85}
            step={5}
            value={partB}
            onChange={(e) => setPartB(Number(e.target.value))}
            className="w-full accent-primary"
          />
          <span className="mt-1 block text-[10px] text-muted-foreground">
            Bottom-left vertex: e.g. Fixed income, Silt fraction.
          </span>
        </div>
      </div>

      {/* Visual Workspace */}
      <div className="mb-5 grid grid-cols-1 gap-4 md:grid-cols-2">
        {/* Simplex Ternary Geometry */}
        <div className="flex flex-col items-center rounded-lg border border-border bg-card/50 p-3">
          <div className="mb-2 flex w-full items-center justify-between text-xs">
            <span className="font-semibold text-foreground">Ternary Simplex Representation</span>
            <span className="font-mono text-[11px] text-muted-foreground">A + B + C = 100%</span>
          </div>

          <svg viewBox="0 0 320 240" className="h-auto w-full max-w-[300px] rounded bg-muted/10">
            {/* Equilateral Triangle */}
            <polygon
              points="160,25 30,220 290,220"
              fill="rgba(59, 130, 246, 0.08)"
              stroke="currentColor"
              strokeOpacity={0.4}
              strokeWidth="1.5"
            />

            {/* Grid altitude lines */}
            {[0.25, 0.5, 0.75].map((frac) => {
              const pA1 = toTernaryCartesian(frac, 1 - frac, 0)
              const pA2 = toTernaryCartesian(frac, 0, 1 - frac)
              return (
                <line
                  key={`grid-${frac}`}
                  x1={pA1.px}
                  y1={pA1.py}
                  x2={pA2.px}
                  y2={pA2.py}
                  stroke="currentColor"
                  strokeOpacity={0.15}
                  strokeDasharray="2 2"
                />
              )
            })}

            {/* Current point */}
            <circle cx={pointCartesian.px} cy={pointCartesian.py} r={6} fill="#f43f5e" stroke="#fff" strokeWidth="2" />

            {/* Vertex labels */}
            <text x="160" y="16" textAnchor="middle" className="fill-blue-500 text-[10px] font-bold">
              Part A ({composition.pctA}%)
            </text>
            <text x="25" y="235" textAnchor="start" className="fill-emerald-500 text-[10px] font-bold">
              Part B ({composition.pctB}%)
            </text>
            <text x="295" y="235" textAnchor="end" className="fill-amber-500 text-[10px] font-bold">
              Part C ({composition.pctC}%)
            </text>
          </svg>
        </div>

        {/* Transformed Euclidean Space Inspector */}
        <div className="flex flex-col justify-between rounded-lg border border-border bg-card/50 p-4">
          <div>
            <div className="mb-2 text-xs font-bold tracking-wider text-foreground uppercase">
              Transformed Unconstrained Coordinates
            </div>

            {transformView === 'alr' && (
              <div className="space-y-3 text-xs">
                <div className="rounded-lg border border-border bg-muted/20 p-3 font-mono">
                  <div className="flex justify-between border-b border-border/50 py-1">
                    <span className="font-sans text-primary">alr_1 = ln(A / C):</span>
                    <span className="font-bold text-foreground">{composition.alr1.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="font-sans text-primary">alr_2 = ln(B / C):</span>
                    <span className="font-bold text-foreground">{composition.alr2.toFixed(3)}</span>
                  </div>
                </div>
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  <strong>Additive Log-Ratio:</strong> Maps 3-part composition to 2 unconstrained real coordinates in R
                  <sup>2</sup>. Part C acts as common reference denominator.
                </p>
              </div>
            )}

            {transformView === 'clr' && (
              <div className="space-y-3 text-xs">
                <div className="rounded-lg border border-border bg-muted/20 p-3 font-mono">
                  <div className="flex justify-between border-b border-border/50 py-1">
                    <span className="font-sans text-primary">clr_A = ln(A / g(x)):</span>
                    <span className="font-bold text-foreground">{composition.clrA.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between border-b border-border/50 py-1">
                    <span className="font-sans text-primary">clr_B = ln(B / g(x)):</span>
                    <span className="font-bold text-foreground">{composition.clrB.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="font-sans text-primary">clr_C = ln(C / g(x)):</span>
                    <span className="font-bold text-foreground">{composition.clrC.toFixed(3)}</span>
                  </div>
                </div>
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  <strong>Centered Log-Ratio:</strong> Symmetric across all parts using geometric mean g(x). The sum of
                  CLR coordinates is identically zero: &sum; clr<sub>j</sub> = 0.
                </p>
              </div>
            )}

            {transformView === 'simplex' && (
              <div className="space-y-2 text-xs leading-relaxed text-muted-foreground">
                <div className="rounded-lg border border-amber-500/20 bg-amber-500/10 p-3 text-card-foreground">
                  <span className="mb-1 block font-semibold text-amber-600 dark:text-amber-400">
                    Karl Pearson's Spurious Correlation Trap (1897)
                  </span>
                  <p className="text-[11px] leading-relaxed">
                    Because parts must sum to a constant ($A + B + C = 1$), increasing $A$ mechanically drives down $B$
                    and $C$.
                  </p>
                  <p className="mt-1 text-[11px] leading-relaxed">
                    Standard Pearson correlation between raw percentages yields{' '}
                    <strong>artifactual negative correlation</strong>. Never run standard linear regression, PCA, or
                    covariance directly on raw proportions!
                  </p>
                </div>
              </div>
            )}
          </div>

          <div className="mt-3 flex items-center justify-between border-t border-border pt-3 text-[11px] text-muted-foreground">
            <span>Aitchison Distance Preserved:</span>
            <span className="font-mono font-bold text-primary">Isometric in Log-Ratios</span>
          </div>
        </div>
      </div>
    </div>
  )
}
