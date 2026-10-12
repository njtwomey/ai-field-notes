import React, { useState, useMemo } from 'react'

type NormType = 'raw' | 'l2' | 'l1' | 'max'

interface VectorDoc {
  id: string
  name: string
  raw: [number, number]
  color: string
}

export const RowNormalizationVisualizer: React.FC = () => {
  const [normType, setNormType] = useState<NormType>('l2')
  const [doc1Coord, setDoc1Coord] = useState<[number, number]>([3, 2])
  const [doc2Coord, setDoc2Coord] = useState<[number, number]>([7.5, 5]) // Collinear with doc1 (longer text)
  const [doc3Coord] = useState<[number, number]>([1.5, 7]) // Different topic

  const docs: VectorDoc[] = useMemo(
    () => [
      { id: 'd1', name: 'Doc 1 (Brief Summary)', raw: doc1Coord, color: '#3b82f6' },
      { id: 'd2', name: 'Doc 2 (Full Article, Same Topic)', raw: doc2Coord, color: '#10b981' },
      { id: 'd3', name: 'Doc 3 (Different Topic)', raw: doc3Coord, color: '#f59e0b' },
    ],
    [doc1Coord, doc2Coord, doc3Coord],
  )

  // Compute norms and normalized coordinates
  const transformed = useMemo(() => {
    return docs.map((d) => {
      const [x, y] = d.raw
      const l2 = Math.sqrt(x * x + y * y)
      const l1 = Math.abs(x) + Math.abs(y)
      const lmax = Math.max(Math.abs(x), Math.abs(y))

      let normVal = 1
      let nx = x
      let ny = y

      if (normType === 'l2') {
        normVal = l2
        nx = l2 > 0 ? x / l2 : 0
        ny = l2 > 0 ? y / l2 : 0
      } else if (normType === 'l1') {
        normVal = l1
        nx = l1 > 0 ? x / l1 : 0
        ny = l1 > 0 ? y / l1 : 0
      } else if (normType === 'max') {
        normVal = lmax
        nx = lmax > 0 ? x / lmax : 0
        ny = lmax > 0 ? y / lmax : 0
      }

      return {
        ...d,
        normVal: Math.round(normVal * 100) / 100,
        nx: Math.round(nx * 1000) / 1000,
        ny: Math.round(ny * 1000) / 1000,
        l2Norm: l2,
      }
    })
  }, [docs, normType])

  // Pairwise Euclidean and Cosine distances between Doc 1 and Doc 2 / Doc 3
  const metrics = useMemo(() => {
    const v1 = transformed[0]
    const v2 = transformed[1]
    const v3 = transformed[2]

    const euclid = (a: { nx: number; ny: number }, b: { nx: number; ny: number }) => {
      return Math.sqrt(Math.pow(a.nx - b.nx, 2) + Math.pow(a.ny - b.ny, 2))
    }

    const cosine = (a: [number, number], b: [number, number]) => {
      const dot = a[0] * b[0] + a[1] * b[1]
      const normA = Math.sqrt(a[0] * a[0] + a[1] * a[1])
      const normB = Math.sqrt(b[0] * b[0] + b[1] * b[1])
      return normA * normB > 0 ? dot / (normA * normB) : 0
    }

    const d1_raw = docs[0].raw
    const d2_raw = docs[1].raw
    const d3_raw = docs[2].raw

    return {
      dist12: euclid(v1, v2).toFixed(3),
      dist13: euclid(v1, v3).toFixed(3),
      cos12: cosine(d1_raw, d2_raw).toFixed(3),
      cos13: cosine(d1_raw, d3_raw).toFixed(3),
    }
  }, [transformed, docs])

  // Map to SVG coordinates
  // Canvas width=380, height=300, origin at (50, 250)
  const toSvg = (x: number, y: number, isNormalized: boolean) => {
    const maxCoord = isNormalized ? 1.2 : 9.5
    const scale = 220 / maxCoord
    const sx = 50 + x * scale
    const sy = 250 - y * scale
    return { sx, sy }
  }

  const isNormActive = normType !== 'raw'

  return (
    <div className="not-prose my-6 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-col gap-1 border-b border-border pb-4">
        <h3 className="text-base font-semibold text-foreground">
          Row Normalisation &amp; Cosine Equivalence Visualizer
        </h3>
        <p className="text-xs text-muted-foreground">
          Row normalisation projects each observation to unit norm, transforming raw Euclidean distances into monotonic
          cosine similarities regardless of document length.
        </p>
      </div>

      {/* Control Buttons */}
      <div className="mb-5 grid grid-cols-1 gap-4 text-xs md:grid-cols-3">
        <div>
          <label className="mb-1 block font-medium text-foreground">Transformation Standard:</label>
          <div className="flex flex-col gap-1.5">
            {[
              { id: 'raw', label: 'Raw Vectors (Un-normalised)' },
              { id: 'l2', label: 'L2 Norm (Euclidean / Unit Sphere)' },
              { id: 'l1', label: 'L1 Norm (Manhattan / Simplex)' },
              { id: 'max', label: 'L-infinity Norm (Max Absolute)' },
            ].map((opt) => (
              <button
                key={opt.id}
                onClick={() => setNormType(opt.id as NormType)}
                className={`rounded border px-2.5 py-1.5 text-left text-xs transition-colors ${
                  normType === opt.id
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
            Doc 1 Term Weight: [{doc1Coord[0]}, {doc1Coord[1]}]
          </label>
          <div className="mt-2 space-y-2">
            <div>
              <span className="text-[10px] text-muted-foreground">Term X (Tech): {doc1Coord[0]}</span>
              <input
                type="range"
                min={1}
                max={6}
                step={0.5}
                value={doc1Coord[0]}
                onChange={(e) => setDoc1Coord([Number(e.target.value), doc1Coord[1]])}
                className="w-full accent-primary"
              />
            </div>
            <div>
              <span className="text-[10px] text-muted-foreground">Term Y (Finance): {doc1Coord[1]}</span>
              <input
                type="range"
                min={1}
                max={6}
                step={0.5}
                value={doc1Coord[1]}
                onChange={(e) => setDoc1Coord([doc1Coord[0], Number(e.target.value)])}
                className="w-full accent-primary"
              />
            </div>
          </div>
        </div>

        <div>
          <label className="mb-1 block font-medium text-foreground">
            Doc 2 Term Weight: [{doc2Coord[0]}, {doc2Coord[1]}]
          </label>
          <div className="mt-2 space-y-2">
            <div>
              <span className="text-[10px] text-muted-foreground">Term X: {doc2Coord[0]}</span>
              <input
                type="range"
                min={4}
                max={9}
                step={0.5}
                value={doc2Coord[0]}
                onChange={(e) => setDoc2Coord([Number(e.target.value), doc2Coord[1]])}
                className="w-full accent-primary"
              />
            </div>
            <div>
              <span className="text-[10px] text-muted-foreground">Term Y: {doc2Coord[1]}</span>
              <input
                type="range"
                min={3}
                max={9}
                step={0.5}
                value={doc2Coord[1]}
                onChange={(e) => setDoc2Coord([doc2Coord[0], Number(e.target.value)])}
                className="w-full accent-primary"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Visual Canvas and Geometric Analysis */}
      <div className="mb-5 grid grid-cols-1 gap-4 md:grid-cols-2">
        {/* Vector Canvas */}
        <div className="rounded-lg border border-border bg-card/50 p-3">
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="font-semibold text-foreground">
              {normType === 'raw'
                ? 'Raw Feature Space (Term Frequencies)'
                : normType === 'l2'
                  ? 'L2 Unit Circle Projection (||x||_2 = 1)'
                  : normType === 'l1'
                    ? 'L1 Simplex Projection (x1 + x2 = 1)'
                    : 'L-inf Box Projection'}
            </span>
          </div>

          <svg viewBox="0 0 320 280" className="h-auto w-full rounded bg-muted/10">
            {/* Axes */}
            <line x1="50" y1="250" x2="300" y2="250" stroke="currentColor" strokeOpacity={0.3} />
            <line x1="50" y1="250" x2="50" y2="20" stroke="currentColor" strokeOpacity={0.3} />
            <text x="300" y="262" textAnchor="end" className="fill-muted-foreground text-[9px]">
              Feature 1
            </text>
            <text x="42" y="25" textAnchor="end" className="fill-muted-foreground text-[9px]">
              Feature 2
            </text>

            {/* Boundary Geometry overlay */}
            {normType === 'l2' &&
              (() => {
                // Arc of radius 1: origin at (50, 250), radius = 220 / 1.2 = 183.33
                const r = 220 / 1.2
                return (
                  <path
                    d={`M 50,${250 - r} A ${r} ${r} 0 0 1 ${50 + r},250`}
                    fill="none"
                    stroke="#a855f7"
                    strokeWidth="1.5"
                    strokeDasharray="3 3"
                  />
                )
              })()}

            {normType === 'l1' &&
              (() => {
                // Line from (0, 1) to (1, 0)
                const scale = 220 / 1.2
                return (
                  <line
                    x1={50}
                    y1={250 - scale}
                    x2={50 + scale}
                    y2={250}
                    stroke="#a855f7"
                    strokeWidth="1.5"
                    strokeDasharray="3 3"
                  />
                )
              })()}

            {/* Render Vectors */}
            {transformed.map((d) => {
              const xVal = isNormActive ? d.nx : d.raw[0]
              const yVal = isNormActive ? d.ny : d.raw[1]
              const { sx, sy } = toSvg(xVal, yVal, isNormActive)

              return (
                <g key={d.id}>
                  {/* Ray from origin */}
                  <line x1="50" y1="250" x2={sx} y2={sy} stroke={d.color} strokeWidth="2.5" />
                  {/* Arrowhead endpoint */}
                  <circle cx={sx} cy={sy} r="4" fill={d.color} />
                  <text x={sx + 6} y={sy - 4} className="text-[9px] font-bold" fill={d.color}>
                    {d.name.split(' ')[0]} ({xVal.toFixed(2)}, {yVal.toFixed(2)})
                  </text>
                </g>
              )
            })}
          </svg>
        </div>

        {/* Diagnostic Comparison Matrix */}
        <div className="flex flex-col justify-between rounded-lg border border-border bg-card/50 p-4">
          <div>
            <div className="mb-3 text-xs font-bold tracking-wider text-foreground uppercase">
              Distance &amp; Similarity Diagnostics
            </div>

            <div className="space-y-3 text-xs">
              <div className="rounded-lg border border-border bg-muted/20 p-3">
                <div className="mb-1 flex items-center justify-between">
                  <span className="font-semibold text-foreground">Doc 1 vs Doc 2 (Same Topic, Long vs Short)</span>
                  <span className="font-mono text-[11px] font-bold text-emerald-500">Cosine = {metrics.cos12}</span>
                </div>
                <div className="flex justify-between text-[11px] text-muted-foreground">
                  <span>Euclidean Distance ||d1 - d2||:</span>
                  <span className="font-mono font-bold text-foreground">{metrics.dist12}</span>
                </div>
                <p className="mt-1 text-[10px] text-muted-foreground">
                  {normType === 'raw'
                    ? 'Misleadingly large Euclidean distance because Doc 2 is simply longer!'
                    : 'Normalized: Euclidean distance drops near 0, reflecting identical topic orientation.'}
                </p>
              </div>

              <div className="rounded-lg border border-border bg-muted/20 p-3">
                <div className="mb-1 flex items-center justify-between">
                  <span className="font-semibold text-foreground">Doc 1 vs Doc 3 (Different Topic)</span>
                  <span className="font-mono text-[11px] font-bold text-amber-500">Cosine = {metrics.cos13}</span>
                </div>
                <div className="flex justify-between text-[11px] text-muted-foreground">
                  <span>Euclidean Distance ||d1 - d3||:</span>
                  <span className="font-mono font-bold text-foreground">{metrics.dist13}</span>
                </div>
                <p className="mt-1 text-[10px] text-muted-foreground">
                  Under L2 normalization, Doc 1 is dramatically closer to Doc 2 than to Doc 3, correctly ordering
                  document relevance.
                </p>
              </div>
            </div>
          </div>

          <div className="mt-4 rounded border border-t border-border border-primary/20 bg-primary/5 p-2.5 pt-3 text-[11px]">
            <span className="mb-0.5 block font-semibold text-primary">Key Mathematical Identity</span>
            <code className="block font-mono text-foreground">||u - v||^2 = 2 - 2 &middot; cos(&theta;)</code>
            <span className="mt-1 block text-muted-foreground">
              For L2-normalised vectors, Euclidean distance and Cosine distance yield identical ranking for nearest
              neighbors.
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
