import React, { useState, useMemo } from 'react'

interface EntityItem {
  id: string
  name: string
  group: 'Electronics' | 'Apparel' | 'Groceries' | 'Rare'
  sampleCount: number
  rawCoords: [number, number] // 2D latent representation
}

export const CategoryEmbeddingExplorer: React.FC = () => {
  const [poolingThreshold, setPoolingThreshold] = useState<number>(10)
  const [testCardinality, setTestCardinality] = useState<number>(250)
  const [showClusters, setShowClusters] = useState<boolean>(true)

  // Entities with simulated 2D embedding coordinates learned by a neural network
  const baseEntities: EntityItem[] = useMemo(
    () => [
      { id: 'e1', name: 'Smartphones', group: 'Electronics', sampleCount: 1200, rawCoords: [1.8, 1.4] },
      { id: 'e2', name: 'Laptops', group: 'Electronics', sampleCount: 850, rawCoords: [1.6, 1.7] },
      { id: 'e3', name: 'Tablets', group: 'Electronics', sampleCount: 340, rawCoords: [1.4, 1.2] },
      { id: 'e4', name: 'Jeans', group: 'Apparel', sampleCount: 720, rawCoords: [-1.5, 1.3] },
      { id: 'e5', name: 'Jackets', group: 'Apparel', sampleCount: 450, rawCoords: [-1.7, 1.0] },
      { id: 'e6', name: 'Sneakers', group: 'Apparel', sampleCount: 610, rawCoords: [-1.3, 1.6] },
      { id: 'e7', name: 'Fresh Apples', group: 'Groceries', sampleCount: 950, rawCoords: [0.1, -1.8] },
      { id: 'e8', name: 'Whole Milk', group: 'Groceries', sampleCount: 1100, rawCoords: [-0.2, -1.6] },
      { id: 'e9', name: 'Artisan Cheeses', group: 'Groceries', sampleCount: 180, rawCoords: [0.3, -1.4] },
      // Rare tail items
      { id: 'e10', name: 'Vintage Gramophone', group: 'Rare', sampleCount: 7, rawCoords: [0.8, 0.4] },
      { id: 'e11', name: 'Handmade Quilt', group: 'Rare', sampleCount: 4, rawCoords: [-0.6, 0.5] },
      { id: 'e12', name: 'Exotic Spices', group: 'Rare', sampleCount: 2, rawCoords: [0.2, -0.4] },
    ],
    [],
  )

  // Compute pooled embeddings: items below poolingThreshold map to shared __rare__ token at (0, 0)
  const renderedEntities = useMemo(() => {
    return baseEntities.map((e) => {
      const isPooled = e.sampleCount < poolingThreshold
      return {
        ...e,
        isPooled,
        coords: isPooled ? ([0, 0] as [number, number]) : e.rawCoords,
      }
    })
  }, [baseEntities, poolingThreshold])

  // Dimension heuristics for testCardinality
  const dimHeuristics = useMemo(() => {
    const k = testCardinality
    const sqrtRule = Math.round(Math.sqrt(k))
    const fastaiRule = Math.min(600, Math.round(1.6 * Math.pow(k, 0.56)))
    const halfRule = Math.min(50, Math.max(2, Math.floor(k / 2)))
    return { sqrtRule, fastaiRule, halfRule }
  }, [testCardinality])

  // Map 2D latent coordinates [-2.5, 2.5] to SVG canvas [340 x 240]
  const toSvg = (coords: [number, number]) => {
    const [x, y] = coords
    const sx = 170 + (x / 2.5) * 140
    const sy = 120 - (y / 2.5) * 95
    return { sx, sy }
  }

  const colors = {
    Electronics: '#3b82f6',
    Apparel: '#8b5cf6',
    Groceries: '#10b981',
    Rare: '#f59e0b',
  }

  return (
    <div className="not-prose my-6 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-col gap-1 border-b border-border pb-4">
        <h3 className="text-base font-semibold text-foreground">
          Learned Category Embedding Space &amp; Rare-Level Pooling
        </h3>
        <p className="text-xs text-muted-foreground">
          Neural network embedding layers project discrete tokens into dense continuous vectors <em>W</em>
          <sub>embed</sub> &isin; R
          <sup>
            <em>K</em> &times; <em>d</em>
          </sup>
          . Semantic proximities emerge automatically via backpropagation.
        </p>
      </div>

      {/* Control Bar */}
      <div className="mb-5 grid grid-cols-1 gap-4 text-xs md:grid-cols-3">
        <div>
          <label className="mb-1 block font-medium text-foreground">
            Rare-Level Pooling Threshold (<em>N</em>
            <sub>min</sub>): <span className="font-mono text-primary">{poolingThreshold} rows</span>
          </label>
          <input
            type="range"
            min={1}
            max={25}
            step={1}
            value={poolingThreshold}
            onChange={(e) => setPoolingThreshold(Number(e.target.value))}
            className="w-full accent-primary"
          />
          <span className="mt-1 block text-[10px] text-muted-foreground">
            Categories with fewer rows collapse into shared <code>__rare__</code> token.
          </span>
        </div>

        <div>
          <label className="mb-1 block font-medium text-foreground">
            Cardinarity Dimension Tester ($K$):{' '}
            <span className="font-mono text-primary">{testCardinality.toLocaleString()} levels</span>
          </label>
          <input
            type="range"
            min={10}
            max={5000}
            step={50}
            value={testCardinality}
            onChange={(e) => setTestCardinality(Number(e.target.value))}
            className="w-full accent-primary"
          />
          <span className="mt-1 block text-[10px] text-muted-foreground">
            Tests standard heuristic dimension rules for neural tabular networks.
          </span>
        </div>

        <div className="flex flex-col justify-between">
          <label className="mb-1 block font-medium text-foreground">Display Overlay:</label>
          <button
            onClick={() => setShowClusters(!showClusters)}
            className={`rounded border px-3 py-1.5 text-xs font-medium transition-colors ${
              showClusters
                ? 'border-primary bg-primary text-primary-foreground'
                : 'border-border bg-muted/40 text-muted-foreground hover:bg-muted'
            }`}
          >
            {showClusters ? 'Semantic Hull Clusters: ON' : 'Semantic Hull Clusters: OFF'}
          </button>
          <span className="mt-1 text-[10px] text-muted-foreground">
            Toggles visual convex hulls around learned clusters.
          </span>
        </div>
      </div>

      {/* Main 2D Embedding Space Visualizer */}
      <div className="mb-5 grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-border bg-card/50 p-3 lg:col-span-2">
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="font-semibold text-foreground">2D t-SNE / PCA Projection of Learned Vector Space</span>
            <div className="flex items-center gap-2 text-[10px]">
              {Object.entries(colors).map(([grp, col]) => (
                <span key={grp} className="flex items-center gap-1">
                  <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: col }}></span>
                  {grp}
                </span>
              ))}
            </div>
          </div>

          <svg viewBox="0 0 340 240" className="h-auto w-full rounded bg-muted/10">
            {/* Origin crosshairs */}
            <line x1="170" y1="20" x2="170" y2="220" stroke="currentColor" strokeOpacity={0.15} />
            <line x1="30" y1="120" x2="310" y2="120" stroke="currentColor" strokeOpacity={0.15} />

            {/* Semantic Cluster Hulls */}
            {showClusters && (
              <g opacity={0.15}>
                {/* Electronics cluster */}
                <ellipse cx="250" cy="65" rx="45" ry="35" fill="#3b82f6" />
                {/* Apparel cluster */}
                <ellipse cx="85" cy="70" rx="45" ry="35" fill="#8b5cf6" />
                {/* Groceries cluster */}
                <ellipse cx="170" cy="180" rx="55" ry="30" fill="#10b981" />
              </g>
            )}

            {/* Pooled cluster origin marker */}
            <circle cx="170" cy="120" r="14" fill="#f59e0b" fillOpacity={0.15} stroke="#f59e0b" strokeDasharray="2 2" />
            <text x="170" y="142" textAnchor="middle" className="fill-amber-500 text-[8px] font-bold">
              Pooled [__rare__]
            </text>

            {/* Render Category Points */}
            {renderedEntities.map((e) => {
              const { sx, sy } = toSvg(e.coords)
              const color = e.isPooled ? '#f59e0b' : colors[e.group]

              return (
                <g key={e.id}>
                  <circle
                    cx={sx}
                    cy={sy}
                    r={e.isPooled ? 3 : 5}
                    fill={color}
                    fillOpacity={0.85}
                    stroke="#fff"
                    strokeWidth="1"
                  />
                  {!e.isPooled && (
                    <text x={sx + 6} y={sy + 3} className="fill-foreground text-[8px] font-medium">
                      {e.name}
                    </text>
                  )}
                </g>
              )
            })}
          </svg>
        </div>

        {/* Dimension Sizing Rules Card */}
        <div className="flex flex-col justify-between rounded-lg border border-border bg-card/50 p-4">
          <div>
            <div className="mb-2 text-xs font-bold tracking-wider text-foreground uppercase">
              Embedding Sizing Rules ($K = {testCardinality}$)
            </div>
            <p className="mb-3 text-[11px] leading-tight text-muted-foreground">
              Calculated dense vector width $d$ across standard industry heuristics:
            </p>

            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between rounded border border-border bg-muted/20 p-2.5">
                <div>
                  <span className="block font-semibold text-foreground">Fastai Heuristic</span>
                  <code className="font-mono text-[10px] text-muted-foreground">1.6 &middot; K^0.56</code>
                </div>
                <span className="font-mono text-base font-bold text-primary">{dimHeuristics.fastaiRule} dims</span>
              </div>

              <div className="flex items-center justify-between rounded border border-border bg-muted/20 p-2.5">
                <div>
                  <span className="block font-semibold text-foreground">Square Root Rule</span>
                  <code className="font-mono text-[10px] text-muted-foreground">&radic;K</code>
                </div>
                <span className="font-mono text-base font-bold text-foreground">{dimHeuristics.sqrtRule} dims</span>
              </div>

              <div className="flex items-center justify-between rounded border border-border bg-muted/20 p-2.5">
                <div>
                  <span className="block font-semibold text-foreground">Cap Rule (Max 50)</span>
                  <code className="font-mono text-[10px] text-muted-foreground">min(50, &lfloor;K/2&rfloor;)</code>
                </div>
                <span className="font-mono text-base font-bold text-foreground">{dimHeuristics.halfRule} dims</span>
              </div>
            </div>
          </div>

          <div className="mt-3 border-t border-border pt-2 text-[11px] text-muted-foreground">
            <strong>Memory Savings:</strong> Replaces a {testCardinality}-wide sparse one-hot matrix with a dense{' '}
            {dimHeuristics.fastaiRule}-wide vector.
          </div>
        </div>
      </div>
    </div>
  )
}
