import React, { useState, useMemo } from 'react'

type FreqScale = 'count' | 'fraction' | 'log'

interface CategoryItem {
  id: string
  name: string
  count: number
  domain: string
  conversionRate: number
}

export const FrequencyEncodingVisualizer: React.FC = () => {
  const [scale, setScale] = useState<FreqScale>('fraction')
  const [splitThreshold, setSplitThreshold] = useState<number>(0.08)

  const rawCategories: CategoryItem[] = useMemo(
    () => [
      { id: 'c1', name: 'Smartphones', count: 1840, domain: 'Electronics', conversionRate: 0.12 },
      { id: 'c2', name: 'Laptops', count: 1250, domain: 'Electronics', conversionRate: 0.08 },
      { id: 'c3', name: 'T-Shirts', count: 960, domain: 'Apparel', conversionRate: 0.22 },
      { id: 'c4', name: 'Sneakers', count: 820, domain: 'Footwear', conversionRate: 0.18 },
      { id: 'c5', name: 'Headphones', count: 540, domain: 'Electronics', conversionRate: 0.14 },
      { id: 'c6', name: 'Coffee Beans', count: 320, domain: 'Grocery', conversionRate: 0.35 },
      { id: 'c7', name: 'Backpacks', count: 320, domain: 'Accessories', conversionRate: 0.09 }, // Collides with Coffee!
      { id: 'c8', name: 'Yoga Mats', count: 140, domain: 'Fitness', conversionRate: 0.16 },
      { id: 'c9', name: 'Board Games', count: 140, domain: 'Toys', conversionRate: 0.24 }, // Collides with Yoga!
      { id: 'c10', name: 'Drone Parts', count: 35, domain: 'Niche Tech', conversionRate: 0.03 },
      { id: 'c11', name: 'Vintage Watches', count: 35, domain: 'Luxury', conversionRate: 0.02 }, // Collides with Drones!
      { id: 'c12', name: 'Rare Antiques', count: 10, domain: 'Collectibles', conversionRate: 0.01 },
    ],
    [],
  )

  const totalN = useMemo(() => {
    return rawCategories.reduce((acc, c) => acc + c.count, 0)
  }, [rawCategories])

  // Compute encoded values and collisions
  const encodedData = useMemo(() => {
    // Detect collisions: duplicate count values
    const countFreqs: Record<number, number> = {}
    rawCategories.forEach((c) => {
      countFreqs[c.count] = (countFreqs[c.count] || 0) + 1
    })

    return rawCategories.map((c) => {
      const frac = c.count / totalN
      const logVal = Math.log(c.count + 1)
      const isCollision = (countFreqs[c.count] || 0) > 1

      let encodedVal = frac
      if (scale === 'count') encodedVal = c.count
      else if (scale === 'log') encodedVal = Math.round(logVal * 100) / 100

      return {
        ...c,
        fraction: Math.round(frac * 1000) / 1000,
        encodedVal: scale === 'fraction' ? Math.round(encodedVal * 1000) / 1000 : encodedVal,
        isCollision,
      }
    })
  }, [rawCategories, totalN, scale])

  // Split diagnostics
  const splitStats = useMemo(() => {
    const thresh = splitThreshold
    const left = encodedData.filter((c) => c.fraction < thresh)
    const right = encodedData.filter((c) => c.fraction >= thresh)

    const calcWeightedConv = (items: typeof encodedData) => {
      const totalCount = items.reduce((s, it) => s + it.count, 0)
      if (totalCount === 0) return 0
      const weightedSum = items.reduce((s, it) => s + it.count * it.conversionRate, 0)
      return Math.round((weightedSum / totalCount) * 1000) / 10
    }

    return {
      leftCount: left.length,
      leftConv: calcWeightedConv(left),
      rightCount: right.length,
      rightConv: calcWeightedConv(right),
    }
  }, [encodedData, splitThreshold])

  const maxVal = Math.max(...encodedData.map((c) => c.encodedVal))

  return (
    <div className="not-prose my-6 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-col gap-1 border-b border-border pb-4">
        <h3 className="text-base font-semibold text-foreground">Count &amp; Frequency Encoding Laboratory</h3>
        <p className="text-xs text-muted-foreground">
          Count encoding maps high-cardinality categories to a single 1D numerical frequency feature. Observe how tree
          splits segment rare vs head items, and identify category collision risks.
        </p>
      </div>

      {/* Control Bar */}
      <div className="mb-5 grid grid-cols-1 gap-4 text-xs md:grid-cols-3">
        <div>
          <label className="mb-1 block font-medium text-foreground">Encoding Format:</label>
          <div className="flex flex-col gap-1.5">
            {[
              { id: 'fraction', label: 'Normalized Frequency (N_c / N)' },
              { id: 'count', label: 'Raw Frequency Count (N_c)' },
              { id: 'log', label: 'Log Count: ln(N_c + 1)' },
            ].map((opt) => (
              <button
                key={opt.id}
                onClick={() => setScale(opt.id as FreqScale)}
                className={`rounded border px-2.5 py-1.5 text-left text-xs transition-colors ${
                  scale === opt.id
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
            Tree Split Threshold ($f_c \le \theta$): <span className="font-mono text-primary">{splitThreshold}</span>
          </label>
          <input
            type="range"
            min={0.01}
            max={0.25}
            step={0.01}
            value={splitThreshold}
            onChange={(e) => setSplitThreshold(Number(e.target.value))}
            className="w-full accent-primary"
          />
          <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
            <span>Rare tail only</span>
            <span>Split median</span>
            <span>Head classes</span>
          </div>
        </div>

        <div className="flex flex-col justify-between rounded-lg border border-border bg-muted/20 p-3 text-xs">
          <div>
            <span className="mb-1 block font-semibold text-foreground">Dimensionality Compression</span>
            <div className="font-mono text-sm font-bold text-primary">12 Levels &rarr; 1 Single Numeric Column</div>
          </div>
          <span className="mt-2 text-[10px] text-muted-foreground">
            Eliminates sparse matrices with zero additional memory overhead.
          </span>
        </div>
      </div>

      {/* Main Bar Chart */}
      <div className="mb-5 rounded-lg border border-border bg-card/50 p-4">
        <div className="mb-3 flex items-center justify-between text-xs">
          <span className="font-semibold text-foreground">Encoded Feature Space</span>
          <div className="flex items-center gap-3 text-[11px]">
            <span className="flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-sm bg-blue-500"></span> Normal Level
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-sm bg-rose-500"></span> Collision Hazard (Tied Counts)
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-0.5 w-4 border-t border-dashed bg-amber-500"></span> Split Cutoff
              (&theta;)
            </span>
          </div>
        </div>

        <svg viewBox="0 0 520 200" className="h-auto w-full rounded bg-muted/10">
          {/* Guide grid */}
          <line x1="40" y1="170" x2="500" y2="170" stroke="currentColor" strokeOpacity={0.2} />

          {/* Render horizontal-ish bars for each category */}
          {encodedData.map((c, i) => {
            const barWidth = 440 / encodedData.length
            const barHeight = (c.encodedVal / maxVal) * 140
            const x = 50 + i * barWidth
            const y = 170 - barHeight

            return (
              <g key={c.id}>
                <rect
                  x={x}
                  y={y}
                  width={barWidth - 4}
                  height={Math.max(2, barHeight)}
                  fill={c.isCollision ? '#f43f5e' : '#3b82f6'}
                  fillOpacity={0.8}
                  rx={1}
                />
                <text
                  x={x + (barWidth - 4) / 2}
                  y={182}
                  textAnchor="end"
                  transform={`rotate(-45, ${x + (barWidth - 4) / 2}, 182)`}
                  className="fill-muted-foreground text-[8px] font-medium"
                >
                  {c.name}
                </text>
              </g>
            )
          })}

          {/* Split threshold line */}
          {scale === 'fraction' &&
            (() => {
              const yPos = 170 - (splitThreshold / maxVal) * 140
              return (
                <g>
                  <line x1="45" y1={yPos} x2="500" y2={yPos} stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="4 3" />
                  <text x="495" y={yPos - 4} textAnchor="end" className="fill-amber-500 text-[9px] font-bold">
                    Cutoff &theta; = {splitThreshold}
                  </text>
                </g>
              )
            })()}
        </svg>
      </div>

      {/* Collision Diagnostic & Split Partition Impact */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {/* Left: Collision Warnings */}
        <div className="rounded-lg border border-border bg-card/40 p-4">
          <div className="mb-2 text-xs font-bold tracking-wider text-foreground uppercase">
            Identified Collision Hazards
          </div>
          <p className="mb-3 text-[11px] leading-tight text-muted-foreground">
            When distinct concepts share identical frequency, count encoding projects them to the exact same numerical
            scalar:
          </p>

          <div className="space-y-2 text-xs">
            <div className="rounded border border-rose-500/20 bg-rose-500/10 p-2.5">
              <div className="flex justify-between font-mono">
                <span className="font-bold text-foreground">"Coffee Beans" &amp; "Backpacks"</span>
                <span className="font-bold text-rose-500">Both Count = 320</span>
              </div>
              <p className="mt-1 text-[10px] text-muted-foreground">
                Coffee has 35% conversion while Backpacks has 9%. Any linear or tree model will treat them as identical!
              </p>
            </div>

            <div className="rounded border border-rose-500/20 bg-rose-500/10 p-2.5">
              <div className="flex justify-between font-mono">
                <span className="font-bold text-foreground">"Drone Parts" &amp; "Vintage Watches"</span>
                <span className="font-bold text-rose-500">Both Count = 35</span>
              </div>
              <p className="mt-1 text-[10px] text-muted-foreground">
                Completely distinct semantic customer intents collapsed into identical numerical coordinate.
              </p>
            </div>
          </div>
        </div>

        {/* Right: Decision Tree Split Partition */}
        <div className="flex flex-col justify-between rounded-lg border border-border bg-card/40 p-4">
          <div>
            <div className="mb-2 text-xs font-bold tracking-wider text-foreground uppercase">
              Decision Tree Partition Behavior
            </div>
            <p className="mb-3 text-[11px] leading-tight text-muted-foreground">
              A single split on frequency isolates niche/rare entities from high-volume head categories:
            </p>

            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="rounded border border-border bg-muted/20 p-2.5">
                <span className="block text-[10px] font-semibold text-muted-foreground">
                  Rare Branch ($f &lt; {splitThreshold}$)
                </span>
                <div className="mt-0.5 text-sm font-bold text-foreground">{splitStats.leftCount} categories</div>
                <span className="mt-1 block font-mono text-[10px] text-primary">Avg Conv: {splitStats.leftConv}%</span>
              </div>

              <div className="rounded border border-border bg-muted/20 p-2.5">
                <span className="block text-[10px] font-semibold text-muted-foreground">
                  Head Branch ($f \ge {splitThreshold}$)
                </span>
                <div className="mt-0.5 text-sm font-bold text-foreground">{splitStats.rightCount} categories</div>
                <span className="mt-1 block font-mono text-[10px] text-primary">Avg Conv: {splitStats.rightConv}%</span>
              </div>
            </div>
          </div>

          <div className="mt-3 border-t border-border pt-2 text-[11px] text-muted-foreground">
            <strong>Remedy for collisions:</strong> Pair frequency encoding with Target Encoding or CatBoost
            permutations.
          </div>
        </div>
      </div>
    </div>
  )
}
