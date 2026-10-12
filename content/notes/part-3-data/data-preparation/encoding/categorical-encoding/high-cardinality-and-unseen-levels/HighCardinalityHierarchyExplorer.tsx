import React, { useState, useMemo } from 'react'

interface HierarchyItem {
  sku: string
  subCategory: string
  department: string
  trainCount: number
  skuMean: number
  subMean: number
  deptMean: number
}

export const HighCardinalityHierarchyExplorer: React.FC = () => {
  const [minCountThreshold, setMinCountThreshold] = useState<number>(15)
  const [selectedSku, setSelectedSku] = useState<string>('nike_air')
  const [topKFilter, setTopKFilter] = useState<number>(6)

  const globalPrior = 0.22 // 22% conversion baseline

  const catalog: HierarchyItem[] = useMemo(
    () => [
      {
        sku: 'nike_air',
        subCategory: 'Running Shoes',
        department: 'Footwear',
        trainCount: 420,
        skuMean: 0.38,
        subMean: 0.31,
        deptMean: 0.28,
      },
      {
        sku: 'adidas_boost',
        subCategory: 'Running Shoes',
        department: 'Footwear',
        trainCount: 310,
        skuMean: 0.34,
        subMean: 0.31,
        deptMean: 0.28,
      },
      {
        sku: 'salomon_xt6',
        subCategory: 'Trail Boots',
        department: 'Footwear',
        trainCount: 85,
        skuMean: 0.26,
        subMean: 0.24,
        deptMean: 0.28,
      },
      {
        sku: 'timberland_pro',
        subCategory: 'Work Boots',
        department: 'Footwear',
        trainCount: 160,
        skuMean: 0.21,
        subMean: 0.22,
        deptMean: 0.28,
      },
      {
        sku: 'iphone_15',
        subCategory: 'Smartphones',
        department: 'Electronics',
        trainCount: 650,
        skuMean: 0.15,
        subMean: 0.14,
        deptMean: 0.12,
      },
      {
        sku: 'pixel_8',
        subCategory: 'Smartphones',
        department: 'Electronics',
        trainCount: 220,
        skuMean: 0.13,
        subMean: 0.14,
        deptMean: 0.12,
      },
      {
        sku: 'rare_artisan_clog',
        subCategory: 'Clogs',
        department: 'Footwear',
        trainCount: 6,
        skuMean: 0.5,
        subMean: 0.25,
        deptMean: 0.28,
      },
      {
        sku: 'custom_e_scooter',
        subCategory: 'Scooters',
        department: 'Electronics',
        trainCount: 3,
        skuMean: 0.67,
        subMean: 0.08,
        deptMean: 0.12,
      },
      {
        sku: 'brand_new_unseen_sku',
        subCategory: 'Running Shoes',
        department: 'Footwear',
        trainCount: 0,
        skuMean: 0.0,
        subMean: 0.31,
        deptMean: 0.28,
      },
    ],
    [],
  )

  // Compute hierarchical fallback for the selected SKU
  const activeItem = catalog.find((c) => c.sku === selectedSku) || catalog[0]

  const resolution = useMemo(() => {
    const item = activeItem
    if (item.trainCount >= minCountThreshold) {
      return {
        levelUsed: 'Fine SKU (Level 3)',
        encodedValue: item.skuMean,
        explanation: `Sufficient observations (${item.trainCount} >= ${minCountThreshold}). Directly uses fine-grained SKU average.`,
        confidence: 'High Specificity',
        color: '#10b981',
      }
    }

    if (item.trainCount > 0) {
      return {
        levelUsed: 'Sub-Category Fallback (Level 2)',
        encodedValue: item.subMean,
        explanation: `Under-sampled rare SKU (${item.trainCount} < ${minCountThreshold}). Backs off to parent "${item.subCategory}" average.`,
        confidence: 'Aggregated Parent',
        color: '#f59e0b',
      }
    }

    // Unseen SKU: count = 0
    return {
      levelUsed: 'Parent Sub-Category (Unseen Handling)',
      encodedValue: item.subMean,
      explanation: `Zero training history (Unseen SKU). Seamlessly falls back to parent "${item.subCategory}" without runtime error!`,
      confidence: 'Taxonomy Fallback',
      color: '#3b82f6',
    }
  }, [activeItem, minCountThreshold])

  // Top-K pooling demonstration
  const topKPooled = useMemo(() => {
    const sorted = [...catalog].sort((a, b) => b.trainCount - a.trainCount)
    const top = sorted.slice(0, topKFilter)
    const pooled = sorted.slice(topKFilter)
    const pooledCount = pooled.reduce((s, it) => s + it.trainCount, 0)
    return {
      top,
      pooledCount,
      pooledItemsCount: pooled.length,
    }
  }, [catalog, topKFilter])

  return (
    <div className="not-prose my-6 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-col gap-1 border-b border-border pb-4">
        <h3 className="text-base font-semibold text-foreground">
          High-Cardinality Management: Hierarchical Fallbacks &amp; Top-K Pooling
        </h3>
        <p className="text-xs text-muted-foreground">
          When categorical variables encompass thousands of levels, one-hot encoding fails. Contrast Top-K rare-level
          pooling with domain taxonomy fallbacks for unseen and rare entities.
        </p>
      </div>

      {/* Control Bar */}
      <div className="mb-5 grid grid-cols-1 gap-4 text-xs md:grid-cols-3">
        <div>
          <label className="mb-1 block font-medium text-foreground">Select Test Entity / SKU:</label>
          <select
            value={selectedSku}
            onChange={(e) => setSelectedSku(e.target.value)}
            className="w-full rounded border border-border bg-muted/20 p-2 font-mono text-xs"
          >
            {catalog.map((c) => (
              <option key={c.sku} value={c.sku}>
                {c.sku} ({c.trainCount === 0 ? 'UNSEEN (0 rows)' : `${c.trainCount} rows`})
              </option>
            ))}
          </select>
          <span className="mt-1 block text-[10px] text-muted-foreground">
            Test how seen, rare, and unseen items resolve through the hierarchy.
          </span>
        </div>

        <div>
          <label className="mb-1 block font-medium text-foreground">
            Hierarchical Fallback Cutoff (<em>N</em>
            <sub>min</sub>): <span className="font-mono text-primary">{minCountThreshold} rows</span>
          </label>
          <input
            type="range"
            min={5}
            max={50}
            step={5}
            value={minCountThreshold}
            onChange={(e) => setMinCountThreshold(Number(e.target.value))}
            className="w-full accent-primary"
          />
          <span className="mt-1 block text-[10px] text-muted-foreground">
            Minimum instances required to trust fine SKU mean.
          </span>
        </div>

        <div>
          <label className="mb-1 block font-medium text-foreground">
            Top-K Retention Filter ($K$): <span className="font-mono text-primary">Top {topKFilter} items</span>
          </label>
          <input
            type="range"
            min={3}
            max={8}
            step={1}
            value={topKFilter}
            onChange={(e) => setTopKFilter(Number(e.target.value))}
            className="w-full accent-primary"
          />
          <span className="mt-1 block text-[10px] text-muted-foreground">
            Remaining {topKPooled.pooledItemsCount} rare items pooled into <code>__other__</code>.
          </span>
        </div>
      </div>

      {/* Hierarchical Tree Resolution Diagram */}
      <div className="mb-5 rounded-lg border border-border bg-card/60 p-4">
        <div className="mb-3 flex items-center justify-between text-xs">
          <span className="font-semibold text-foreground">Taxonomy Fallback Pathway for "{activeItem.sku}"</span>
          <span
            className="rounded px-2 py-0.5 text-[10px] font-bold"
            style={{ backgroundColor: `${resolution.color}20`, color: resolution.color }}
          >
            {resolution.levelUsed}
          </span>
        </div>

        {/* Tree Path Diagram */}
        <div className="mb-3 grid grid-cols-4 gap-2 text-center text-xs">
          <div className="rounded border border-border bg-muted/20 p-2.5">
            <span className="block text-[10px] text-muted-foreground">Global Prior</span>
            <div className="mt-0.5 font-mono font-bold text-foreground">{(globalPrior * 100).toFixed(0)}%</div>
            <span className="text-[9px] text-muted-foreground">Level 0</span>
          </div>

          <div className="rounded border border-border bg-muted/20 p-2.5">
            <span className="block text-[10px] text-muted-foreground">Department</span>
            <div className="mt-0.5 truncate font-bold text-foreground">{activeItem.department}</div>
            <span className="font-mono text-[10px] text-primary">{(activeItem.deptMean * 100).toFixed(0)}%</span>
          </div>

          <div
            className={`rounded border p-2.5 ${resolution.levelUsed.includes('Sub-Category') ? 'border-primary bg-primary/10 ring-1 ring-primary/40' : 'border-border bg-muted/20'}`}
          >
            <span className="block text-[10px] text-muted-foreground">Sub-Category</span>
            <div className="mt-0.5 truncate font-bold text-foreground">{activeItem.subCategory}</div>
            <span className="font-mono text-[10px] text-primary">{(activeItem.subMean * 100).toFixed(0)}%</span>
          </div>

          <div
            className={`rounded border p-2.5 ${resolution.levelUsed.includes('Fine SKU') ? 'border-emerald-500 bg-emerald-500/10 ring-1 ring-emerald-500/40' : 'border-border bg-muted/20 opacity-70'}`}
          >
            <span className="block text-[10px] text-muted-foreground">Fine SKU ({activeItem.trainCount} rows)</span>
            <div className="mt-0.5 truncate font-bold text-foreground">{activeItem.sku}</div>
            <span className="font-mono text-[10px] font-bold text-emerald-500">
              {activeItem.trainCount > 0 ? `${(activeItem.skuMean * 100).toFixed(0)}%` : '0 rows'}
            </span>
          </div>
        </div>

        <div className="rounded border border-border bg-muted/30 p-3 text-xs">
          <div className="mb-1 flex items-center justify-between">
            <span className="font-bold text-foreground">Encoded Feature Value:</span>
            <span className="font-mono text-base font-bold text-primary">{resolution.encodedValue.toFixed(3)}</span>
          </div>
          <p className="text-[11px] leading-normal text-muted-foreground">{resolution.explanation}</p>
        </div>
      </div>

      {/* Top-K Pooling vs Hierarchy Comparison */}
      <div className="grid grid-cols-1 gap-4 text-xs md:grid-cols-2">
        <div className="rounded-lg border border-border bg-muted/20 p-3.5">
          <span className="mb-1 block font-semibold text-foreground">1. Top-K Frequency Pooling</span>
          <p className="mb-2 text-[11px] leading-relaxed text-muted-foreground">
            Retains top {topKFilter} high-frequency levels as explicit features. All {topKPooled.pooledItemsCount}{' '}
            remaining tail items ({topKPooled.pooledCount.toLocaleString()} total rows) are collapsed into a single{' '}
            <code>__other__</code> bucket.
          </p>
          <span className="block text-[10px] font-medium text-primary">
            Best when long tail is unstructured or taxonomy metadata is unavailable.
          </span>
        </div>

        <div className="rounded-lg border border-border bg-muted/20 p-3.5">
          <span className="mb-1 block font-semibold text-foreground">2. Domain Hierarchy Backoff</span>
          <p className="mb-2 text-[11px] leading-relaxed text-muted-foreground">
            Leverages structured domain ontologies (ZIP &rarr; City &rarr; State, ICD-10 &rarr; Diagnosis Block). Rare
            and unseen items inherit their parent category properties rather than a generic blind bucket.
          </p>
          <span className="block text-[10px] font-medium text-emerald-500">
            Preserves rich semantic signal even on 100% brand-new entities!
          </span>
        </div>
      </div>
    </div>
  )
}
