import { useMemo } from 'react'
import { choice, Figure, Readout, useFigureState } from 'aifn-render'

interface RawItem {
  rawCategory: string
  rawPrice: string
  rawWeight: string
  rawDate: string
}

const RAW_INVENTORY: RawItem[] = [
  { rawCategory: 'Laptop', rawPrice: '$1,200', rawWeight: '2.2 kg', rawDate: '2026-10-11' },
  { rawCategory: 'laptop', rawPrice: '€1,100', rawWeight: '4.85 lbs', rawDate: '11/10/2026' },
  { rawCategory: ' LAPTOP ', rawPrice: '£950', rawWeight: '2200 g', rawDate: 'Oct 11, 2026' },
  { rawCategory: 'Smart Phone', rawPrice: '$800', rawWeight: '180 g', rawDate: '2026-10-11' },
  { rawCategory: 'smartphone', rawPrice: '€740', rawWeight: '0.18 kg', rawDate: '2026/10/11' },
  { rawCategory: 'Cell Phone', rawPrice: '$820', rawWeight: '0.40 lbs', rawDate: '11-Oct-2026' },
]

export function ValueStandardisationVisualizer() {
  const state = useFigureState({
    pipelineStep: choice<'raw' | 'canonical'>(
      [
        { value: 'raw', label: 'raw multi-source inputs (heterogeneous variants)' },
        { value: 'canonical', label: 'standardised canonical representation' },
      ],
      'raw',
      { label: 'standardisation state' },
    ),
  })

  const results = useMemo(() => {
    if (state.pipelineStep === 'raw') {
      return {
        rows: RAW_INVENTORY.map((item) => ({
          category: item.rawCategory,
          price: item.rawPrice,
          weight: item.rawWeight,
          date: item.rawDate,
          isVariant: true,
        })),
        uniqueCategories: 5,
        priceCurrencies: '3 currencies (USD, EUR, GBP)',
        weightUnits: '3 units (kg, lbs, g)',
        dateFormats: '4 syntax formats',
      }
    }

    // Standardised canonical form:
    // Category: uppercase canonical enum ['LAPTOP', 'SMARTPHONE']
    // Price: USD normalized (1 EUR = 1.08 USD, 1 GBP = 1.28 USD)
    // Weight: grams (integer) or kg (float)
    // Date: ISO 8601 YYYY-MM-DD
    const canonicalRows = [
      { category: 'LAPTOP', price: '$1,200.00', weight: '2.20 kg', date: '2026-10-11', isVariant: false },
      { category: 'LAPTOP', price: '$1,188.00', weight: '2.20 kg', date: '2026-10-11', isVariant: false },
      { category: 'LAPTOP', price: '$1,216.00', weight: '2.20 kg', date: '2026-10-11', isVariant: false },
      { category: 'SMARTPHONE', price: '$800.00', weight: '0.18 kg', date: '2026-10-11', isVariant: false },
      { category: 'SMARTPHONE', price: '$799.20', weight: '0.18 kg', date: '2026-10-11', isVariant: false },
      { category: 'SMARTPHONE', price: '$820.00', weight: '0.18 kg', date: '2026-10-11', isVariant: false },
    ]

    return {
      rows: canonicalRows,
      uniqueCategories: 2,
      priceCurrencies: 'Unified USD',
      weightUnits: 'Unified SI (kg)',
      dateFormats: 'Unified ISO 8601',
    }
  }, [state.pipelineStep])

  return (
    <Figure
      title="Entity canonicalisation and value standardisation"
      state={state}
      hoverReadout={false}
      caption="Value standardisation collapses equivalence classes of heterogeneous representations into canonical formats. In raw multi-source ingestion, six records for two products exhibit five distinct category strings, three currency units, three weight units, and four date dialects. Standardisation case-folds and aliases categories, applies point-in-time exchange rates, normalizes physical units to SI kilograms, and standardizes dates to ISO 8601."
      readouts={
        <>
          <Readout label="category cardinality" value={`${results.uniqueCategories} levels`} />
          <Readout label="price scale" value={results.priceCurrencies} />
          <Readout label="weight scale" value={results.weightUnits} />
          <Readout label="date standard" value={results.dateFormats} />
        </>
      }
    >
      <div style={{ padding: '12px 16px', overflowX: 'auto', fontSize: '13px', lineHeight: 1.5 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid rgba(128,128,128,0.3)', fontWeight: 600 }}>
              <th style={{ padding: '6px 8px' }}>Category</th>
              <th style={{ padding: '6px 8px' }}>Normalized Price</th>
              <th style={{ padding: '6px 8px' }}>Normalized Weight</th>
              <th style={{ padding: '6px 8px' }}>Date (ISO 8601)</th>
            </tr>
          </thead>
          <tbody>
            {results.rows.map((row, idx) => (
              <tr key={idx} style={{ borderBottom: '1px solid rgba(128,128,128,0.15)' }}>
                <td
                  style={{
                    padding: '6px 8px',
                    color: state.pipelineStep === 'canonical' ? '#16a34a' : 'inherit',
                    fontWeight: state.pipelineStep === 'canonical' ? 600 : 'normal',
                  }}
                >
                  {row.category}
                </td>
                <td style={{ padding: '6px 8px' }}>{row.price}</td>
                <td style={{ padding: '6px 8px' }}>{row.weight}</td>
                <td style={{ padding: '6px 8px' }}>{row.date}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Figure>
  )
}
