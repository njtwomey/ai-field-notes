import { useMemo } from 'react'
import { choice, Figure, Readout, slider, useFigureState } from 'aifn-render'

export function DataValidationExplorer() {
  const state = useFigureState({
    schemaAnomaly: choice<'none' | 'type_mismatch' | 'duplicate_keys'>(
      [
        { value: 'none', label: 'valid structural schema' },
        { value: 'type_mismatch', label: 'type mismatch (timestamp string)' },
        { value: 'duplicate_keys', label: 'duplicate primary keys (0.8% dupes)' },
      ],
      'none',
      { label: 'schema anomaly' },
    ),
    nullRate: slider(0.0, 0.08, 0.01, { step: 0.01, label: 'observed null fraction' }),
    psiDrift: slider(0.02, 0.35, 0.06, { step: 0.02, label: 'distribution drift (PSI)' }),
  })

  const results = useMemo(() => {
    // 5 Expectations:
    // 1. Schema & Types
    const schemaPass = state.schemaAnomaly === 'none'
    const schemaMsg =
      state.schemaAnomaly === 'none'
        ? 'Pass (all 12 columns match contract types)'
        : state.schemaAnomaly === 'type_mismatch'
          ? 'Fail (expected DateTime, got Utf8)'
          : 'Pass (schema matches, pending key check)'

    // 2. Uniqueness
    const uniquePass = state.schemaAnomaly !== 'duplicate_keys'
    const uniqueMsg = uniquePass ? 'Pass (0 duplicate user_ids)' : 'Fail (8 duplicate user_ids detected)'

    // 3. Nullability threshold (contract SLA <= 3%)
    const nullPass = state.nullRate <= 0.03
    const nullMsg = `${(state.nullRate * 100).toFixed(1)}% missing (SLA threshold: ≤ 3.0%)`

    // 4. Value Bounds Check [0, 100]
    const boundsPass = true
    const boundsMsg = 'Pass (100% of values in range [0, 100])'

    // 5. Distribution Stability (PSI <= 0.10)
    const psiPass = state.psiDrift <= 0.1
    const psiLevel =
      state.psiDrift <= 0.1
        ? 'Stable (no drift)'
        : state.psiDrift <= 0.2
          ? 'Moderate drift'
          : 'Severe drift (model degradation)'
    const psiMsg = `PSI = ${state.psiDrift.toFixed(3)} (${psiLevel}, threshold: ≤ 0.10)`

    const checks = [
      { name: 'Schema & Column Types', pass: schemaPass, detail: schemaMsg },
      { name: 'Primary Key Uniqueness', pass: uniquePass, detail: uniqueMsg },
      { name: 'Column Nullability SLA', pass: nullPass, detail: nullMsg },
      { name: 'Value Domain Ranges', pass: boundsPass, detail: boundsMsg },
      { name: 'Distribution Stability (PSI)', pass: psiPass, detail: psiMsg },
    ]

    const failures = checks.filter((c) => !c.pass).length
    const pipelineAction = failures === 0 ? 'INGEST & SCORE' : 'CIRCUIT BREAKER: QUARANTINE'

    return {
      checks,
      failures,
      pipelineAction,
    }
  }, [state.schemaAnomaly, state.nullRate, state.psiDrift])

  return (
    <Figure
      title="Automated batch data validation suite and circuit breaking"
      state={state}
      hoverReadout={false}
      caption="Evaluating an incoming production batch against an automated validation contract. Five formal assertions monitor structural types, primary key uniqueness, nullability bounds, domain value ranges, and population stability (PSI). If any contractual assertion fails, automated circuit breakers halt the scoring pipeline, quarantining the tainted payload and preventing training-serving skew or corrupted predictions."
      readouts={
        <>
          <Readout label="pipeline decision" value={results.pipelineAction} />
          <Readout label="failed expectations" value={`${results.failures} / 5`} />
          <Readout label="nullability status" value={state.nullRate <= 0.03 ? 'Within SLA' : 'SLA Breached'} />
          <Readout label="distribution stability" value={state.psiDrift <= 0.1 ? 'Stable' : 'Drifting'} />
        </>
      }
    >
      <div style={{ padding: '12px 16px', overflowX: 'auto', fontSize: '13px', lineHeight: 1.5 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid rgba(128,128,128,0.3)', fontWeight: 600 }}>
              <th style={{ padding: '6px 8px' }}>Expectation / Assertion</th>
              <th style={{ padding: '6px 8px' }}>Status</th>
              <th style={{ padding: '6px 8px' }}>Measured Result</th>
            </tr>
          </thead>
          <tbody>
            {results.checks.map((check, idx) => (
              <tr key={idx} style={{ borderBottom: '1px solid rgba(128,128,128,0.15)' }}>
                <td style={{ padding: '6px 8px', fontWeight: 500 }}>{check.name}</td>
                <td
                  style={{
                    padding: '6px 8px',
                    fontWeight: 600,
                    color: check.pass ? '#16a34a' : '#ef4444',
                  }}
                >
                  {check.pass ? '✓ PASS' : '✗ FAIL'}
                </td>
                <td style={{ padding: '6px 8px' }}>{check.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Figure>
  )
}
