import { useMemo } from 'react'
import { choice, Figure, Readout, useFigureState } from 'aifn-render'

interface RawRecord {
  idText: string
  countryCode: string
  sensorValue: string
  timestampStr: string
}

const RAW_SAMPLES: RawRecord[] = [
  { idText: '0042', countryCode: 'US', sensorValue: '21.5', timestampStr: '2026-10-11T20:00:00Z' },
  { idText: '0043', countryCode: 'DE', sensorValue: '19.8', timestampStr: '2026-10-11T20:05:00Z' },
  // Trap 1: Country code "NA" is Namibia! Naive parsers turn this into NaN.
  { idText: '0044', countryCode: 'NA', sensorValue: '28.2', timestampStr: '2026-10-11T20:10:00Z' },
  // Trap 2: Sensor value "-999" is a hardware error sentinel code, not -999 degrees.
  { idText: '0045', countryCode: 'FR', sensorValue: '-999', timestampStr: '2026-10-11T20:15:00Z' },
  // Trap 3: European comma decimal "23,4"
  { idText: '0046', countryCode: 'IT', sensorValue: '23,4', timestampStr: '2026-10-11T20:20:00Z' },
]

export function TypeInferenceVisualizer() {
  const state = useFigureState({
    parserType: choice<'naive_heuristic' | 'strict_schema'>(
      [
        { value: 'naive_heuristic', label: 'naive heuristic parser (auto dtype guessing)' },
        { value: 'strict_schema', label: 'strict schema contract (Arrow / Polars explicit)' },
      ],
      'naive_heuristic',
      { label: 'ingestion parser' },
    ),
  })

  const parsed = useMemo(() => {
    if (state.parserType === 'naive_heuristic') {
      return {
        rows: RAW_SAMPLES.map((r) => ({
          // 0042 -> 42 (loss of leading zeros!)
          idDisplay: `${parseInt(r.idText, 10)} (int64)`,
          idCorrupted: true,
          // 'NA' -> null (Namibia wiped out!)
          countryDisplay: r.countryCode === 'NA' ? 'NaN (null)' : `${r.countryCode} (string)`,
          countryCorrupted: r.countryCode === 'NA',
          // '-999' kept as literal number -999, '23,4' turns column into object
          sensorDisplay: r.sensorValue === '23,4' ? '"23,4" (string)' : `${parseFloat(r.sensorValue)} (float64)`,
          sensorCorrupted: r.sensorValue === '-999' || r.sensorValue === '23,4',
          timeDisplay: r.timestampStr.replace('Z', ' (naive string)'),
        })),
        silentCorruptions: 4,
        memoryPerRecord: '144 bytes (Python object pointers)',
        throughput: '12 MB/s (mixed fallback loops)',
      }
    }

    // Strict schema:
    return {
      rows: RAW_SAMPLES.map((r) => ({
        // Preserved as string
        idDisplay: `"${r.idText}" (Utf8)`,
        idCorrupted: false,
        // Namibia preserved via keep_default_na=False
        countryDisplay: `"${r.countryCode}" (Cat)`,
        countryCorrupted: false,
        // -999 mapped to null, European decimal correctly parsed
        sensorDisplay:
          r.sensorValue === '-999'
            ? 'null (Float32)'
            : r.sensorValue === '23,4'
              ? '23.4 (Float32)'
              : `${parseFloat(r.sensorValue)} (Float32)`,
        sensorCorrupted: false,
        timeDisplay: `${r.timestampStr.slice(11, 19)} (UTC Timestamp)`,
      })),
      silentCorruptions: 0,
      memoryPerRecord: '24 bytes (contiguous Arrow array)',
      throughput: '850 MB/s (SIMD parallel Rust engine)',
    }
  }, [state.parserType])

  return (
    <Figure
      title="Silent data corruption across type inference paradigms"
      state={state}
      hoverReadout={false}
      caption="Evaluating raw CSV parsing hazards. A naive heuristic parser guesses column types from string patterns: it strips leading zeros from ID strings, converts the ISO country code 'NA' (Namibia) into a missing null, treats the hardware error sentinel -999 as a genuine physical measurement, and falls back to slow Python object strings when encountering European comma decimals. Supplying an explicit schema contract with custom null sentinels preserves ground truth integrity."
      readouts={
        <>
          <Readout label="silent corruptions" value={`${parsed.silentCorruptions} cells`} />
          <Readout label="memory efficiency" value={parsed.memoryPerRecord} />
          <Readout label="ingestion throughput" value={parsed.throughput} />
          <Readout label="schema safety" value={state.parserType === 'strict_schema' ? 'Guaranteed' : 'Compromised'} />
        </>
      }
    >
      <div style={{ padding: '12px 16px', overflowX: 'auto', fontSize: '13px', lineHeight: 1.5 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid rgba(128,128,128,0.3)', fontWeight: 600 }}>
              <th style={{ padding: '6px 8px' }}>User ID</th>
              <th style={{ padding: '6px 8px' }}>Country</th>
              <th style={{ padding: '6px 8px' }}>Sensor (°C)</th>
              <th style={{ padding: '6px 8px' }}>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {parsed.rows.map((row, idx) => (
              <tr key={idx} style={{ borderBottom: '1px solid rgba(128,128,128,0.15)' }}>
                <td
                  style={{
                    padding: '6px 8px',
                    color: row.idCorrupted ? '#ef4444' : 'inherit',
                    backgroundColor: row.idCorrupted ? 'rgba(239, 68, 68, 0.15)' : 'transparent',
                    fontWeight: row.idCorrupted ? 600 : 'normal',
                  }}
                >
                  {row.idDisplay} {row.idCorrupted && '⚠️ (zero stripped)'}
                </td>
                <td
                  style={{
                    padding: '6px 8px',
                    color: row.countryCorrupted ? '#ef4444' : 'inherit',
                    backgroundColor: row.countryCorrupted ? 'rgba(239, 68, 68, 0.15)' : 'transparent',
                    fontWeight: row.countryCorrupted ? 600 : 'normal',
                  }}
                >
                  {row.countryDisplay} {row.countryCorrupted && '⚠️ (Namibia nullified)'}
                </td>
                <td
                  style={{
                    padding: '6px 8px',
                    color: row.sensorCorrupted ? '#ef4444' : 'inherit',
                    backgroundColor: row.sensorCorrupted ? 'rgba(239, 68, 68, 0.15)' : 'transparent',
                    fontWeight: row.sensorCorrupted ? 600 : 'normal',
                  }}
                >
                  {row.sensorDisplay} {row.sensorCorrupted && '⚠️ (corrupt type/sentinel)'}
                </td>
                <td style={{ padding: '6px 8px' }}>{row.timeDisplay}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Figure>
  )
}
