import { useMemo } from 'react'
import { choice, Figure, Readout, useFigureState } from 'aifn-render'

interface EmployeeTuple {
  id: number
  name: string
  role: string
  department: string
  salary: number
  managerId: number
}

const BASELINE_DATA: EmployeeTuple[] = [
  { id: 1, name: 'Alice', role: 'Engineering Director', department: 'Platform', salary: 210000, managerId: 0 },
  { id: 2, name: 'Bob', role: 'Staff Engineer', department: 'Platform', salary: 180000, managerId: 1 },
  { id: 3, name: 'Charlie', role: 'Data Scientist', department: 'Analytics', salary: 140000, managerId: 1 },
  // Corruption 1: Charlie has role Data Scientist but department typed as "Hardware" (violates FD: Role -> Department)
  { id: 4, name: 'Dana', role: 'Data Scientist', department: 'Hardware', salary: 145000, managerId: 1 },
  // Corruption 2: Evan earns 250,000 while reporting to Bob (180,000) (violates Denial Constraint: Salary <= Manager.Salary)
  { id: 5, name: 'Evan', role: 'Junior Engineer', department: 'Platform', salary: 250000, managerId: 2 },
  // Corruption 3: Frank has negative salary (-80,000) (violates Domain Constraint: Salary > 0)
  { id: 6, name: 'Frank', role: 'Data Scientist', department: 'Analytics', salary: -80000, managerId: 1 },
]

export function ConstraintRepairExplorer() {
  const state = useFigureState({
    repairMethod: choice<'raw' | 'min_cardinality' | 'probabilistic'>(
      [
        { value: 'raw', label: 'unrepaired database (active violations)' },
        { value: 'min_cardinality', label: 'minimal cardinality repair (ILP)' },
        { value: 'probabilistic', label: 'probabilistic repair (HoloClean PGM)' },
      ],
      'raw',
      { label: 'repair policy' },
    ),
  })

  const repaired = useMemo(() => {
    const rows = BASELINE_DATA.map((r) => ({ ...r }))
    let cellsChanged = 0

    if (state.repairMethod === 'raw') {
      return {
        rows,
        cellsChanged: 0,
        fdViolations: 1,
        dcViolations: 2,
        isConsistent: false,
      }
    }

    if (state.repairMethod === 'min_cardinality') {
      // Minimal cell updates:
      // 1. Dana.department -> 'Analytics' (1 cell to fix FD)
      rows[3].department = 'Analytics'
      cellsChanged++

      // 2. Evan.salary -> 180,000 (clamp to manager salary Bob: 180,000)
      rows[4].salary = 180000
      cellsChanged++

      // 3. Frank.salary -> 1 (minimal positive integer satisfying Salary > 0)
      // Note: minimal cardinality doesn't know typical data scientist salaries!
      rows[5].salary = 1
      cellsChanged++

      return {
        rows,
        cellsChanged,
        fdViolations: 0,
        dcViolations: 0,
        isConsistent: true,
      }
    }

    // Probabilistic repair (HoloClean):
    // Leverages distribution priors:
    // 1. Dana.department -> 'Analytics' (P(Dept='Analytics' | Role='Data Scientist') ~ 0.99)
    rows[3].department = 'Analytics'
    cellsChanged++

    // 2. Evan.salary -> 95,000 (draws from Junior Engineer salary prior N(95k, 10k) <= 180k)
    rows[4].salary = 95000
    cellsChanged++

    // 3. Frank.salary -> 142,000 (inverts sign / draws from Data Scientist prior ~ 140k)
    rows[5].salary = 142000
    cellsChanged++

    return {
      rows,
      cellsChanged,
      fdViolations: 0,
      dcViolations: 0,
      isConsistent: true,
    }
  }, [state.repairMethod])

  return (
    <Figure
      title="Integrity constraints, conflict hypergraphs, and holistic repair"
      state={state}
      hoverReadout={false}
      caption="Evaluating relational repair under Functional Dependencies (Role → Department) and Denial Constraints (Salary > 0 and Salary ≤ Manager.Salary). The raw database has three conflicting cells. Minimum cardinality repair alters the fewest cells to restore logical consistency, but sets Frank's salary to $1 (unrealistic). Probabilistic repair (HoloClean) couples logical constraint satisfiability with learned feature priors, restoring realistic statistical distributions."
      readouts={
        <>
          <Readout label="database consistency" value={repaired.isConsistent ? 'Consistent (Σ ⊨ I)' : 'Violated'} />
          <Readout label="FD violations" value={`${repaired.fdViolations}`} />
          <Readout label="denial constraint violations" value={`${repaired.dcViolations}`} />
          <Readout label="cells modified" value={`${repaired.cellsChanged} cells`} />
        </>
      }
    >
      <div style={{ padding: '12px 16px', overflowX: 'auto', fontSize: '13px', lineHeight: 1.5 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid rgba(128,128,128,0.3)', fontWeight: 600 }}>
              <th style={{ padding: '6px 8px' }}>ID</th>
              <th style={{ padding: '6px 8px' }}>Name</th>
              <th style={{ padding: '6px 8px' }}>Role</th>
              <th style={{ padding: '6px 8px' }}>Department</th>
              <th style={{ padding: '6px 8px' }}>Salary</th>
              <th style={{ padding: '6px 8px' }}>Manager</th>
            </tr>
          </thead>
          <tbody>
            {repaired.rows.map((row) => {
              const isDanaDept = row.id === 4
              const isEvanSal = row.id === 5
              const isFrankSal = row.id === 6

              const isDeptModified = isDanaDept && state.repairMethod !== 'raw'
              const isEvanSalModified = isEvanSal && state.repairMethod !== 'raw'
              const isFrankSalModified = isFrankSal && state.repairMethod !== 'raw'

              const isDeptViolating = isDanaDept && state.repairMethod === 'raw'
              const isEvanViolating = isEvanSal && state.repairMethod === 'raw'
              const isFrankViolating = isFrankSal && state.repairMethod === 'raw'

              return (
                <tr key={row.id} style={{ borderBottom: '1px solid rgba(128,128,128,0.15)' }}>
                  <td style={{ padding: '6px 8px' }}>{row.id}</td>
                  <td style={{ padding: '6px 8px', fontWeight: 500 }}>{row.name}</td>
                  <td style={{ padding: '6px 8px' }}>{row.role}</td>
                  <td
                    style={{
                      padding: '6px 8px',
                      backgroundColor: isDeptViolating
                        ? 'rgba(239, 68, 68, 0.2)'
                        : isDeptModified
                          ? 'rgba(34, 197, 94, 0.2)'
                          : 'transparent',
                      color: isDeptViolating ? '#ef4444' : isDeptModified ? '#16a34a' : 'inherit',
                      fontWeight: isDeptViolating || isDeptModified ? 600 : 'normal',
                    }}
                  >
                    {row.department} {isDeptViolating && '⚠️ (FD error)'} {isDeptModified && '✓'}
                  </td>
                  <td
                    style={{
                      padding: '6px 8px',
                      backgroundColor:
                        isEvanViolating || isFrankViolating
                          ? 'rgba(239, 68, 68, 0.2)'
                          : isEvanSalModified || isFrankSalModified
                            ? 'rgba(34, 197, 94, 0.2)'
                            : 'transparent',
                      color:
                        isEvanViolating || isFrankViolating
                          ? '#ef4444'
                          : isEvanSalModified || isFrankSalModified
                            ? '#16a34a'
                            : 'inherit',
                      fontWeight:
                        isEvanViolating || isFrankViolating || isEvanSalModified || isFrankSalModified ? 600 : 'normal',
                    }}
                  >
                    ${row.salary.toLocaleString()} {isEvanViolating && '⚠️ (> Manager)'}
                    {isFrankViolating && '⚠️ (< 0)'}
                    {(isEvanSalModified || isFrankSalModified) && '✓'}
                  </td>
                  <td style={{ padding: '6px 8px' }}>ID: {row.managerId || 'None'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Figure>
  )
}
