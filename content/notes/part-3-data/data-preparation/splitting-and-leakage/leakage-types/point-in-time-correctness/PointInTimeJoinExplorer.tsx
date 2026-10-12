import React, { useMemo, useState } from 'react'

interface EventRecord {
  id: string
  eventName: string
  attribute: string
  value: string
  validTimeMinutes: number // e.g. minutes from 08:00 (0 to 600)
  validTimeStr: string
  recordTimeMinutes: number // when it landed in DB
  recordTimeStr: string
  nature: 'normal' | 'delayed_ingest' | 'future_event' | 'backfill'
}

const CUSTOMER_EVENTS: EventRecord[] = [
  {
    id: 'e1',
    eventName: 'Initial Account Snapshot',
    attribute: 'account_balance',
    value: '$1,200',
    validTimeMinutes: 60, // 09:00
    validTimeStr: '09:00',
    recordTimeMinutes: 65, // 09:05
    recordTimeStr: '09:05',
    nature: 'normal',
  },
  {
    id: 'e2',
    eventName: 'Wire Transfer Inflow (+$8,000)',
    attribute: 'account_balance',
    value: '$9,200',
    validTimeMinutes: 210, // 11:30
    validTimeStr: '11:30',
    recordTimeMinutes: 270, // 12:30 (1hr pipeline ingestion delay!)
    recordTimeStr: '12:30',
    nature: 'delayed_ingest',
  },
  {
    id: 'e3',
    eventName: 'Residential Address Update',
    attribute: 'risk_jurisdiction',
    value: 'High-Risk Zone B',
    validTimeMinutes: 360, // 14:00
    validTimeStr: '14:00',
    recordTimeMinutes: 370, // 14:10
    recordTimeStr: '14:10',
    nature: 'future_event',
  },
  {
    id: 'e4',
    eventName: 'Fraud Flag Backfill (Retroactive)',
    attribute: 'prior_fraud_flag',
    value: 'TRUE',
    validTimeMinutes: 120, // Fraud was backdated to 10:00
    validTimeStr: '10:00 (backdated)',
    recordTimeMinutes: 480, // But logged at 16:00!
    recordTimeStr: '16:00',
    nature: 'backfill',
  },
]

export default function PointInTimeJoinExplorer(): React.ReactElement {
  // Prediction time in minutes from 08:00. Default 240 = 12:00
  const [predTimeMinutes, setPredTimeMinutes] = useState<number>(240)
  const [useAsOfJoin, setUseAsOfJoin] = useState<boolean>(true)

  // Format pred time
  const predTimeStr = useMemo(() => {
    const hours = Math.floor(predTimeMinutes / 60) + 8
    const mins = predTimeMinutes % 60
    return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}`
  }, [predTimeMinutes])

  // Compute joined feature values based on strategy
  const { balanceValue, addressValue, fraudValue, hasLeakage, leakageReason } = useMemo(() => {
    if (!useAsOfJoin) {
      // Current-state / Naive join uses the latest state regardless of time!
      return {
        balanceValue: '$9,200',
        addressValue: 'High-Risk Zone B',
        fraudValue: 'TRUE',
        hasLeakage: true,
        leakageReason:
          'Naive join reads current database snapshot! The model receives the 14:00 address change, the 12:30 delayed wire transfer, and the 16:00 backfilled fraud flag—all completely unknown at prediction time!',
      }
    }

    // Point-in-time as-of join: only records with recordTimeMinutes <= predTimeMinutes are visible
    const visibleBalance = [...CUSTOMER_EVENTS]
      .filter((e) => e.attribute === 'account_balance' && e.recordTimeMinutes <= predTimeMinutes)
      .sort((a, b) => b.recordTimeMinutes - a.recordTimeMinutes)[0]

    const visibleAddress = [...CUSTOMER_EVENTS]
      .filter((e) => e.attribute === 'risk_jurisdiction' && e.recordTimeMinutes <= predTimeMinutes)
      .sort((a, b) => b.recordTimeMinutes - a.recordTimeMinutes)[0]

    const visibleFraud = [...CUSTOMER_EVENTS]
      .filter((e) => e.attribute === 'prior_fraud_flag' && e.recordTimeMinutes <= predTimeMinutes)
      .sort((a, b) => b.recordTimeMinutes - a.recordTimeMinutes)[0]

    const bal = visibleBalance ? visibleBalance.value : '$1,200'
    const addr = visibleAddress ? visibleAddress.value : 'Standard Zone A (Default)'
    const fraud = visibleFraud ? visibleFraud.value : 'FALSE'

    return {
      balanceValue: bal,
      addressValue: addr,
      fraudValue: fraud,
      hasLeakage: false,
      leakageReason: '',
    }
  }, [useAsOfJoin, predTimeMinutes])

  return (
    <div
      style={{
        margin: '1.75rem 0',
        padding: '1.25rem',
        borderRadius: '10px',
        border: '1px solid var(--color-border-subtle, #333)',
        background: 'var(--color-bg-subtle, #14171d)',
        fontFamily: 'inherit',
      }}
    >
      {/* Header */}
      <div style={{ marginBottom: '1.1rem' }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '0.75rem',
          }}
        >
          <h4 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 600, color: 'var(--color-text-emphasis, #f0f3f6)' }}>
            Point-in-Time As-Of Join Simulator
          </h4>
          <span
            style={{
              fontSize: '0.74rem',
              padding: '0.2rem 0.55rem',
              borderRadius: '999px',
              background: hasLeakage ? 'rgba(244, 63, 94, 0.15)' : 'rgba(52, 211, 153, 0.15)',
              color: hasLeakage ? '#f43f5e' : '#34d399',
              fontWeight: 600,
            }}
          >
            {hasLeakage ? 'FUTURE LEAKAGE ACTIVE' : 'POINT-IN-TIME COMPLIANT'}
          </span>
        </div>
        <p
          style={{
            margin: '0.4rem 0 0',
            fontSize: '0.82rem',
            color: 'var(--color-text-muted, #94a3b8)',
            lineHeight: 1.45,
          }}
        >
          Explore the difference between current-state joins and as-of joins when event timestamps diverge from database
          ingestion timestamps.
        </p>
      </div>

      {/* Control row */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '0.85rem',
          marginBottom: '1.25rem',
          padding: '0.75rem',
          borderRadius: '8px',
          background: 'rgba(0, 0, 0, 0.25)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
        }}
      >
        <div>
          <label
            style={{
              display: 'block',
              fontSize: '0.72rem',
              fontWeight: 600,
              textTransform: 'uppercase',
              color: '#94a3b8',
              marginBottom: '0.35rem',
            }}
          >
            Feature Extraction Strategy
          </label>
          <div style={{ display: 'flex', gap: '0.4rem' }}>
            <button
              type="button"
              onClick={() => setUseAsOfJoin(true)}
              style={{
                flex: 1,
                padding: '0.35rem 0.6rem',
                fontSize: '0.76rem',
                borderRadius: '5px',
                border: useAsOfJoin ? '1px solid #34d399' : '1px solid #444',
                background: useAsOfJoin ? 'rgba(52, 211, 153, 0.2)' : 'transparent',
                color: useAsOfJoin ? '#34d399' : '#aaa',
                cursor: 'pointer',
                fontWeight: useAsOfJoin ? 600 : 400,
              }}
            >
              As-Of Join (Record-Time)
            </button>
            <button
              type="button"
              onClick={() => setUseAsOfJoin(false)}
              style={{
                flex: 1,
                padding: '0.35rem 0.6rem',
                fontSize: '0.76rem',
                borderRadius: '5px',
                border: !useAsOfJoin ? '1px solid #f43f5e' : '1px solid #444',
                background: !useAsOfJoin ? 'rgba(244, 63, 94, 0.2)' : 'transparent',
                color: !useAsOfJoin ? '#f43f5e' : '#aaa',
                cursor: 'pointer',
                fontWeight: !useAsOfJoin ? 600 : 400,
              }}
            >
              Naive Current-State Join
            </button>
          </div>
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
            <label style={{ fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', color: '#94a3b8' }}>
              Prediction Timestamp (t_pred): <strong style={{ color: '#38bdf8' }}>{predTimeStr}</strong>
            </label>
          </div>
          <input
            type="range"
            min={60}
            max={540}
            step={15}
            value={predTimeMinutes}
            onChange={(e) => setPredTimeMinutes(parseInt(e.target.value, 10))}
            style={{ width: '100%', cursor: 'pointer', accentColor: '#38bdf8' }}
          />
        </div>
      </div>

      {/* Visual Timeline of Events */}
      <div
        style={{
          padding: '0.85rem',
          borderRadius: '8px',
          background: 'rgba(0, 0, 0, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
          marginBottom: '1.25rem',
        }}
      >
        <div
          style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}
        >
          <span style={{ fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', color: '#94a3b8' }}>
            Event Log & Ingestion Delays vs Prediction Time ({predTimeStr})
          </span>
          <span style={{ fontSize: '0.7rem', color: '#64748b' }}>Dual Timestamps: Valid vs Ingested</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {CUSTOMER_EVENTS.map((evt) => {
            const isIngested = evt.recordTimeMinutes <= predTimeMinutes

            let statusTag = 'VISIBLE TO AS-OF JOIN'
            let tagBg = 'rgba(52, 211, 153, 0.2)'
            let tagColor = '#6ee7b7'

            if (!isIngested) {
              if (evt.nature === 'delayed_ingest') {
                statusTag = 'PIPELINE DELAY (NOT IN DB YET)'
                tagBg = 'rgba(245, 158, 11, 0.2)'
                tagColor = '#fcd34d'
              } else if (evt.nature === 'backfill') {
                statusTag = 'RETROACTIVE BACKFILL (FUTURE)'
                tagBg = 'rgba(168, 85, 247, 0.2)'
                tagColor = '#d8b4fe'
              } else {
                statusTag = 'FUTURE EVENT (UNKNOWN)'
                tagBg = 'rgba(244, 63, 94, 0.2)'
                tagColor = '#fda4af'
              }
            }

            return (
              <div
                key={evt.id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '0.5rem 0.75rem',
                  borderRadius: '6px',
                  background: isIngested ? 'rgba(255, 255, 255, 0.04)' : 'rgba(255, 255, 255, 0.01)',
                  border: isIngested ? '1px solid rgba(255, 255, 255, 0.08)' : '1px dashed #333',
                  opacity: isIngested || !useAsOfJoin ? 1 : 0.45,
                  fontSize: '0.78rem',
                }}
              >
                <div>
                  <strong style={{ color: isIngested ? '#f8fafc' : '#94a3b8' }}>{evt.eventName}</strong>
                  <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '0.15rem' }}>
                    Valid Time: <span style={{ color: '#cbd5e1' }}>{evt.validTimeStr}</span> &middot; Ingested in DB:{' '}
                    <span style={{ color: '#cbd5e1' }}>{evt.recordTimeStr}</span> &middot; Value:{' '}
                    <span style={{ color: '#38bdf8' }}>{evt.value}</span>
                  </div>
                </div>

                <div>
                  <span
                    style={{
                      fontSize: '0.65rem',
                      fontWeight: 700,
                      padding: '0.15rem 0.45rem',
                      borderRadius: '3px',
                      background: tagBg,
                      color: tagColor,
                    }}
                  >
                    {statusTag}
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Feature Snapshot Vector at Prediction Time */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '0.75rem',
          marginBottom: '1.25rem',
        }}
      >
        <div
          style={{
            padding: '0.75rem',
            borderRadius: '6px',
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid rgba(255, 255, 255, 0.05)',
          }}
        >
          <span style={{ display: 'block', fontSize: '0.68rem', color: '#94a3b8', textTransform: 'uppercase' }}>
            Feature: account_balance
          </span>
          <strong
            style={{
              fontSize: '1.25rem',
              color: balanceValue === '$9,200' && predTimeMinutes < 270 ? '#f43f5e' : '#34d399',
            }}
          >
            {balanceValue}
          </strong>
          <span style={{ display: 'block', fontSize: '0.68rem', color: '#64748b', marginTop: '0.2rem' }}>
            {balanceValue === '$9,200' && predTimeMinutes < 270
              ? 'LEAKED (landed at 12:30)'
              : 'Correct for ' + predTimeStr}
          </span>
        </div>

        <div
          style={{
            padding: '0.75rem',
            borderRadius: '6px',
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid rgba(255, 255, 255, 0.05)',
          }}
        >
          <span style={{ display: 'block', fontSize: '0.68rem', color: '#94a3b8', textTransform: 'uppercase' }}>
            Feature: risk_jurisdiction
          </span>
          <strong
            style={{
              fontSize: '1.1rem',
              color: addressValue.includes('High-Risk') && predTimeMinutes < 370 ? '#f43f5e' : '#34d399',
            }}
          >
            {addressValue}
          </strong>
          <span style={{ display: 'block', fontSize: '0.68rem', color: '#64748b', marginTop: '0.2rem' }}>
            {addressValue.includes('High-Risk') && predTimeMinutes < 370
              ? 'LEAKED (moved at 14:00)'
              : 'Correct for ' + predTimeStr}
          </span>
        </div>

        <div
          style={{
            padding: '0.75rem',
            borderRadius: '6px',
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid rgba(255, 255, 255, 0.05)',
          }}
        >
          <span style={{ display: 'block', fontSize: '0.68rem', color: '#94a3b8', textTransform: 'uppercase' }}>
            Feature: prior_fraud_flag
          </span>
          <strong
            style={{
              fontSize: '1.25rem',
              color: fraudValue === 'TRUE' && predTimeMinutes < 480 ? '#f43f5e' : '#34d399',
            }}
          >
            {fraudValue}
          </strong>
          <span style={{ display: 'block', fontSize: '0.68rem', color: '#64748b', marginTop: '0.2rem' }}>
            {fraudValue === 'TRUE' && predTimeMinutes < 480
              ? 'LEAKED (backfilled at 16:00)'
              : 'Correct for ' + predTimeStr}
          </span>
        </div>
      </div>

      {/* Explanatory callout */}
      <div
        style={{
          padding: '0.75rem',
          borderRadius: '7px',
          background: hasLeakage ? 'rgba(244, 63, 94, 0.08)' : 'rgba(52, 211, 153, 0.08)',
          borderLeft: hasLeakage ? '3px solid #f43f5e' : '3px solid #34d399',
          fontSize: '0.8rem',
          color: '#e2e8f0',
          lineHeight: 1.5,
        }}
      >
        {hasLeakage ? (
          <div>
            <strong>Future Leakage Alert: </strong> {leakageReason}
          </div>
        ) : (
          <div>
            <strong>Point-in-Time Correctness Enforced: </strong>
            Notice how at 12:00, the 11:30 wire transfer is excluded because it had not yet cleared ingestion into the
            feature database (which occurred at 12:30). An as-of join guarantees that offline training features
            precisely mirror real-time serving states.
          </div>
        )}
      </div>
    </div>
  )
}
