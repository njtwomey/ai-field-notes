import React, { useState } from 'react'

type SplitMode = 'expanding' | 'rolling' | 'purged_cv'

export default function TemporalSplitExplorer(): React.ReactElement {
  const [splitMode, setSplitMode] = useState<SplitMode>('purged_cv')
  const [labelHorizonDays, setLabelHorizonDays] = useState<number>(7)
  const [embargoDays, setEmbargoDays] = useState<number>(5)
  const [activeFold, setActiveFold] = useState<number>(2)

  // Timeline width: 100 days
  const totalDays = 100

  // For Purged CV with 4 folds (each 25 days)
  // Fold 1: [0..25], Fold 2: [25..50], Fold 3: [50..75], Fold 4: [75..100]
  const foldSize = 25
  const testStart = activeFold * foldSize
  const testEnd = testStart + foldSize

  // Purged region: training rows whose label spans into test: [testStart - labelHorizonDays, testStart]
  const purgeStart = Math.max(0, testStart - labelHorizonDays)
  const purgeEnd = testStart

  // Embargo region: post-test buffer [testEnd, testEnd + embargoDays]
  const embargoEnd = Math.min(totalDays, testEnd + embargoDays)

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
            Temporal Splitting, Purging & Embargo Simulator
          </h4>
          <span
            style={{
              fontSize: '0.74rem',
              padding: '0.2rem 0.55rem',
              borderRadius: '999px',
              background: 'rgba(56, 189, 248, 0.15)',
              color: '#38bdf8',
              fontWeight: 600,
            }}
          >
            Horizon: {labelHorizonDays}d &middot; Embargo: {embargoDays}d
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
          Explore why multi-step forward labels and auto-regressive feature memory require strict purge and embargo
          buffers around test boundaries.
        </p>
      </div>

      {/* Mode toggle */}
      <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '1.1rem', flexWrap: 'wrap' }}>
        {(
          [
            ['purged_cv', 'Purged & Embargoed CV (López de Prado)'],
            ['expanding', 'Expanding Walk-Forward'],
            ['rolling', 'Rolling (Sliding Window)'],
          ] as const
        ).map(([mKey, label]) => (
          <button
            key={mKey}
            type="button"
            onClick={() => setSplitMode(mKey)}
            style={{
              flex: 1,
              minWidth: '180px',
              padding: '0.4rem 0.6rem',
              fontSize: '0.76rem',
              borderRadius: '5px',
              border: splitMode === mKey ? '1px solid #38bdf8' : '1px solid #444',
              background: splitMode === mKey ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
              color: splitMode === mKey ? '#38bdf8' : '#aaa',
              cursor: 'pointer',
              fontWeight: splitMode === mKey ? 600 : 400,
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Parameter Controls */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '0.85rem',
          marginBottom: '1.25rem',
          padding: '0.75rem',
          borderRadius: '8px',
          background: 'rgba(0, 0, 0, 0.25)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
        }}
      >
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
            <label style={{ fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', color: '#94a3b8' }}>
              Label Horizon (H): <strong style={{ color: '#f59e0b' }}>{labelHorizonDays} days</strong>
            </label>
          </div>
          <input
            type="range"
            min={1}
            max={14}
            step={1}
            value={labelHorizonDays}
            onChange={(e) => setLabelHorizonDays(parseInt(e.target.value, 10))}
            style={{ width: '100%', cursor: 'pointer', accentColor: '#f59e0b' }}
          />
          <span style={{ fontSize: '0.68rem', color: '#64748b' }}>Labels look H days into the future</span>
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
            <label style={{ fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', color: '#94a3b8' }}>
              Post-Test Embargo: <strong style={{ color: '#a855f7' }}>{embargoDays} days</strong>
            </label>
          </div>
          <input
            type="range"
            min={0}
            max={12}
            step={1}
            value={embargoDays}
            onChange={(e) => setEmbargoDays(parseInt(e.target.value, 10))}
            style={{ width: '100%', cursor: 'pointer', accentColor: '#a855f7' }}
          />
          <span style={{ fontSize: '0.68rem', color: '#64748b' }}>Buffers auto-regressive feature leakage</span>
        </div>

        {splitMode === 'purged_cv' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
              <label style={{ fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', color: '#94a3b8' }}>
                Active Test Fold: <strong style={{ color: '#38bdf8' }}>Fold {activeFold + 1} of 4</strong>
              </label>
            </div>
            <div style={{ display: 'flex', gap: '0.3rem' }}>
              {[0, 1, 2, 3].map((fIdx) => (
                <button
                  key={fIdx}
                  type="button"
                  onClick={() => setActiveFold(fIdx)}
                  style={{
                    flex: 1,
                    padding: '0.3rem',
                    fontSize: '0.74rem',
                    borderRadius: '4px',
                    border: activeFold === fIdx ? '1px solid #38bdf8' : '1px solid #444',
                    background: activeFold === fIdx ? 'rgba(56,189,248,0.2)' : 'transparent',
                    color: activeFold === fIdx ? '#38bdf8' : '#aaa',
                    cursor: 'pointer',
                  }}
                >
                  F{fIdx + 1}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Timeline Bar Visualization */}
      <div
        style={{
          padding: '1rem',
          borderRadius: '8px',
          background: 'rgba(0, 0, 0, 0.35)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
          marginBottom: '1.25rem',
        }}
      >
        <div
          style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}
        >
          <span style={{ fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', color: '#94a3b8' }}>
            100-Day Chronological Timeline Partition
          </span>
          <div style={{ display: 'flex', gap: '0.75rem', fontSize: '0.68rem' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <span style={{ width: '9px', height: '9px', borderRadius: '2px', background: '#38bdf8' }} /> Train
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <span style={{ width: '9px', height: '9px', borderRadius: '2px', background: '#f59e0b' }} /> Purged (Label
              Gap)
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <span style={{ width: '9px', height: '9px', borderRadius: '2px', background: '#34d399' }} /> Test /
              Holdout
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <span style={{ width: '9px', height: '9px', borderRadius: '2px', background: '#a855f7' }} /> Embargo
            </span>
          </div>
        </div>

        {/* Timeline Bar */}
        <div
          style={{
            position: 'relative',
            width: '100%',
            height: '36px',
            background: 'rgba(255, 255, 255, 0.05)',
            borderRadius: '6px',
            overflow: 'hidden',
            display: 'flex',
          }}
        >
          {splitMode === 'purged_cv' ? (
            <>
              {/* Pre-test train */}
              {purgeStart > 0 && (
                <div
                  style={{ width: `${purgeStart}%`, height: '100%', background: '#38bdf8', opacity: 0.85 }}
                  title={`Train: Day 0 to ${purgeStart}`}
                />
              )}
              {/* Purge zone */}
              <div
                style={{ width: `${purgeEnd - purgeStart}%`, height: '100%', background: '#f59e0b', opacity: 0.9 }}
                title={`Purged: Day ${purgeStart} to ${purgeEnd}`}
              />
              {/* Test fold */}
              <div
                style={{
                  width: `${testEnd - testStart}%`,
                  height: '100%',
                  background: '#34d399',
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#000',
                  fontSize: '0.75rem',
                }}
                title={`Test: Day ${testStart} to ${testEnd}`}
              >
                TEST
              </div>
              {/* Embargo zone */}
              {embargoEnd > testEnd && (
                <div
                  style={{ width: `${embargoEnd - testEnd}%`, height: '100%', background: '#a855f7', opacity: 0.85 }}
                  title={`Embargo: Day ${testEnd} to ${embargoEnd}`}
                />
              )}
              {/* Post-test train */}
              {100 > embargoEnd && (
                <div
                  style={{ width: `${100 - embargoEnd}%`, height: '100%', background: '#38bdf8', opacity: 0.85 }}
                  title={`Train: Day ${embargoEnd} to 100`}
                />
              )}
            </>
          ) : splitMode === 'expanding' ? (
            <>
              {/* Expanding walk-forward: Train [0..60], Purge [60..67], Test [67..85] */}
              <div
                style={{ width: '55%', height: '100%', background: '#38bdf8', opacity: 0.85 }}
                title="Expanding Train: Days 0..55"
              />
              <div
                style={{ width: `${labelHorizonDays}%`, height: '100%', background: '#f59e0b', opacity: 0.9 }}
                title="Label Horizon Gap"
              />
              <div
                style={{
                  flex: 1,
                  height: '100%',
                  background: '#34d399',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#000',
                  fontSize: '0.75rem',
                  fontWeight: 700,
                }}
                title="Test Window"
              >
                OUT-OF-TIME TEST
              </div>
            </>
          ) : (
            <>
              {/* Rolling sliding window: Inactive past [0..25], Train [25..65], Gap, Test */}
              <div
                style={{ width: '25%', height: '100%', background: 'rgba(255, 255, 255, 0.08)' }}
                title="Discarded Past"
              />
              <div
                style={{ width: '40%', height: '100%', background: '#38bdf8', opacity: 0.85 }}
                title="Sliding Train Window"
              />
              <div
                style={{ width: `${labelHorizonDays}%`, height: '100%', background: '#f59e0b', opacity: 0.9 }}
                title="Label Horizon Gap"
              />
              <div
                style={{
                  flex: 1,
                  height: '100%',
                  background: '#34d399',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#000',
                  fontSize: '0.75rem',
                  fontWeight: 700,
                }}
                title="Test Window"
              >
                TEST
              </div>
            </>
          )}
        </div>

        {/* Axis Ticks */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: '0.68rem',
            color: '#64748b',
            marginTop: '0.35rem',
          }}
        >
          <span>Day 0</span>
          <span>Day 25</span>
          <span>Day 50</span>
          <span>Day 75</span>
          <span>Day 100</span>
        </div>
      </div>

      {/* Explanatory callout */}
      <div
        style={{
          padding: '0.75rem',
          borderRadius: '7px',
          background: 'rgba(0, 0, 0, 0.25)',
          fontSize: '0.78rem',
          color: '#cbd5e1',
          lineHeight: 1.5,
        }}
      >
        <span style={{ fontWeight: 600, color: '#f8fafc' }}>Why Purging & Embargo Matter: </span>
        If a financial label predicts 7-day cumulative returns, a training instance on Day 23 evaluates returns through
        Day 30. Without the amber <strong>Purge zone</strong>, that training row would observe prices inside the test
        set! Similarly, the purple <strong>Embargo zone</strong> stops rolling 30-day moving averages in post-test
        training rows from leaking test observations.
      </div>
    </div>
  )
}
