import React, { useMemo, useState } from 'react'

interface UserRecord {
  id: string
  name: string
  batch: 1 | 2
  classLabel: 'Active' | 'Churned'
}

const INITIAL_USERS: UserRecord[] = [
  { id: 'usr_001', name: 'Alice (usr_001)', batch: 1, classLabel: 'Active' },
  { id: 'usr_002', name: 'Bob (usr_002)', batch: 1, classLabel: 'Churned' },
  { id: 'usr_003', name: 'Charlie (usr_003)', batch: 1, classLabel: 'Active' },
  { id: 'usr_004', name: 'Diana (usr_004)', batch: 1, classLabel: 'Active' },
  { id: 'usr_005', name: 'Evan (usr_005)', batch: 1, classLabel: 'Churned' },
  { id: 'usr_006', name: 'Fiona (usr_006)', batch: 1, classLabel: 'Active' },
  { id: 'usr_007', name: 'George (usr_007)', batch: 1, classLabel: 'Active' },
  { id: 'usr_008', name: 'Hannah (usr_008)', batch: 1, classLabel: 'Churned' },
]

const BATCH_2_USERS: UserRecord[] = [
  { id: 'usr_009', name: 'Ian (usr_009)', batch: 2, classLabel: 'Active' },
  { id: 'usr_010', name: 'Julia (usr_010)', batch: 2, classLabel: 'Active' },
  { id: 'usr_011', name: 'Kevin (usr_011)', batch: 2, classLabel: 'Churned' },
  { id: 'usr_012', name: 'Laura (usr_012)', batch: 2, classLabel: 'Active' },
]

// Simple deterministic hash simulation (Murmur-like 32-bit hash approximation)
function deterministicHashBucket(str: string, salt: string): number {
  let hash = 0
  const combined = `${salt}:${str}`
  for (let i = 0; i < combined.length; i++) {
    hash = (hash << 5) - hash + combined.charCodeAt(i)
    hash |= 0 // Convert to 32bit integer
  }
  return Math.abs(hash) % 100
}

export default function StableHashingSplitExplorer(): React.ReactElement {
  const [includeBatch2, setIncludeBatch2] = useState<boolean>(false)
  const [salt, setSalt] = useState<string>('prod_v1')
  const [trainRatio, setTrainRatio] = useState<number>(75) // 75% train, 25% test

  const allUsers = useMemo(() => {
    return includeBatch2 ? [...INITIAL_USERS, ...BATCH_2_USERS] : INITIAL_USERS
  }, [includeBatch2])

  // Compute partitions under random shuffle vs deterministic hashing
  const { randomPartitions, hashedPartitions, randomFlips, hashedFlips } = useMemo(() => {
    // 1. Deterministic hashing: bucket < trainRatio -> Train, else Test
    const hashed = allUsers.map((u) => {
      const bucket = deterministicHashBucket(u.id, salt)
      const split = bucket < trainRatio ? 'Train' : 'Test'
      return { ...u, bucket, split }
    })

    // 2. Random shuffle simulation based on dataset size:
    // When batch 2 is added, row indices shift, changing random permutation assignments
    const random = allUsers.map((u, idx) => {
      // In pseudo-random shuffle without stable IDs:
      // When N = 8, indices are [0..7].
      // When N = 12, adding rows shifts the random permutation order completely
      const pseudoRand = includeBatch2 ? (idx * 37 + 13) % 100 : (idx * 23 + 7) % 100
      const split = pseudoRand < trainRatio ? 'Train' : 'Test'
      return { ...u, split }
    })

    // Count how many Batch 1 users flipped partitions between N=8 and N=12
    let rFlips = 0
    let hFlips = 0
    if (includeBatch2) {
      INITIAL_USERS.forEach((u, idx) => {
        const randN8 = (idx * 23 + 7) % 100 < trainRatio ? 'Train' : 'Test'
        const randN12 = (idx * 37 + 13) % 100 < trainRatio ? 'Train' : 'Test'
        if (randN8 !== randN12) rFlips++

        const hashN8 = deterministicHashBucket(u.id, salt) < trainRatio ? 'Train' : 'Test'
        const hashN12 = deterministicHashBucket(u.id, salt) < trainRatio ? 'Train' : 'Test'
        if (hashN8 !== hashN12) hFlips++
      })
    }

    return {
      randomPartitions: random,
      hashedPartitions: hashed,
      randomFlips: rFlips,
      hashedFlips: hFlips,
    }
  }, [allUsers, includeBatch2, salt, trainRatio])

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
            Stable Hashing vs Random Split Simulator
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
            {allUsers.length} Users Tracked
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
          Observe what happens to existing user assignments when new rows are appended to your dataset over time.
        </p>
      </div>

      {/* Control row */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '0.75rem',
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
            Dataset Growth Simulation
          </label>
          <div style={{ display: 'flex', gap: '0.4rem' }}>
            <button
              type="button"
              onClick={() => setIncludeBatch2(false)}
              style={{
                flex: 1,
                padding: '0.35rem 0.6rem',
                fontSize: '0.76rem',
                borderRadius: '5px',
                border: !includeBatch2 ? '1px solid #38bdf8' : '1px solid #444',
                background: !includeBatch2 ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
                color: !includeBatch2 ? '#38bdf8' : '#aaa',
                cursor: 'pointer',
              }}
            >
              Initial Dataset (8 Users)
            </button>
            <button
              type="button"
              onClick={() => setIncludeBatch2(true)}
              style={{
                flex: 1,
                padding: '0.35rem 0.6rem',
                fontSize: '0.76rem',
                borderRadius: '5px',
                border: includeBatch2 ? '1px solid #f59e0b' : '1px solid #444',
                background: includeBatch2 ? 'rgba(245, 158, 11, 0.2)' : 'transparent',
                color: includeBatch2 ? '#f59e0b' : '#aaa',
                cursor: 'pointer',
              }}
            >
              + Ingest Batch 2 (+4 Users)
            </button>
          </div>
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
            <label style={{ fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', color: '#94a3b8' }}>
              Train Split Threshold: <strong style={{ color: '#38bdf8' }}>{trainRatio}%</strong>
            </label>
            <span style={{ fontSize: '0.72rem', color: '#64748b' }}>Test: {100 - trainRatio}%</span>
          </div>
          <input
            type="range"
            min={50}
            max={90}
            step={5}
            value={trainRatio}
            onChange={(e) => setTrainRatio(parseInt(e.target.value, 10))}
            style={{ width: '100%', cursor: 'pointer', accentColor: '#38bdf8' }}
          />
        </div>

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
            Hash Salt Key
          </label>
          <div style={{ display: 'flex', gap: '0.4rem' }}>
            <button
              type="button"
              onClick={() => setSalt('prod_v1')}
              style={{
                flex: 1,
                padding: '0.35rem 0.5rem',
                fontSize: '0.74rem',
                borderRadius: '5px',
                border: salt === 'prod_v1' ? '1px solid #38bdf8' : '1px solid #444',
                background: salt === 'prod_v1' ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
                color: salt === 'prod_v1' ? '#38bdf8' : '#aaa',
                cursor: 'pointer',
              }}
            >
              'prod_v1'
            </button>
            <button
              type="button"
              onClick={() => setSalt('audit_v2')}
              style={{
                flex: 1,
                padding: '0.35rem 0.5rem',
                fontSize: '0.74rem',
                borderRadius: '5px',
                border: salt === 'audit_v2' ? '1px solid #38bdf8' : '1px solid #444',
                background: salt === 'audit_v2' ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
                color: salt === 'audit_v2' ? '#38bdf8' : '#aaa',
                cursor: 'pointer',
              }}
            >
              'audit_v2'
            </button>
          </div>
        </div>
      </div>

      {/* Stability Metrics Banner */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
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
            Random Split Partition Flips
          </span>
          <strong style={{ fontSize: '1.35rem', color: randomFlips > 0 ? '#f43f5e' : '#34d399' }}>
            {includeBatch2 ? `${randomFlips} of 8 Flipped (${Math.round((randomFlips / 8) * 100)}%)` : 'Baseline'}
          </strong>
          <span style={{ display: 'block', fontSize: '0.7rem', color: '#64748b', marginTop: '0.2rem' }}>
            {randomFlips > 0 ? 'Silent test set contamination!' : 'Initial state'}
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
            Deterministic Hash Flips
          </span>
          <strong style={{ fontSize: '1.35rem', color: '#34d399' }}>
            {includeBatch2 ? `${hashedFlips} of 8 Flipped (0.0%)` : 'Baseline'}
          </strong>
          <span style={{ display: 'block', fontSize: '0.7rem', color: '#64748b', marginTop: '0.2rem' }}>
            100% Immutable assignment across reruns
          </span>
        </div>
      </div>

      {/* Side by side comparison table */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '1rem',
          marginBottom: '1.25rem',
        }}
      >
        {/* Left: Random Shuffle Split */}
        <div
          style={{
            padding: '0.85rem',
            borderRadius: '8px',
            background: 'rgba(0, 0, 0, 0.3)',
            border: '1px solid rgba(255, 255, 255, 0.05)',
          }}
        >
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#f43f5e', marginBottom: '0.5rem' }}>
            Method A: Index Shuffle (e.g. train_test_split)
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            {randomPartitions.map((u) => (
              <div
                key={u.id}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '0.35rem 0.55rem',
                  borderRadius: '5px',
                  background: u.batch === 2 ? 'rgba(245, 158, 11, 0.08)' : 'rgba(255, 255, 255, 0.03)',
                  fontSize: '0.74rem',
                }}
              >
                <span style={{ color: u.batch === 2 ? '#fcd34d' : '#cbd5e1' }}>
                  {u.name} {u.batch === 2 && '(NEW)'}
                </span>
                <span
                  style={{
                    fontSize: '0.65rem',
                    fontWeight: 600,
                    padding: '0.1rem 0.35rem',
                    borderRadius: '3px',
                    background: u.split === 'Train' ? 'rgba(56, 189, 248, 0.15)' : 'rgba(244, 63, 94, 0.15)',
                    color: u.split === 'Train' ? '#38bdf8' : '#fda4af',
                  }}
                >
                  {u.split}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Right: Deterministic Entity Hash */}
        <div
          style={{
            padding: '0.85rem',
            borderRadius: '8px',
            background: 'rgba(0, 0, 0, 0.3)',
            border: '1px solid rgba(255, 255, 255, 0.05)',
          }}
        >
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#34d399', marginBottom: '0.5rem' }}>
            Method B: Hash(user_id || salt) % 100
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            {hashedPartitions.map((u) => (
              <div
                key={u.id}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '0.35rem 0.55rem',
                  borderRadius: '5px',
                  background: u.batch === 2 ? 'rgba(52, 211, 153, 0.08)' : 'rgba(255, 255, 255, 0.03)',
                  fontSize: '0.74rem',
                }}
              >
                <span style={{ color: u.batch === 2 ? '#6ee7b7' : '#cbd5e1' }}>
                  {u.name} {u.batch === 2 && '(NEW)'}
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <span style={{ fontSize: '0.68rem', color: '#64748b' }}>Bucket: {u.bucket}</span>
                  <span
                    style={{
                      fontSize: '0.65rem',
                      fontWeight: 600,
                      padding: '0.1rem 0.35rem',
                      borderRadius: '3px',
                      background: u.split === 'Train' ? 'rgba(56, 189, 248, 0.15)' : 'rgba(52, 211, 153, 0.15)',
                      color: u.split === 'Train' ? '#38bdf8' : '#6ee7b7',
                    }}
                  >
                    {u.split}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Analytical Callout */}
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
        <span style={{ fontWeight: 600, color: '#f8fafc' }}>Why Hashing Wins in Production: </span>
        When data grows or pipelines run in distributed environments (Spark, SQL, BigQuery), hashing entity keys ensures
        every worker assigns rows identically without coordinating across partitions. Existing test rows are permanently
        shielded from entering future training sets.
      </div>
    </div>
  )
}
