import React, { useState } from 'react'

type IdType = 'autoincrement' | 'ulid' | 'sharded'
type SplitPolicy = 'random' | 'grouped' | 'engineered'

export const IdentifierFeaturesVisualizer: React.FC = () => {
  const [selectedIdType, setSelectedIdType] = useState<IdType>('ulid')
  const [splitPolicy, setSplitPolicy] = useState<SplitPolicy>('random')

  const idExamples = {
    autoincrement: {
      raw: '1,428,571',
      anatomy: [
        {
          label: 'Auto-Increment Counter',
          span: '1428571',
          desc: 'Monotonically strictly increasing integer. Encodes exact chronological sequence of database insert!',
        },
      ],
      hiddenLeak:
        'A tree split on "id > 1,200,000" acts as an illicit temporal leak, splitting on the date of record creation rather than user behavior.',
    },
    ulid: {
      raw: '01HRZ3NDEKTSV4RRFFQ69G5FAV',
      anatomy: [
        {
          label: 'Unix Timestamp (48 bits)',
          span: '01HRZ3NDEK',
          desc: 'Millisecond-precision epoch time (March 2024). Encodes time drift directly into the key.',
        },
        {
          label: 'Machine/Random Entropy (80 bits)',
          span: 'TSV4RRFFQ69G5FAV',
          desc: 'Random cryptographic entropy ensuring distributed uniqueness across server clusters.',
        },
      ],
      hiddenLeak:
        'Sorting or splitting on ULID / UUIDv1 features leaks record timestamp, invalidating cross-validation.',
    },
    sharded: {
      raw: 'TX-EU-CENTRAL-09412',
      anatomy: [
        { label: 'Entity Prefix', span: 'TX', desc: 'Transaction record marker.' },
        {
          label: 'Data Center Shard',
          span: 'EU-CENTRAL',
          desc: 'Regional server partition. Directly reveals geographical site of origin.',
        },
        { label: 'Local Sequence', span: '09412', desc: 'Shard-local counter.' },
      ],
      hiddenLeak:
        'Model learns to associate EU-CENTRAL directly with outcome distributions rather than discovering real demographic features.',
    },
  }

  const activeId = idExamples[selectedIdType]

  // Simulation outcomes under different split policies
  const policyOutcomes = {
    random: {
      trainAcc: '99.4%',
      testAcc: '58.2%',
      gap: '41.2% Overfitting Gap (Memorization Collapse)',
      desc: 'Random row shuffle allows the same customer_id to appear in both training and test folds. The model memorizes customer-specific quirks instead of generalizable patterns.',
      warning: 'High false optimism during cross-validation!',
    },
    grouped: {
      trainAcc: '84.1%',
      testAcc: '81.6%',
      gap: '2.5% Generalization Gap',
      desc: 'GroupKFold guarantees that all transactions from customer C belong strictly to train or strictly to test. The model is forced to evaluate genuinely unseen entities.',
      warning: 'Accurate evaluation of generalization to new customers.',
    },
    engineered: {
      trainAcc: '86.8%',
      testAcc: '85.4%',
      gap: '1.4% Generalization Gap',
      desc: 'Customer ID dropped entirely. Replaced with domain aggregates: historical order count, mean transaction amount, and days since last purchase.',
      warning: 'Production-grade feature architecture.',
    },
  }

  const outcome = policyOutcomes[splitPolicy]

  return (
    <div className="not-prose my-6 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-col gap-1 border-b border-border pb-4">
        <h3 className="text-base font-semibold text-foreground">
          Identifier Features: Hidden Information &amp; Leakage Forensics
        </h3>
        <p className="text-xs text-muted-foreground">
          Primary keys, customer IDs, and session tokens are database indexes, not predictive signals. Deconstruct
          hidden timestamps, server shards, and grouped split safeguards.
        </p>
      </div>

      {/* ID Anatomy Selector */}
      <div className="mb-5 grid grid-cols-1 gap-3 text-xs md:grid-cols-3">
        {[
          { id: 'ulid', title: 'Snowflake / ULID Token', desc: '01HRZ3NDEKTSV4RRFFQ69G5FAV' },
          { id: 'autoincrement', title: 'SQL Auto-Increment Key', desc: 'id = 1,428,571' },
          { id: 'sharded', title: 'Sharded Business Key', desc: 'TX-EU-CENTRAL-09412' },
        ].map((item) => (
          <button
            key={item.id}
            onClick={() => setSelectedIdType(item.id as IdType)}
            className={`rounded-lg border p-3 text-left transition-all ${
              selectedIdType === item.id
                ? 'border-primary bg-primary font-semibold text-primary-foreground shadow-xs'
                : 'border-border bg-muted/40 text-muted-foreground hover:bg-muted'
            }`}
          >
            <div className="text-xs font-bold">{item.title}</div>
            <div className="mt-0.5 truncate font-mono text-[10px] opacity-80">{item.desc}</div>
          </button>
        ))}
      </div>

      {/* ID Anatomy Inspection Box */}
      <div className="mb-5 rounded-lg border border-border bg-card/60 p-4">
        <div className="mb-2 flex items-center justify-between text-xs">
          <span className="font-semibold text-foreground">Decoded Token Anatomy</span>
          <span className="font-mono text-xs font-bold text-primary">{activeId.raw}</span>
        </div>

        <div className="mb-3 space-y-2">
          {activeId.anatomy.map((part, idx) => (
            <div key={idx} className="rounded border border-border/70 bg-muted/20 p-2.5 text-xs">
              <div className="mb-0.5 flex items-center justify-between">
                <span className="font-semibold text-foreground">{part.label}</span>
                <code className="py-0.2 rounded bg-muted/60 px-1.5 font-mono text-[10px] font-bold text-primary">
                  {part.span}
                </code>
              </div>
              <p className="text-[11px] leading-normal text-muted-foreground">{part.desc}</p>
            </div>
          ))}
        </div>

        <div className="flex items-center gap-2 rounded border border-amber-500/20 bg-amber-500/10 p-2.5 text-xs">
          <span className="inline-block h-2 w-2 flex-shrink-0 rounded-full bg-amber-500"></span>
          <span className="text-[11px] text-card-foreground">
            <strong>Hidden Leakage Risk:</strong> {activeId.hiddenLeak}
          </span>
        </div>
      </div>

      {/* Splitting Strategy Simulator */}
      <div className="rounded-lg border border-border bg-card/60 p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <span className="text-xs font-semibold tracking-wider text-foreground uppercase">
            Validation Strategy: Handling Customer / Device IDs
          </span>
          <div className="flex items-center gap-1.5 text-xs">
            {[
              { id: 'random', label: '1. Naive Random Shuffle' },
              { id: 'grouped', label: '2. Grouped Split (GroupKFold)' },
              { id: 'engineered', label: '3. Drop ID + Feature Aggregates' },
            ].map((pol) => (
              <button
                key={pol.id}
                onClick={() => setSplitPolicy(pol.id as SplitPolicy)}
                className={`rounded px-2.5 py-1 text-[11px] font-medium transition-colors ${
                  splitPolicy === pol.id
                    ? 'bg-foreground font-bold text-background'
                    : 'bg-muted/40 text-muted-foreground hover:bg-muted'
                }`}
              >
                {pol.label}
              </button>
            ))}
          </div>
        </div>

        <div className="mb-3 grid grid-cols-1 gap-3 text-xs sm:grid-cols-3">
          <div className="rounded border border-border bg-muted/20 p-2.5">
            <span className="block text-[10px] text-muted-foreground">Train Set Accuracy:</span>
            <div className="mt-0.5 font-mono text-lg font-bold text-foreground">{outcome.trainAcc}</div>
          </div>
          <div className="rounded border border-border bg-muted/20 p-2.5">
            <span className="block text-[10px] text-muted-foreground">Test Set Accuracy:</span>
            <div className="mt-0.5 font-mono text-lg font-bold text-foreground">{outcome.testAcc}</div>
          </div>
          <div
            className={`rounded border p-2.5 ${
              splitPolicy === 'random' ? 'border-rose-500/20 bg-rose-500/10' : 'border-emerald-500/20 bg-emerald-500/10'
            }`}
          >
            <span className="block text-[10px] text-muted-foreground">Generalization Gap:</span>
            <div
              className={`mt-0.5 font-mono text-xs font-bold ${
                splitPolicy === 'random' ? 'text-rose-600 dark:text-rose-400' : 'text-emerald-600 dark:text-emerald-400'
              }`}
            >
              {outcome.gap}
            </div>
          </div>
        </div>

        <p className="text-[11px] leading-relaxed text-muted-foreground">{outcome.desc}</p>
      </div>
    </div>
  )
}
