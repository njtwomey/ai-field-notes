import React, { useState, useMemo } from 'react'

// Deterministic 32-bit Murmur-like hash function for demonstration
function stringHash(str: string, seed: number = 0): number {
  let h = seed ^ 0x12345678
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 0x5bd1e995)
    h ^= h >>> 15
  }
  return h >>> 0
}

export const FeatureHashingVisualizer: React.FC = () => {
  const [numBuckets, setNumBuckets] = useState<number>(8)
  const [useSign, setUseSign] = useState<boolean>(true)
  const [activeTokens, setActiveTokens] = useState<string[]>([
    'apple',
    'google',
    'amazon',
    'netflix',
    'meta',
    'microsoft',
    'tesla',
    'nvidia',
  ])
  const [customInput, setCustomInput] = useState<string>('')

  // Hash each active token into [0, numBuckets - 1] and sign in {-1, +1}
  const hashedTokens = useMemo(() => {
    return activeTokens.map((tok) => {
      const rawHash = stringHash(tok, 42)
      const bucket = rawHash % numBuckets
      const signHash = stringHash(tok, 1337)
      const sign = useSign ? (signHash % 2 === 0 ? 1 : -1) : 1
      return {
        token: tok,
        bucket,
        sign,
      }
    })
  }, [activeTokens, numBuckets, useSign])

  // Aggregate into buckets
  const buckets = useMemo(() => {
    const arr = Array.from({ length: numBuckets }, (_, i) => ({
      bucketId: i,
      tokens: [] as { token: string; sign: number }[],
      netValue: 0,
    }))

    hashedTokens.forEach((item) => {
      arr[item.bucket].tokens.push({ token: item.token, sign: item.sign })
      arr[item.bucket].netValue += item.sign
    })

    return arr
  }, [hashedTokens, numBuckets])

  // Compute collision metrics
  const metrics = useMemo(() => {
    const occupied = buckets.filter((b) => b.tokens.length > 0).length
    const collidedBuckets = buckets.filter((b) => b.tokens.length > 1).length
    const totalCollisions = activeTokens.length - occupied
    const collisionRate = activeTokens.length > 0 ? Math.round((totalCollisions / activeTokens.length) * 100) : 0

    return {
      totalTokens: activeTokens.length,
      occupied,
      collidedBuckets,
      totalCollisions,
      collisionRate,
    }
  }, [buckets, activeTokens])

  const handleAddToken = () => {
    const trimmed = customInput.trim()
    if (trimmed && !activeTokens.includes(trimmed)) {
      setActiveTokens([...activeTokens, trimmed])
      setCustomInput('')
    }
  }

  const handleRemoveToken = (tok: string) => {
    setActiveTokens(activeTokens.filter((t) => t !== tok))
  }

  return (
    <div className="not-prose my-6 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-col gap-1 border-b border-border pb-4">
        <h3 className="text-base font-semibold text-foreground">Feature Hashing (The Hashing Trick) Laboratory</h3>
        <p className="text-xs text-muted-foreground">
          Map arbitrary-cardinality tokens to a fixed number of buckets <em>B</em> without storing a vocabulary. Observe
          how a second sign hash &xi;(s) &isin; &#123;-1, +1&#125; renders collision interference zero-mean.
        </p>
      </div>

      {/* Control Bar */}
      <div className="mb-5 grid grid-cols-1 gap-4 text-xs md:grid-cols-4">
        <div>
          <label className="mb-1 block font-medium text-foreground">Hash Table Size ($B$):</label>
          <div className="flex gap-1.5">
            {[4, 8, 16, 32].map((b) => (
              <button
                key={b}
                onClick={() => setNumBuckets(b)}
                className={`flex-1 rounded border px-1.5 py-1 font-mono text-xs font-medium transition-colors ${
                  numBuckets === b
                    ? 'border-primary bg-primary text-primary-foreground'
                    : 'border-transparent bg-muted/40 text-muted-foreground hover:bg-muted'
                }`}
              >
                {b}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="mb-1 block font-medium text-foreground">Sign Hash (&xi;(s)):</label>
          <div className="flex gap-1.5">
            {[
              { val: true, label: 'Signed (+/- 1, Unbiased)' },
              { val: false, label: 'Unsigned (+1, Additive)' },
            ].map((opt) => (
              <button
                key={String(opt.val)}
                onClick={() => setUseSign(opt.val)}
                className={`flex-1 rounded border px-2 py-1 text-[11px] font-medium transition-colors ${
                  useSign === opt.val
                    ? 'border-primary bg-primary text-primary-foreground'
                    : 'border-transparent bg-muted/40 text-muted-foreground hover:bg-muted'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        <div className="md:col-span-2">
          <label className="mb-1 block font-medium text-foreground">Test New Token (Online O(1)):</label>
          <div className="flex gap-2">
            <input
              type="text"
              value={customInput}
              onChange={(e) => setCustomInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAddToken()}
              placeholder="e.g. unseen_user_99"
              className="flex-1 rounded border border-border bg-muted/20 px-2.5 py-1 font-mono text-xs"
            />
            <button
              onClick={handleAddToken}
              className="rounded bg-foreground px-3 py-1 font-medium text-background transition-opacity hover:opacity-90"
            >
              Hash Token
            </button>
          </div>
        </div>
      </div>

      {/* Active Token Pill Bar */}
      <div className="mb-4">
        <span className="mb-1.5 block text-[10px] font-semibold tracking-wider text-muted-foreground uppercase">
          Active Tokens ({activeTokens.length} total) — Click to remove:
        </span>
        <div className="flex flex-wrap gap-1.5">
          {activeTokens.map((tok) => (
            <button
              key={tok}
              onClick={() => handleRemoveToken(tok)}
              className="flex items-center gap-1 rounded-full border border-border/80 bg-muted px-2 py-0.5 font-mono text-[11px] transition-colors hover:bg-rose-500/20 hover:text-rose-600"
            >
              <span>{tok}</span>
              <span className="text-muted-foreground hover:text-foreground">&times;</span>
            </button>
          ))}
        </div>
      </div>

      {/* Hash Buckets Visual Grid */}
      <div className="mb-5 rounded-lg border border-border bg-card/50 p-4">
        <div className="mb-3 flex items-center justify-between text-xs">
          <span className="font-semibold text-foreground">
            Resulting Fixed Hash Vector &phi;(x) &isin; R^{numBuckets}
          </span>
          <div className="flex items-center gap-3 text-[11px]">
            <span className="font-mono text-muted-foreground">
              Collided Buckets: <span className="font-bold text-amber-500">{metrics.collidedBuckets}</span> /{' '}
              {numBuckets}
            </span>
            <span className="font-mono text-muted-foreground">
              Net Collisions: <span className="font-bold text-rose-500">{metrics.totalCollisions}</span>
            </span>
          </div>
        </div>

        <div
          className={`grid gap-2 text-xs ${
            numBuckets <= 8 ? 'grid-cols-2 sm:grid-cols-4' : 'grid-cols-2 sm:grid-cols-4 lg:grid-cols-8'
          }`}
        >
          {buckets.map((b) => {
            const hasCollision = b.tokens.length > 1
            const isEmpty = b.tokens.length === 0

            return (
              <div
                key={b.bucketId}
                className={`flex flex-col justify-between rounded-lg border p-2.5 transition-colors ${
                  isEmpty
                    ? 'border-border/40 bg-muted/10 opacity-60'
                    : hasCollision
                      ? 'border-amber-500/40 bg-amber-500/5'
                      : 'border-primary/30 bg-primary/5'
                }`}
              >
                <div>
                  <div className="mb-1 flex items-center justify-between">
                    <span className="font-mono text-[10px] text-muted-foreground">Index [{b.bucketId}]</span>
                    <span
                      className={`font-mono text-xs font-bold ${
                        b.netValue > 0 ? 'text-emerald-500' : b.netValue < 0 ? 'text-rose-500' : 'text-muted-foreground'
                      }`}
                    >
                      {b.netValue > 0 ? `+${b.netValue}` : b.netValue}
                    </span>
                  </div>

                  <div className="my-1 space-y-1">
                    {b.tokens.map((t, idx) => (
                      <div
                        key={idx}
                        className="flex justify-between truncate rounded border border-border/50 bg-muted/40 px-1.5 py-0.5 font-mono text-[10px]"
                      >
                        <span className="truncate">{t.token}</span>
                        <span className={`ml-1 font-bold ${t.sign > 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                          {t.sign > 0 ? '+' : '-'}
                        </span>
                      </div>
                    ))}
                    {isEmpty && <span className="text-[10px] text-muted-foreground italic">empty</span>}
                  </div>
                </div>

                {hasCollision && (
                  <span className="mt-1 block text-[9px] font-semibold text-amber-500">
                    {useSign ? 'Signed balance' : 'Additive sum'}
                  </span>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Theoretical Proof Guarantee Box */}
      <div className="grid grid-cols-1 gap-3 rounded-lg border border-border bg-muted/30 p-3.5 text-xs md:grid-cols-2">
        <div>
          <span className="mb-1 block font-semibold text-foreground">
            Weinberger (2009) Unbiased Inner Product Theorem:
          </span>
          <code className="block rounded bg-muted/60 p-1.5 font-mono text-xs text-primary">
            E[&lang;&phi;(u), &phi;(v)&rang;] = &lang;u, v&rang;
          </code>
          <p className="mt-1 text-[10px] leading-relaxed text-muted-foreground">
            Because sign hash &xi;(s) has E[&xi;] = 0, cross-terms between collided distinct tokens s &ne; s' cancel in
            expectation: E[&xi;(s)&xi;(s')] = 0.
          </p>
        </div>

        <div>
          <span className="mb-1 block font-semibold text-foreground">
            Trade-off: Zero Vocabulary vs Irreversibility
          </span>
          <p className="text-[11px] leading-relaxed text-muted-foreground">
            Hashing requires zero RAM and never fails on unseen categories. However, hash functions are non-invertible:
            after training, you cannot inspect which original string caused a large regression weight w_j.
          </p>
        </div>
      </div>
    </div>
  )
}
