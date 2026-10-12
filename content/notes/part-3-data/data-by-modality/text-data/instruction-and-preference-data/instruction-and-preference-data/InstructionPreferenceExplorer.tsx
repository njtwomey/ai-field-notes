import { useMemo, useState } from 'react'

interface MessageTurn {
  role: 'system' | 'user' | 'assistant'
  content: string
  tokens: string[]
  isSupervised: boolean
}

const CONVERSATION: MessageTurn[] = [
  {
    role: 'system',
    content: 'You are an accurate, helpful AI coding and science assistant.',
    tokens: [
      '<|im_start|>',
      'system',
      '\n',
      'You',
      'are',
      'an',
      'accurate',
      ',',
      'helpful',
      'AI',
      'assistant',
      '.',
      '<|im_end|>',
      '\n',
    ],
    isSupervised: false,
  },
  {
    role: 'user',
    content: 'Why does Rayleigh scattering cause the sky to look blue?',
    tokens: [
      '<|im_start|>',
      'user',
      '\n',
      'Why',
      'does',
      'Rayleigh',
      'scattering',
      'cause',
      'the',
      'sky',
      'to',
      'look',
      'blue',
      '?',
      '<|im_end|>',
      '\n',
    ],
    isSupervised: false,
  },
  {
    role: 'assistant',
    content: 'Air molecules scatter short-wavelength blue light much more efficiently than red light.',
    tokens: [
      '<|im_start|>',
      'assistant',
      '\n',
      'Air',
      'molecules',
      'scatter',
      'short',
      '-wavelength',
      'blue',
      'light',
      'much',
      'more',
      'efficiently',
      'than',
      'red',
      'light',
      '.',
      '<|im_end|>',
    ],
    isSupervised: true,
  },
]

export function InstructionPreferenceExplorer() {
  const [activeTab, setActiveTab] = useState<'sft' | 'dpo'>('sft')
  const [beta, setBeta] = useState<number>(0.1)
  const [logRatioWinner, setLogRatioWinner] = useState<number>(0.8)
  const [logRatioLoser, setLogRatioLoser] = useState<number>(-0.4)

  // SFT Token stats
  const sftStats = useMemo(() => {
    let promptTokens = 0
    let targetTokens = 0
    CONVERSATION.forEach((turn) => {
      if (turn.isSupervised) targetTokens += turn.tokens.length
      else promptTokens += turn.tokens.length
    })
    return {
      total: promptTokens + targetTokens,
      prompt: promptTokens,
      target: targetTokens,
    }
  }, [])

  // DPO implicit margin & loss
  const dpoMetrics = useMemo(() => {
    const margin = beta * (logRatioWinner - logRatioLoser)
    const prob = 1 / (1 + Math.exp(-margin))
    const loss = -Math.log(Math.max(1e-12, prob))
    return { margin, prob, loss }
  }, [beta, logRatioWinner, logRatioLoser])

  return (
    <div className="my-6 rounded-lg border border-border bg-card p-4 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
        <div>
          <h4 className="text-text font-semibold">Instruction & Preference Data Inspector</h4>
          <p className="text-text-muted text-xs">
            Inspect SFT prompt-completion loss masking (ChatML) and pairwise DPO implicit reward margin scaling.
          </p>
        </div>
        <div className="flex gap-2 text-xs">
          <button
            onClick={() => setActiveTab('sft')}
            className={`rounded px-3 py-1 font-medium transition-colors ${
              activeTab === 'sft'
                ? 'bg-primary text-primary-foreground'
                : 'bg-surface hover:bg-surface-elevated text-text'
            }`}
          >
            SFT Loss Masking
          </button>
          <button
            onClick={() => setActiveTab('dpo')}
            className={`rounded px-3 py-1 font-medium transition-colors ${
              activeTab === 'dpo'
                ? 'bg-primary text-primary-foreground'
                : 'bg-surface hover:bg-surface-elevated text-text'
            }`}
          >
            Pairwise Preference (DPO)
          </button>
        </div>
      </div>

      {activeTab === 'sft' && (
        <div className="space-y-4">
          <div className="text-text-muted flex flex-wrap items-center justify-between gap-2 text-xs">
            <span className="font-semibold tracking-wider uppercase">ChatML Format Multi-turn Token Sequence</span>
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1.5">
                <span className="h-3 w-3 rounded border border-blue-500/40 bg-blue-500/20" />
                Prompt (Weight 0, -100)
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-3 w-3 rounded border border-emerald-500/40 bg-emerald-500/20" />
                Completion (Weight 1, Loss Target)
              </span>
            </div>
          </div>

          <div className="bg-surface space-y-2.5 rounded border border-border p-3 font-mono text-xs">
            {CONVERSATION.map((turn, tIdx) => (
              <div key={tIdx} className="bg-surface-elevated rounded border border-border/80 p-2">
                <div className="mb-1.5 flex items-center justify-between text-[11px]">
                  <span className="font-bold text-primary uppercase">Role: {turn.role}</span>
                  <span
                    className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                      turn.isSupervised
                        ? 'bg-emerald-500/20 text-emerald-800 dark:text-emerald-300'
                        : 'bg-blue-500/20 text-blue-800 dark:text-blue-300'
                    }`}
                  >
                    {turn.isSupervised ? 'Supervised Loss: ACTIVE (Weight 1.0)' : 'Conditioning Only: MASKED (-100)'}
                  </span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {turn.tokens.map((tok, kIdx) => {
                    const isDelimiter = tok.includes('<|im_')
                    return (
                      <span
                        key={kIdx}
                        className={`rounded px-1.5 py-0.5 text-[11px] ${
                          isDelimiter
                            ? 'border border-amber-500/40 bg-amber-500/20 font-bold text-amber-900 dark:text-amber-200'
                            : turn.isSupervised
                              ? 'border border-emerald-500/30 bg-emerald-500/15 text-emerald-900 dark:text-emerald-200'
                              : 'border border-blue-500/20 bg-blue-500/10 text-blue-900 dark:text-blue-200'
                        }`}
                      >
                        {tok === '\n' ? '↵' : tok}
                      </span>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>

          <div className="bg-surface/50 text-text-muted grid grid-cols-3 gap-3 rounded p-2.5 text-xs">
            <div>
              <span className="text-text block font-semibold">Total Tokens:</span>
              <span className="text-text font-mono">{sftStats.total} tokens</span>
            </div>
            <div>
              <span className="text-text block font-semibold">Prompt / Context (Masked):</span>
              <span className="font-mono text-blue-700 dark:text-blue-300">
                {sftStats.prompt} tokens ({((sftStats.prompt / sftStats.total) * 100).toFixed(0)}%)
              </span>
            </div>
            <div>
              <span className="text-text block font-semibold">Supervised Target:</span>
              <span className="font-mono text-emerald-700 dark:text-emerald-300">
                {sftStats.target} tokens ({((sftStats.target / sftStats.total) * 100).toFixed(0)}%)
              </span>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'dpo' && (
        <div className="space-y-4">
          <div className="bg-surface rounded border border-border p-3">
            <span className="text-text-muted text-xs font-semibold uppercase">Shared Prompt ($x$)</span>
            <p className="text-text mt-1 text-sm font-medium">
              &ldquo;Write a Python function to check whether a string is a palindrome.&rdquo;
            </p>
          </div>

          {/* Winning vs Losing Completion */}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="rounded border border-emerald-500/30 bg-emerald-500/5 p-3 text-xs">
              <div className="mb-1.5 flex items-center justify-between">
                <span className="font-bold text-emerald-800 dark:text-emerald-300">Winning Completion ($y_w$)</span>
                <span className="rounded bg-emerald-500/20 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-800 dark:text-emerald-200">
                  Preferred (+1)
                </span>
              </div>
              <p className="text-text font-mono leading-relaxed">
                def is_palindrome(s: str) -&gt; bool:
                <br />
                &nbsp;&nbsp;&nbsp;&nbsp;&quot;&quot;&quot;Ignore case and non-alphanumerics.&quot;&quot;&quot;
                <br />
                &nbsp;&nbsp;&nbsp;&nbsp;cleaned = [c.lower() for c in s if c.isalnum()]
                <br />
                &nbsp;&nbsp;&nbsp;&nbsp;return cleaned == cleaned[::-1]
              </p>
              <p className="text-text-muted mt-2 text-[11px]">
                Comprehensive: strips punctuation, handles case insensitivity, includes docstring.
              </p>
            </div>

            <div className="rounded border border-rose-500/30 bg-rose-500/5 p-3 text-xs">
              <div className="mb-1.5 flex items-center justify-between">
                <span className="font-bold text-rose-800 dark:text-rose-300">Losing Completion ($y_l$)</span>
                <span className="rounded bg-rose-500/20 px-1.5 py-0.5 text-[10px] font-semibold text-rose-800 dark:text-rose-200">
                  Rejected (-1)
                </span>
              </div>
              <p className="text-text font-mono leading-relaxed">
                def is_palindrome(s):
                <br />
                &nbsp;&nbsp;&nbsp;&nbsp;return s == s[::-1]
              </p>
              <p className="text-text-muted mt-2 text-[11px]">
                Brittle: fails on &ldquo;A man, a plan, a canal: Panama!&rdquo; due to casing and punctuation.
              </p>
            </div>
          </div>

          {/* DPO Dynamics Parameters */}
          <div className="bg-surface/50 rounded border border-border p-3">
            <div className="mb-2 flex items-center justify-between text-xs">
              <span className="text-text font-semibold">Direct Preference Optimisation (DPO) Loss Simulator</span>
              <div className="flex items-center gap-2">
                <span className="text-text-muted">Regularisation $\beta$:</span>
                <input
                  type="range"
                  min={0.02}
                  max={0.5}
                  step={0.02}
                  value={beta}
                  onChange={(e) => setBeta(Number(e.target.value))}
                  className="w-24 accent-primary"
                />
                <span className="font-mono font-bold text-primary">{beta.toFixed(2)}</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
              <div>
                <span className="text-text-muted block">Policy Ratio log(&pi;(y_w)/&pi;_ref)</span>
                <input
                  type="range"
                  min={-1.0}
                  max={2.0}
                  step={0.1}
                  value={logRatioWinner}
                  onChange={(e) => setLogRatioWinner(Number(e.target.value))}
                  className="w-full accent-primary"
                />
                <span className="text-text font-mono font-semibold">{logRatioWinner.toFixed(1)}</span>
              </div>

              <div>
                <span className="text-text-muted block">Policy Ratio log(&pi;(y_l)/&pi;_ref)</span>
                <input
                  type="range"
                  min={-2.0}
                  max={1.0}
                  step={0.1}
                  value={logRatioLoser}
                  onChange={(e) => setLogRatioLoser(Number(e.target.value))}
                  className="w-full accent-primary"
                />
                <span className="text-text font-mono font-semibold">{logRatioLoser.toFixed(1)}</span>
              </div>

              <div>
                <span className="text-text-muted block">Implicit Reward Margin $\Delta r$</span>
                <span className="font-mono text-base font-bold text-primary">{dpoMetrics.margin.toFixed(3)}</span>
              </div>

              <div>
                <span className="text-text-muted block">DPO Objective Loss</span>
                <span className="font-mono text-base font-bold text-emerald-600 dark:text-emerald-400">
                  {dpoMetrics.loss.toFixed(3)}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
