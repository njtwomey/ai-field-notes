import { useMemo, useState } from 'react'

interface TraceStep {
  stepIndex: number
  role: 'user' | 'thought' | 'action' | 'observation' | 'answer'
  badge: string
  content: string
  isTrained: boolean
  description: string
}

const SUCCESS_TRAJECTORY: TraceStep[] = [
  {
    stepIndex: 1,
    role: 'user',
    badge: 'User Query',
    content: 'What was Acme Corp Q3 revenue, and what was its net profit margin?',
    isTrained: false,
    description: 'Human user prompt. External input: masked with -100 in training loss.',
  },
  {
    stepIndex: 2,
    role: 'thought',
    badge: 'Internal Thought',
    content: 'I need to look up Acme Corp Q3 2026 financial filing data using the company search tool.',
    isTrained: true,
    description: 'Autoregressive reasoning trace (CoT). Supervised: trained with loss weight 1.0.',
  },
  {
    stepIndex: 3,
    role: 'action',
    badge: 'Tool Call (Action)',
    content: 'finance_filing_search({"ticker": "ACME", "period": "2026-Q3"})',
    isTrained: true,
    description: 'Structured tool invocation JSON. Supervised: trained with loss weight 1.0.',
  },
  {
    stepIndex: 4,
    role: 'observation',
    badge: 'Tool Observation',
    content: '{"revenue_usd": 12500000000, "net_income_usd": 2500000000, "currency": "USD"}',
    isTrained: false,
    description: 'External sandbox/API response. Environment feedback: masked with -100 in loss.',
  },
  {
    stepIndex: 5,
    role: 'thought',
    badge: 'Internal Thought',
    content: 'Revenue is $12.5B and net income is $2.5B. Profit margin = 2.5 / 12.5 = 20.0%.',
    isTrained: true,
    description: 'Synthesis & calculation step. Supervised: trained with loss weight 1.0.',
  },
  {
    stepIndex: 6,
    role: 'answer',
    badge: 'Final Response',
    content: 'In Q3 2026, Acme Corp reported $12.5 billion in revenue and a 20.0% net profit margin.',
    isTrained: true,
    description: 'Final conversational output delivered to user. Supervised: trained with loss weight 1.0.',
  },
]

const RECOVERY_TRAJECTORY: TraceStep[] = [
  {
    stepIndex: 1,
    role: 'user',
    badge: 'User Query',
    content: 'Run unit test suite for the auth module.',
    isTrained: false,
    description: 'External user prompt: masked with -100 in training loss.',
  },
  {
    stepIndex: 2,
    role: 'thought',
    badge: 'Internal Thought',
    content: 'I will execute pytest on tests/test_auth.py.',
    isTrained: true,
    description: 'Reasoning turn. Supervised with loss weight 1.0.',
  },
  {
    stepIndex: 3,
    role: 'action',
    badge: 'Tool Call (Action)',
    content: 'bash_run({"command": "pytest tests/test_auth.py"})',
    isTrained: true,
    description: 'Action execution. Supervised with loss weight 1.0.',
  },
  {
    stepIndex: 4,
    role: 'observation',
    badge: 'Tool Observation (Error)',
    content: 'ModuleNotFoundError: No module named "jwt". 1 failed in 0.05s.',
    isTrained: false,
    description: 'Environment failure response. Masked with -100 in training loss.',
  },
  {
    stepIndex: 5,
    role: 'thought',
    badge: 'Self-Reflection',
    content: 'PyJWT is missing from virtualenv. I must install PyJWT before retrying test.',
    isTrained: true,
    description: 'Self-correction reasoning trace. Teaches model error recovery dynamics.',
  },
  {
    stepIndex: 6,
    role: 'action',
    badge: 'Recovery Action',
    content: 'bash_run({"command": "pip install pyjwt && pytest tests/test_auth.py"})',
    isTrained: true,
    description: 'Corrected recovery tool call. Supervised with loss weight 1.0.',
  },
  {
    stepIndex: 7,
    role: 'observation',
    badge: 'Tool Observation',
    content: 'Successfully installed pyjwt. 4 passed in 0.22s.',
    isTrained: false,
    description: 'Environment verification output: masked with -100.',
  },
  {
    stepIndex: 8,
    role: 'answer',
    badge: 'Final Response',
    content: 'Installed missing pyjwt dependency; all 4 auth tests are now passing.',
    isTrained: true,
    description: 'Final status delivery. Supervised with loss weight 1.0.',
  },
]

export function AgentTraceVisualizer() {
  const [scenario, setScenario] = useState<'success' | 'recovery'>('success')
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(6)
  const [showMasking, setShowMasking] = useState<boolean>(true)

  const steps = scenario === 'success' ? SUCCESS_TRAJECTORY : RECOVERY_TRAJECTORY
  const maxStep = steps.length

  const activeStep = useMemo(() => {
    const clamped = Math.min(currentStepIndex, maxStep)
    return steps.find((s) => s.stepIndex === clamped) ?? steps[0]
  }, [steps, currentStepIndex, maxStep])

  const stats = useMemo(() => {
    let trainedCount = 0
    let maskedCount = 0
    steps.forEach((s) => {
      if (s.isTrained) trainedCount++
      else maskedCount++
    })
    return { trainedCount, maskedCount, total: steps.length }
  }, [steps])

  return (
    <div className="my-6 rounded-lg border border-border bg-card p-4 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
        <div>
          <h4 className="text-text font-semibold">Agent Trajectory & Training Mask Visualizer</h4>
          <p className="text-text-muted text-xs">
            Step through sequential tool-calling traces and inspect observation vs reasoning loss supervision.
          </p>
        </div>
        <div className="flex gap-2 text-xs">
          <button
            onClick={() => {
              setScenario('success')
              setCurrentStepIndex(6)
            }}
            className={`rounded px-2.5 py-1 font-medium transition-colors ${
              scenario === 'success'
                ? 'bg-primary text-primary-foreground'
                : 'bg-surface hover:bg-surface-elevated text-text'
            }`}
          >
            Successful API Trace
          </button>
          <button
            onClick={() => {
              setScenario('recovery')
              setCurrentStepIndex(8)
            }}
            className={`rounded px-2.5 py-1 font-medium transition-colors ${
              scenario === 'recovery'
                ? 'bg-primary text-primary-foreground'
                : 'bg-surface hover:bg-surface-elevated text-text'
            }`}
          >
            Error-Recovery Loop
          </button>
        </div>
      </div>

      {/* Trajectory Step Scrubber */}
      <div className="bg-surface/50 mb-4 rounded border border-border p-3">
        <div className="mb-2 flex items-center justify-between text-xs">
          <div className="flex items-center gap-2">
            <span className="text-text font-semibold">Trajectory Progress:</span>
            <span className="font-mono font-bold text-primary">
              Step {activeStep.stepIndex} of {maxStep}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-text-muted flex cursor-pointer items-center gap-1.5 text-xs">
              <input
                type="checkbox"
                checked={showMasking}
                onChange={(e) => setShowMasking(e.target.checked)}
                className="rounded accent-primary"
              />
              Show Loss Masks
            </label>
          </div>
        </div>
        <input
          type="range"
          min={1}
          max={maxStep}
          value={Math.min(currentStepIndex, maxStep)}
          onChange={(e) => setCurrentStepIndex(Number(e.target.value))}
          className="w-full accent-primary"
        />
        <div className="text-text-muted mt-1 flex justify-between font-mono text-[10px]">
          <span>Start (User Prompt)</span>
          <span>Final Outcome</span>
        </div>
      </div>

      {/* Step stream view */}
      <div className="bg-surface space-y-2.5 rounded border border-border p-3">
        {steps.slice(0, currentStepIndex).map((s) => {
          let roleBg = 'bg-surface-elevated border-border text-text'
          let maskBadge = s.isTrained ? (
            <span className="rounded bg-emerald-500/20 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-800 dark:text-emerald-300">
              Target Loss: Active (1.0)
            </span>
          ) : (
            <span className="rounded bg-blue-500/20 px-1.5 py-0.5 text-[10px] font-semibold text-blue-800 dark:text-blue-300">
              Conditioning: Masked (-100)
            </span>
          )

          if (s.role === 'thought') {
            roleBg = 'bg-purple-500/5 border-purple-500/30'
          } else if (s.role === 'action') {
            roleBg = 'bg-amber-500/5 border-amber-500/30 font-mono'
          } else if (s.role === 'observation') {
            roleBg = 'bg-sky-500/5 border-sky-500/30 font-mono text-[11px]'
          } else if (s.role === 'answer') {
            roleBg = 'bg-emerald-500/5 border-emerald-500/30 font-medium'
          }

          return (
            <div key={s.stepIndex} className={`rounded border p-2.5 transition-all ${roleBg}`}>
              <div className="mb-1 flex items-center justify-between text-xs">
                <span className="text-text font-bold">
                  Step {s.stepIndex}: {s.badge}
                </span>
                {showMasking && maskBadge}
              </div>
              <p className="text-text text-xs leading-relaxed">{s.content}</p>
              <p className="text-text-muted mt-1 text-[11px] italic">{s.description}</p>
            </div>
          )
        })}
      </div>

      {/* Trajectory Metadata Stats */}
      <div className="bg-surface/50 text-text-muted mt-4 grid grid-cols-3 gap-3 rounded p-2.5 text-xs">
        <div>
          <span className="text-text block font-semibold">Total Trajectory Steps:</span>
          <span className="text-text font-mono">{stats.total} execution steps</span>
        </div>
        <div>
          <span className="text-text block font-semibold">Supervised Turns (Thoughts/Actions):</span>
          <span className="font-mono font-bold text-emerald-700 dark:text-emerald-300">{stats.trainedCount} turns</span>
        </div>
        <div>
          <span className="text-text block font-semibold">Environment / User Masks:</span>
          <span className="font-mono font-bold text-blue-700 dark:text-blue-300">{stats.maskedCount} turns (-100)</span>
        </div>
      </div>
    </div>
  )
}
