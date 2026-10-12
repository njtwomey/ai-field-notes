import { useMemo, useState } from 'react'

interface LanguageSnippet {
  id: string
  name: string
  script: string
  direction: 'ltr' | 'rtl'
  hasWhitespace: boolean
  originalText: string
  englishMeaning: string
  words: string[]
  subwords: string[]
  bytesPerChar: number
  fertility: number // tokens per semantic word
  scriptDistribution: { script: string; count: number; pct: number }[]
}

const SNIPPETS: LanguageSnippet[] = [
  {
    id: 'en',
    name: 'English (Latin script)',
    script: 'Latin',
    direction: 'ltr',
    hasWhitespace: true,
    originalText: 'Statistical machine learning models require clean, verified training datasets.',
    englishMeaning: 'Statistical machine learning models require clean, verified training datasets.',
    words: ['Statistical', 'machine', 'learning', 'models', 'require', 'clean,', 'verified', 'training', 'datasets.'],
    subwords: [
      'Statistical',
      'machine',
      'learning',
      'models',
      'require',
      'clean',
      ',',
      'verified',
      'training',
      'data',
      '##sets',
      '.',
    ],
    bytesPerChar: 1.0,
    fertility: 1.33,
    scriptDistribution: [
      { script: 'Latin', count: 70, pct: 89.7 },
      { script: 'Common/Punct', count: 8, pct: 10.3 },
    ],
  },
  {
    id: 'zh',
    name: 'Chinese (CJK Ideographs)',
    script: 'Han',
    direction: 'ltr',
    hasWhitespace: false,
    originalText: '统计机器学习模型需要经过清洗和验证的训练数据集。',
    englishMeaning: 'Statistical machine learning models require clean, verified training datasets.',
    words: ['统计', '机器学习', '模型', '需要', '经过', '清洗', '和', '验证', '的', '训练', '数据集。'],
    subwords: ['统计', '机器', '学习', '模型', '需要', '经过', '清洗', '和', '验证', '的', '训练', '数据', '集', '。'],
    bytesPerChar: 3.0,
    fertility: 1.27,
    scriptDistribution: [
      { script: 'Han (CJK)', count: 23, pct: 95.8 },
      { script: 'Common/Punct', count: 1, pct: 4.2 },
    ],
  },
  {
    id: 'hi',
    name: 'Hindi (Devanagari script)',
    script: 'Devanagari',
    direction: 'ltr',
    hasWhitespace: true,
    originalText: 'सांख्यिकीय मशीन लर्निंग मॉडल को स्वच्छ और सत्यापित प्रशिक्षण डेटा की आवश्यकता होती है।',
    englishMeaning: 'Statistical machine learning models require clean, verified training datasets.',
    words: [
      'सांख्यिकीय',
      'मशीन',
      'लर्निंग',
      'मॉडल',
      'को',
      'स्वच्छ',
      'और',
      'सत्यापित',
      'प्रशिक्षण',
      'डेटा',
      'की',
      'आवश्यकता',
      'होती',
      'है।',
    ],
    subwords: [
      'सां',
      '##ख्य',
      '##िकीय',
      'मशीन',
      'ल',
      '##र्निंग',
      'मॉडल',
      'को',
      'स्व',
      '##च्छ',
      'और',
      'सत्या',
      '##पित',
      'प्र',
      '##शिक्षण',
      'डेटा',
      'की',
      'आव',
      '##श्यक',
      '##ता',
      'होती',
      'है',
      '।',
    ],
    bytesPerChar: 2.85,
    fertility: 1.64,
    scriptDistribution: [
      { script: 'Devanagari', count: 70, pct: 82.4 },
      { script: 'Common/Space', count: 14, pct: 16.5 },
      { script: 'Punct', count: 1, pct: 1.1 },
    ],
  },
  {
    id: 'ar',
    name: 'Arabic (Right-to-Left script)',
    script: 'Arabic',
    direction: 'rtl',
    hasWhitespace: true,
    originalText: 'تتطلب نماذج التعلم الآلي الإحصائي مجموعات بيانات تدريبية نظيفة ومحققة.',
    englishMeaning: 'Statistical machine learning models require clean, verified training datasets.',
    words: ['تتطلب', 'نماذج', 'التعلم', 'الآلي', 'الإحصائي', 'مجموعات', 'بيانات', 'تدريبية', 'نظيفة', 'ومحققة.'],
    subwords: [
      'تت',
      '##طلب',
      'نماذج',
      'التع',
      '##لم',
      'الآلي',
      'الإ',
      '##حصائي',
      'مجموع',
      '##ات',
      'بيانات',
      'تدريب',
      '##ية',
      'نظيفة',
      'وم',
      '##حققة',
      '.',
    ],
    bytesPerChar: 1.95,
    fertility: 1.7,
    scriptDistribution: [
      { script: 'Arabic', count: 59, pct: 85.5 },
      { script: 'Common/Space', count: 9, pct: 13.0 },
      { script: 'Punct', count: 1, pct: 1.5 },
    ],
  },
  {
    id: 'hinglish',
    name: 'Hinglish (Code-switched & Romanised)',
    script: 'Mixed Latin / Romanised Hindi',
    direction: 'ltr',
    hasWhitespace: true,
    originalText: 'Statistical ML models ko deploy karne ke liye clean verified training data chahiye.',
    englishMeaning: 'Clean verified training data is needed to deploy statistical ML models.',
    words: [
      'Statistical',
      'ML',
      'models',
      'ko',
      'deploy',
      'karne',
      'ke',
      'liye',
      'clean',
      'verified',
      'training',
      'data',
      'chahiye.',
    ],
    subwords: [
      'Statistical',
      'ML',
      'models',
      'ko',
      'deploy',
      'k',
      '##arne',
      'ke',
      'liye',
      'clean',
      'verified',
      'training',
      'data',
      'ch',
      '##ahi',
      '##ye',
      '.',
    ],
    bytesPerChar: 1.0,
    fertility: 1.31,
    scriptDistribution: [
      { script: 'Latin (English roots)', count: 48, pct: 57.8 },
      { script: 'Latin (Romanised Indic)', count: 23, pct: 27.7 },
      { script: 'Common/Punct', count: 12, pct: 14.5 },
    ],
  },
]

export function MultilingualScriptExplorer() {
  const [selectedId, setSelectedId] = useState<string>('zh')
  const [activeTab, setActiveTab] = useState<'tokens' | 'fertility' | 'confusables'>('tokens')

  const snippet = useMemo(() => {
    return SNIPPETS.find((s) => s.id === selectedId) ?? SNIPPETS[0]
  }, [selectedId])

  return (
    <div className="my-6 rounded-lg border border-border bg-card p-4 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
        <div>
          <h4 className="text-text font-semibold">Multilingual Text & Token Fertility Explorer</h4>
          <p className="text-text-muted text-xs">
            Examine script properties, word segmentation barriers, UTF-8 byte density, and subword token fertility.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-text-muted text-xs font-medium">Language:</label>
          <select
            value={selectedId}
            onChange={(e) => setSelectedId(e.target.value)}
            className="bg-surface text-text rounded border border-border px-2 py-1 text-xs shadow-sm"
          >
            {SNIPPETS.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Tabs */}
      <div className="mb-4 flex gap-2 border-b border-border/60 pb-2 text-xs">
        <button
          onClick={() => setActiveTab('tokens')}
          className={`rounded px-3 py-1 font-medium transition-colors ${
            activeTab === 'tokens'
              ? 'bg-primary text-primary-foreground'
              : 'bg-surface hover:bg-surface-elevated text-text'
          }`}
        >
          Segmentation & Subwords
        </button>
        <button
          onClick={() => setActiveTab('fertility')}
          className={`rounded px-3 py-1 font-medium transition-colors ${
            activeTab === 'fertility'
              ? 'bg-primary text-primary-foreground'
              : 'bg-surface hover:bg-surface-elevated text-text'
          }`}
        >
          Cross-Lingual Fertility & Byte Cost
        </button>
        <button
          onClick={() => setActiveTab('confusables')}
          className={`rounded px-3 py-1 font-medium transition-colors ${
            activeTab === 'confusables'
              ? 'bg-primary text-primary-foreground'
              : 'bg-surface hover:bg-surface-elevated text-text'
          }`}
        >
          Homoglyphs & Confusables
        </button>
      </div>

      {activeTab === 'tokens' && (
        <div className="space-y-4">
          {/* Text preview */}
          <div className="bg-surface rounded border border-border p-3">
            <div className="text-text-muted mb-1.5 flex items-center justify-between text-xs">
              <span className="font-semibold tracking-wider uppercase">Original Text ({snippet.script})</span>
              <span className="font-mono text-[11px]">
                Dir: {snippet.direction.toUpperCase()} | Whitespace delimited:{' '}
                {snippet.hasWhitespace ? 'Yes' : 'No (Scriptio Continua)'}
              </span>
            </div>
            <p
              dir={snippet.direction}
              className={`text-text text-base font-medium ${
                snippet.direction === 'rtl' ? 'text-right font-serif' : ''
              }`}
            >
              {snippet.originalText}
            </p>
            <p className="text-text-muted mt-1 text-xs italic">Meaning: &ldquo;{snippet.englishMeaning}&rdquo;</p>
          </div>

          {/* Word Segmentation comparison */}
          <div className="bg-surface/50 rounded border border-border p-3">
            <div className="text-text-muted mb-1.5 flex items-center justify-between text-xs">
              <span className="font-semibold tracking-wider uppercase">
                {snippet.hasWhitespace
                  ? 'Word Segmentation (Whitespace-delimited)'
                  : 'Lexical Word Segmentation (Dictionary/CRF required)'}
              </span>
              <span>{snippet.words.length} semantic words</span>
            </div>
            <div className="flex flex-wrap gap-1.5" dir={snippet.direction}>
              {snippet.words.map((w, idx) => (
                <span
                  key={idx}
                  className="bg-surface-elevated text-text rounded border border-border/80 px-2 py-0.5 font-mono text-xs"
                >
                  {w}
                </span>
              ))}
            </div>
          </div>

          {/* Subwords pill list */}
          <div className="bg-surface/50 rounded border border-border p-3">
            <div className="text-text-muted mb-1.5 flex items-center justify-between text-xs">
              <span className="font-semibold tracking-wider uppercase">Subword Tokenisation Output (BPE)</span>
              <span>
                {snippet.subwords.length} tokens ({snippet.fertility.toFixed(2)} tokens/word)
              </span>
            </div>
            <div className="flex flex-wrap gap-1.5" dir={snippet.direction}>
              {snippet.subwords.map((sw, idx) => {
                const isContinuation = sw.startsWith('##')
                return (
                  <span
                    key={idx}
                    className={`rounded border px-2 py-0.5 font-mono text-xs ${
                      isContinuation
                        ? 'border-dashed border-sky-400 bg-sky-500/10 text-sky-800 dark:text-sky-200'
                        : 'text-text border-border bg-card'
                    }`}
                  >
                    {sw}
                  </span>
                )
              })}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'fertility' && (
        <div className="space-y-4">
          <p className="text-text-muted text-xs">
            Token fertility is the average number of subword tokens produced per semantic word or unit of meaning.
            Underrepresented languages and non-Latin scripts suffer from higher fertility, consuming more of the
            model&apos;s context window and inflating per-document training and inference compute.
          </p>

          <div className="space-y-3">
            {SNIPPETS.map((s) => {
              const isSelected = s.id === snippet.id
              return (
                <div
                  key={s.id}
                  onClick={() => setSelectedId(s.id)}
                  className={`cursor-pointer rounded border p-3 transition-colors ${
                    isSelected
                      ? 'border-primary bg-primary/5 shadow-sm'
                      : 'bg-surface hover:bg-surface-elevated border-border'
                  }`}
                >
                  <div className="mb-1.5 flex items-center justify-between text-xs">
                    <span className="text-text font-semibold">{s.name}</span>
                    <span className="text-text-muted font-mono">
                      {s.subwords.length} tokens / {s.words.length} words ={' '}
                      <strong className="text-primary">{s.fertility.toFixed(2)}x</strong> fertility
                    </span>
                  </div>
                  {/* Progress bar */}
                  <div className="h-2 w-full overflow-hidden rounded-full bg-border">
                    <div
                      className="h-full bg-primary transition-all duration-300"
                      style={{ width: `${Math.min(100, (s.fertility / 2.0) * 100)}%` }}
                    />
                  </div>
                  <div className="text-text-muted mt-2 flex justify-between font-mono text-[11px]">
                    <span>UTF-8 Density: {s.bytesPerChar.toFixed(2)} bytes/char</span>
                    <span>Script: {s.script}</span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {activeTab === 'confusables' && (
        <div className="space-y-3">
          <div className="bg-surface rounded border border-border p-3">
            <h5 className="text-text mb-1 text-xs font-semibold uppercase">Homoglyphs and Script Confusables</h5>
            <p className="text-text-muted text-xs">
              Glyphs from distinct Unicode scripts can appear identical visually while carrying different code points.
              Adversaries exploit this to bypass moderation filters, and optical character recognition (OCR) models
              frequently conflate them.
            </p>
          </div>

          <div className="overflow-x-auto rounded border border-border">
            <table className="w-full text-left text-xs">
              <thead className="bg-surface-elevated text-text border-b border-border font-semibold">
                <tr>
                  <th className="p-2">Visual Glyph</th>
                  <th className="p-2">Latin Code Point</th>
                  <th className="p-2">Cyrillic Confusable</th>
                  <th className="p-2">Greek Confusable</th>
                </tr>
              </thead>
              <tbody className="text-text divide-y divide-border font-mono">
                <tr>
                  <td className="p-2 text-base font-bold">a</td>
                  <td className="p-2">U+0061 (Latin Small A)</td>
                  <td className="p-2">U+0430 (Cyrillic Small A)</td>
                  <td className="text-text-muted p-2">&mdash;</td>
                </tr>
                <tr>
                  <td className="p-2 text-base font-bold">e</td>
                  <td className="p-2">U+0065 (Latin Small E)</td>
                  <td className="p-2">U+0435 (Cyrillic Small Ie)</td>
                  <td className="text-text-muted p-2">&mdash;</td>
                </tr>
                <tr>
                  <td className="p-2 text-base font-bold">o</td>
                  <td className="p-2">U+006F (Latin Small O)</td>
                  <td className="p-2">U+043E (Cyrillic Small O)</td>
                  <td className="p-2">U+03BF (Greek Small Omicron)</td>
                </tr>
                <tr>
                  <td className="p-2 text-base font-bold">p</td>
                  <td className="p-2">U+0070 (Latin Small P)</td>
                  <td className="p-2">U+0440 (Cyrillic Small Er)</td>
                  <td className="p-2">U+03C1 (Greek Small Rho)</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-text-muted text-[11px] italic">
            Cleaning remedy: Unicode skeleton matching (UTS #39) normalises confusable character variants into a
            canonical skeleton representation before deduplication or safety classification.
          </p>
        </div>
      )}

      {/* Footer metadata */}
      <div className="text-text-muted mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-border pt-3 text-[11px]">
        <div>
          Script breakdown:{' '}
          {snippet.scriptDistribution.map((d, i) => (
            <span key={i} className="mr-2">
              <strong>{d.script}</strong>: {d.pct.toFixed(1)}%
            </span>
          ))}
        </div>
        <div className="font-mono">Avg UTF-8 density: {snippet.bytesPerChar.toFixed(2)} bytes / char</div>
      </div>
    </div>
  )
}
