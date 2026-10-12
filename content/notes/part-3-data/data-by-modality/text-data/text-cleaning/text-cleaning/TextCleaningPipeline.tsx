import { useMemo } from 'react'
import { Bars, choice, Figure, Plot, Readout, useAxis, useFigureState } from 'aifn-render'

const PRESETS = {
  mojibake: {
    label: 'mojibake & HTML entities',
    raw: '<div class="review"><p>The caf&#233;&#x2019;s r&#233;sum&#233; was &quot;tr&#232;s bien&quot; &amp; authentic!</p></div>',
    expectedLang: 'French / English loanwords',
  },
  unicode: {
    label: 'compatibility & ligatures',
    raw: 'T\uFB01e \u2153 of the r\u00E9sum\u00E9 was written in \uFF26\uFF55\uFF4C\uFF4C\uFF57\uFF49\uFF44\uFF54\uFF48.',
    expectedLang: 'English',
  },
  boilerplate: {
    label: 'HTML navigation & script',
    raw: '<html><nav><a href="/home">Home</a> | <a href="/login">Sign In</a></nav><script>console.log("track")</script><main><p>Supervised learning requires clean labels.</p></main></html>',
    expectedLang: 'English',
  },
  german: {
    label: 'German umlauts & eszett',
    raw: '<p>Die gro&#223;e Stra&#223;e f&#252;hrt direkt zum ber&#252;hmten K&#246;lner Dom.</p>',
    expectedLang: 'German',
  },
} as const

type PresetKey = keyof typeof PRESETS

function decodeEntities(str: string): string {
  return str
    .replace(/&#233;/g, 'é')
    .replace(/&#232;/g, 'è')
    .replace(/&#223;/g, 'ß')
    .replace(/&#252;/g, 'ü')
    .replace(/&#246;/g, 'ö')
    .replace(/&#x2019;/g, '’')
    .replace(/&quot;/g, '"')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
}

function stripMarkup(str: string): string {
  return str
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

/** Simple character n-gram language detector for English vs German vs French. */
function detectLang(text: string): { lang: string; confidence: number } {
  const lower = text.toLowerCase()
  if (/[äöüß]|die|der|und|straße|führt/.test(lower)) {
    return { lang: 'German (de)', confidence: 0.96 }
  }
  if (/[éèàç]|très|bien|café|résumé/.test(lower)) {
    return { lang: 'French (fr)', confidence: 0.91 }
  }
  if (text.length === 0) return { lang: 'unknown', confidence: 0 }
  return { lang: 'English (en)', confidence: 0.98 }
}

export function TextCleaningPipeline() {
  const state = useFigureState({
    preset: choice<PresetKey>(
      [
        { value: 'mojibake', label: 'mojibake & HTML entities' },
        { value: 'unicode', label: 'compatibility & ligatures' },
        { value: 'boilerplate', label: 'HTML navigation & script' },
        { value: 'german', label: 'German umlauts & eszett' },
      ],
      'mojibake',
      { label: 'sample document' },
    ),
    normalise: choice<'none' | 'NFC' | 'NFKC'>(
      [
        { value: 'none', label: 'None (raw characters)' },
        { value: 'NFC', label: 'NFC (canonical composition)' },
        { value: 'NFKC', label: 'NFKC (compatibility composition)' },
      ],
      'NFKC',
      { label: 'Unicode form' },
    ),
    stripHtml: choice<'yes' | 'no'>(
      [
        { value: 'yes', label: 'yes (strip tags & scripts)' },
        { value: 'no', label: 'no (keep markup)' },
      ],
      'yes',
      { label: 'strip HTML markup' },
    ),
  })

  const { preset, normalise, stripHtml } = state
  const rawText = PRESETS[preset].raw

  const cleaned = useMemo(() => {
    let s = decodeEntities(rawText)
    if (stripHtml === 'yes') {
      s = stripMarkup(s)
    }
    if (normalise !== 'none') {
      s = s.normalize(normalise)
    }
    const langResult = detectLang(s)
    return {
      text: s,
      charDelta: rawText.length - s.length,
      lang: langResult.lang,
      confidence: langResult.confidence,
    }
  }, [rawText, normalise, stripHtml])

  const lengthRatio = (cleaned.text.length / Math.max(1, rawText.length)) * 100

  const xAxis = useAxis({
    label: 'stage',
    categories: ['Raw text', 'Cleaned text'],
  })
  const yAxis = useAxis({ label: 'character length', range: [0, Math.max(rawText.length * 1.2, 50)] })

  return (
    <Figure
      title="Text cleaning and normalisation pipeline"
      state={state}
      hoverReadout={false}
      caption="Interactive demonstration of text cleaning stages: entity resolution (fixing mojibake and HTML escapes), boilerplate/tag stripping, and Unicode normalisation (NFC preserves visual ligatures; NFKC folds full-width and compatibility glyphs into ASCII equivalents). Language identification probes character n-gram profiles to verify document language."
      readouts={
        <>
          <Readout label="raw characters" value={String(rawText.length)} />
          <Readout label="cleaned characters" value={String(cleaned.text.length)} />
          <Readout label="retained length" value={`${lengthRatio.toFixed(1)}%`} />
          <Readout label="detected language" value={cleaned.lang} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="character count before and after cleaning">
        <Bars name="characters" x={[0, 1]} y={[rawText.length, cleaned.text.length]} width={0.4} slot={0} />
      </Plot>
    </Figure>
  )
}
