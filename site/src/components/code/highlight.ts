import type { HighlighterCore } from 'shiki/core'

/** Languages the site highlights. Add a language here and nowhere else. */
const languages = {
  python: () => import('@shikijs/langs/python'),
  bash: () => import('@shikijs/langs/bash'),
  toml: () => import('@shikijs/langs/toml'),
  json: () => import('@shikijs/langs/json'),
  typescript: () => import('@shikijs/langs/typescript'),
  tsx: () => import('@shikijs/langs/tsx'),
}
export type Language = keyof typeof languages

const aliases: Record<string, Language> = { py: 'python', sh: 'bash', shell: 'bash', ts: 'typescript' }

export function resolveLanguage(lang: string | undefined): Language | undefined {
  if (!lang) return undefined
  return lang in languages ? (lang as Language) : aliases[lang]
}

export function languageForPath(path: string): Language | undefined {
  return resolveLanguage(path.split('.').pop())
}

let highlighter: Promise<HighlighterCore> | undefined

function load(): Promise<HighlighterCore> {
  highlighter ??= (async () => {
    const [{ createHighlighterCore }, { createJavaScriptRegexEngine }] = await Promise.all([
      import('shiki/core'),
      import('shiki/engine/javascript'),
    ])
    return createHighlighterCore({
      themes: [import('@shikijs/themes/github-light'), import('@shikijs/themes/github-dark')],
      langs: Object.values(languages).map((l) => l()),
      engine: createJavaScriptRegexEngine(),
    })
  })()
  return highlighter
}

/** HTML with both light and dark token colours as CSS variables; index.css picks one per mode. */
export async function highlight(code: string, lang: Language | undefined): Promise<string> {
  const h = await load()
  return h.codeToHtml(code, {
    lang: lang ?? 'text',
    themes: { light: 'github-light', dark: 'github-dark' },
    defaultColor: false,
  })
}
