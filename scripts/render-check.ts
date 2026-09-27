/**
 * Render note pages in headless Chrome and report crashes, failed figures and maths errors.
 *
 *   node scripts/render-check.ts <slug> [<slug> ...]      # against the dev server on --port (default 5180)
 *   node scripts/render-check.ts --port 5173 matrix-profile
 *   node scripts/render-check.ts --save /tmp/dom k-means   # also write each page's DOM to <dir>/<slug>.html
 *
 * This is the only sanctioned way to render pages headlessly. Pages run one at a time, and each Chrome is killed after
 * a hard timeout: `--virtual-time-budget` alone can wait forever on the dev server's open HMR socket, and ad-hoc loops
 * without a kill left browsers stalled for an hour. A timeout is reported as such, never retried.
 */
import { spawn } from 'node:child_process'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const TIMEOUT_MS = 30_000

const args = process.argv.slice(2)
let port = 5180
let save: string | undefined
const slugs: string[] = []
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--port') port = Number(args[++i])
  else if (args[i] === '--save') save = args[++i]
  else slugs.push(args[i])
}
if (slugs.length === 0) {
  console.error('usage: node scripts/render-check.ts [--port N] <slug> [<slug> ...]')
  process.exit(2)
}

type Result = { html: string; timedOut: boolean }

function dump(url: string): Promise<Result> {
  const profile = mkdtempSync(join(tmpdir(), 'render-check-'))
  return new Promise((resolve) => {
    const chrome = spawn(
      CHROME,
      [
        '--headless=new',
        '--disable-gpu',
        `--user-data-dir=${profile}`,
        '--window-size=1400,20000',
        '--virtual-time-budget=10000',
        '--dump-dom',
        url,
      ],
      { stdio: ['ignore', 'pipe', 'ignore'] },
    )
    let html = ''
    let timedOut = false
    // Chrome often stays alive after printing the DOM, so a complete document ends the run.
    chrome.stdout.on('data', (chunk: Buffer) => {
      html += chunk.toString()
      if (html.includes('</html>')) chrome.kill('SIGKILL')
    })
    const timer = setTimeout(() => {
      timedOut = true
      chrome.kill('SIGKILL')
    }, TIMEOUT_MS)
    chrome.on('close', () => {
      clearTimeout(timer)
      rmSync(profile, { recursive: true, force: true })
      resolve({ html, timedOut })
    })
  })
}

const count = (html: string, pattern: RegExp) => html.match(pattern)?.length ?? 0

let failures = 0
for (const slug of slugs) {
  const { html, timedOut } = await dump(`http://localhost:${port}/ai-field-notes/n/${slug}`)
  if (save) {
    mkdirSync(save, { recursive: true })
    writeFileSync(join(save, `${slug}.html`), html)
  }
  const problems: string[] = []
  if (timedOut) problems.push(`timed out after ${TIMEOUT_MS / 1000} s`)
  else {
    if (/<h1[^>]*>Not found<\/h1>/.test(html)) problems.push('no such note')
    else if (!/<h1[^>]*>/.test(html)) problems.push('no title (page crashed or did not load)')
    const failed = count(html, /This figure failed to render/g)
    if (failed) problems.push(`${failed} failed figure(s)`)
    const maths = count(html, /katex-error/g)
    if (maths) problems.push(`${maths} maths error(s)`)
  }
  const charts = count(html, /_echarts_instance_/g)
  if (problems.length) failures++
  console.log(
    `${problems.length ? 'FAIL' : 'ok  '}  ${slug}  (${charts} charts)${problems.length ? '  ' + problems.join('; ') : ''}`,
  )
}
process.exit(failures ? 1 : 0)
