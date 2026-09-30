import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const src = path.join(import.meta.dirname, 'src')

/**
 * The aifn lab: a standalone app for exploring aifn. It depends on aifn (the workspace package) and third-party packages
 * only; nothing is imported from the site.
 */
export default defineConfig({
  root: import.meta.dirname,
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: [{ find: /^@lab\//, replacement: `${src}/` }],
  },
  // A single-page app with path URLs (/<module>/<specimen>): dev and preview serve index.html for any unknown path.
  appType: 'spa',
  server: { port: 5190 },
  preview: { port: 5190 },
})
