import path from 'node:path'
import mdx from '@mdx-js/rollup'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import rehypeKatex from 'rehype-katex'
import rehypeSlug from 'rehype-slug'
import remarkFrontmatter from 'remark-frontmatter'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkMdxFrontmatter from 'remark-mdx-frontmatter'
import { defineConfig } from 'vite'
import { macros } from './content/macros.ts'
import { contentIndex } from './plugins/content-index.ts'

const contentDir = path.resolve(import.meta.dirname, 'content')

export default defineConfig({
  root: 'site',
  // Served from https://<user>.github.io/ml/. Change here only; the router and asset URLs read import.meta.env.BASE_URL.
  base: '/ml/',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    // ECharts alone is ~640 kB minified (215 kB gzip) and is split into its own cached chunk.
    chunkSizeWarningLimit: 700,
    rolldownOptions: {
      output: {
        // Vendor chunks change rarely, so browsers keep them cached across content updates.
        codeSplitting: {
          groups: [
            { name: 'echarts', test: /node_modules[\\/](echarts|zrender)/ },
            { name: 'react', test: /node_modules[\\/](react|react-dom|react-router|scheduler)[\\/]/ },
            { name: 'ui', test: /node_modules[\\/](@base-ui|cmdk|lucide-react|@floating-ui)/ },
          ],
        },
      },
    },
  },
  plugins: [
    {
      enforce: 'pre',
      ...mdx({
        providerImportSource: '@mdx-js/react',
        remarkPlugins: [remarkFrontmatter, [remarkMdxFrontmatter, { name: 'frontmatter' }], remarkGfm, remarkMath],
        // Unknown commands fail the build rather than rendering red text. Macros: content/macros.ts.
        rehypePlugins: [rehypeSlug, [rehypeKatex, { macros, throwOnError: true, strict: 'ignore' }]],
      }),
    },
    react({ include: /\.(mdx|tsx|ts)$/ }),
    tailwindcss(),
    contentIndex({ contentDir }),
  ],
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, 'site/src'),
      '@content': contentDir,
      '@design': path.resolve(import.meta.dirname, 'design'),
      '@python': path.resolve(import.meta.dirname, 'python'),
    },
  },
  server: { fs: { allow: [import.meta.dirname] } },
})
