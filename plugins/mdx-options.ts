/** MDX compile options, shared by the site build (vite.config.ts) and `make doctor`, which compiles every note. */
import type { CompileOptions } from '@mdx-js/mdx'
import rehypeKatex from 'rehype-katex'
import rehypeSlug from 'rehype-slug'
import remarkFrontmatter from 'remark-frontmatter'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkMdxFrontmatter from 'remark-mdx-frontmatter'
import remarkSmartypants from 'remark-smartypants'
import { macros } from '../content/macros.ts'

export const mdxOptions = {
  providerImportSource: '@mdx-js/react',
  // Smartypants turns straight quotes into curly ones: the prose font draws a straight " as a closing quote.
  remarkPlugins: [
    remarkFrontmatter,
    [remarkMdxFrontmatter, { name: 'frontmatter' }],
    remarkGfm,
    remarkMath,
    [remarkSmartypants, { dashes: false }],
  ],
  // Unknown commands fail the build rather than rendering red text. Macros: content/macros.ts.
  rehypePlugins: [rehypeSlug, [rehypeKatex, { macros, throwOnError: true, strict: 'ignore' }]],
} satisfies CompileOptions
