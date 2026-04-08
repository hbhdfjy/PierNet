import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    fontSize: {
      xs:   ['0.875rem',  { lineHeight: '1.375rem' }],   // 14px
      sm:   ['1rem',      { lineHeight: '1.625rem' }],   // 16px
      base: ['1.125rem',  { lineHeight: '1.875rem' }],   // 18px
      lg:   ['1.25rem',   { lineHeight: '2rem'     }],   // 20px
      xl:   ['1.4375rem', { lineHeight: '2.125rem' }],   // 23px
      '2xl':['1.6875rem', { lineHeight: '2.375rem' }],   // 27px
      '3xl':['2.0625rem', { lineHeight: '2.625rem' }],   // 33px
      '4xl':['2.5rem',    { lineHeight: '3rem'     }],   // 40px
    },
    extend: {
      colors: {
        brand: {
          50:  '#f0f9ff',
          100: '#e0f2fe',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          900: '#0c4a6e',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
} satisfies Config
