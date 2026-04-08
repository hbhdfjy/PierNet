import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    fontSize: {
      xs:   ['1.0625rem', { lineHeight: '1.625rem' }],   // 17px
      sm:   ['1.1875rem', { lineHeight: '1.875rem' }],   // 19px
      base: ['1.3125rem', { lineHeight: '2rem'     }],   // 21px
      lg:   ['1.5rem',    { lineHeight: '2.125rem' }],   // 24px
      xl:   ['1.6875rem', { lineHeight: '2.375rem' }],   // 27px
      '2xl':['1.9375rem', { lineHeight: '2.625rem' }],   // 31px
      '3xl':['2.25rem',   { lineHeight: '2.875rem' }],   // 36px
      '4xl':['2.75rem',   { lineHeight: '3.25rem'  }],   // 44px
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
