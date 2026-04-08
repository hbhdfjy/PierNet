import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    fontSize: {
      xs:   ['0.9375rem', { lineHeight: '1.5rem'   }],   // 15px
      sm:   ['1.0625rem', { lineHeight: '1.75rem'  }],   // 17px
      base: ['1.1875rem', { lineHeight: '1.875rem' }],   // 19px
      lg:   ['1.3125rem', { lineHeight: '2rem'     }],   // 21px
      xl:   ['1.5rem',    { lineHeight: '2.125rem' }],   // 24px
      '2xl':['1.75rem',   { lineHeight: '2.375rem' }],   // 28px
      '3xl':['2.125rem',  { lineHeight: '2.625rem' }],   // 34px
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
