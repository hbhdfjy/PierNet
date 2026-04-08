import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    fontSize: {
      xs:   ['0.8125rem', { lineHeight: '1.25rem'  }],   // 13px
      sm:   ['0.9375rem', { lineHeight: '1.5rem'   }],   // 15px
      base: ['1.0625rem', { lineHeight: '1.75rem'  }],   // 17px
      lg:   ['1.1875rem', { lineHeight: '1.875rem' }],   // 19px
      xl:   ['1.375rem',  { lineHeight: '2rem'     }],   // 22px
      '2xl':['1.625rem',  { lineHeight: '2.25rem'  }],   // 26px
      '3xl':['2rem',      { lineHeight: '2.5rem'   }],   // 32px
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
