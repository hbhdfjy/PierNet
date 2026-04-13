import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    fontSize: {
      xs:   ['0.75rem',  { lineHeight: '1.125rem' }],   // 12px
      sm:   ['0.875rem', { lineHeight: '1.375rem' }],   // 14px
      base: ['1rem',     { lineHeight: '1.5rem'   }],   // 16px
      lg:   ['1.125rem', { lineHeight: '1.75rem'  }],   // 18px
      xl:   ['1.25rem',  { lineHeight: '1.875rem' }],   // 20px
      '2xl':['1.5rem',   { lineHeight: '2rem'     }],   // 24px
      '3xl':['1.875rem', { lineHeight: '2.375rem' }],   // 30px
      '4xl':['2.25rem',  { lineHeight: '2.75rem'  }],   // 36px
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
      boxShadow: {
        'glow-amber':   '0 0 12px rgba(245,158,11,0.25)',
        'glow-sky':     '0 0 12px rgba(14,165,233,0.25)',
        'glow-violet':  '0 0 12px rgba(139,92,246,0.25)',
        'glow-emerald': '0 0 12px rgba(16,185,129,0.25)',
        'glow-rose':    '0 0 12px rgba(244,63,94,0.25)',
      },
    },
  },
  plugins: [],
} satisfies Config
