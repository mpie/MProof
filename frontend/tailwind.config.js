/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic':
          'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
      },
      colors: {
        navy: '#1A2A4F',
        coral: '#F7A5A5',
        peach: '#FFDBB6',
        blush: '#FFF2EF',
        brand: {
          cyan: '#CFFFFE',
          cream: '#F9F7D9',
          peach: '#FCE2CE',
          pink: '#FFC1F3',
        },
        accent: {
          cyan: '#22d3d3',
        },
      },
    },
  },
  plugins: [],
}