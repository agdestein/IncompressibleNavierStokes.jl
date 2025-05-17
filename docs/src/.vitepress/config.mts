import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import mathjax3 from "markdown-it-mathjax3";
import footnote from "markdown-it-footnote";
import { transformerMetaWordHighlight } from '@shikijs/transformers';
import path from 'path'

function getBaseRepository(base: string): string {
  if (!base || base === '/') return '/';
  const parts = base.split('/').filter(Boolean);
  return parts.length > 0 ? `/${parts[0]}/` : '/';
}

const baseTemp = {
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',// TODO: replace this in makedocs!
}

const nav = [
  { text: 'Home', link: '/' },
  { text: 'Getting Started', link: '/getting_started' },
  { text: 'Examples', link: '/examples/', activeMatch: '/examples/.*' },
  {
    text: 'Manual',
    items: [
      {
        text: 'Equations',
        items: [
          { text: 'Incompressible Navier-Stokes equations', link: '/manual/ns' },
          { text: 'Spatial discretization', link: '/manual/spatial' },
          { text: 'Time discretization', link: '/manual/time' },
        ],
      },
      {
        text: 'API',
        items: [
          { text: 'Problem setup', link: '/manual/setup' },
          { text: 'Operators', link: '/manual/operators' },
          { text: 'Pressure solvers', link: '/manual/pressure' },
          { text: 'Solver', link: '/manual/solver' },
          { text: 'Utils', link: '/manual/utils' },
        ],
      },
      {
        text: 'Guide',
        items: [
          { text: 'Floating point precision', link: '/manual/precision' },
          { text: 'GPU support', link: '/manual/gpu' },
          { text: 'Differentiating code', link: '/manual/differentiability' },
          { text: 'Sparse matrices', link: '/manual/matrices' },
          { text: 'Temperature equation', link: '/manual/temperature' },
          { text: 'Large eddy simulation', link: '/manual/les' },
          { text: 'SciML', link: '/manual/sciml' },
        ],
      },
    ],
  },
  {
    text: 'About',
    items: [
      { text: 'About', link: '/about/' },
      { text: 'License', link: '/about/license' },
      { text: 'Citing', link: '/about/citing' },
      { text: 'Local development', link: '/about/development' },
      { text: 'Contributing', link: '/about/contributing' },
      { text: 'Package versions', link: '/about/versions' },
    ],
  },
  { text: 'References', link: '/references' },
  {
    component: 'VersionPicker',
  },
]

const sidebar = {
  "/examples/": {
    items: [
      { text: 'Examples gallery', link: '/examples/' },
      {
        text: 'Periodic box',
        items: [
          { text: 'Decaying Turbulunce (2D)', link: '/examples/generated/DecayingTurbulence2D' },
          { text: 'Decaying Turbulunce (3D)', link: '/examples/generated/DecayingTurbulence3D' },
          { text: 'Taylor-Green Vortex (2D)', link: '/examples/generated/TaylorGreenVortex2D' },
          { text: 'Taylor-Green Vortex (3D)', link: '/examples/generated/TaylorGreenVortex3D' },
          { text: 'Kolmogorov flow (2D)', link: '/examples/generated/Kolmogorov2D' },
        ],
      },
      {
        text: 'Mixed boundary conditions',
        items: [
          { text: 'Actuator (2D)', link: '/examples/generated/Actuator2D' },
          { text: 'Actuator (3D)', link: '/examples/generated/Actuator3D' },
          { text: 'Backward Facing Step (2D)', link: '/examples/generated/BackwardFacingStep2D' },
          { text: 'Backward Facing Step (3D)', link: '/examples/generated/BackwardFacingStep3D' },
          { text: 'Lid-Driven Cavity (2D)', link: '/examples/generated/LidDrivenCavity2D' },
          { text: 'Multiple actuators (2D)', link: '/examples/generated/MultiActuator' },
        ],
      },
      {
        text: 'With temperature field',
        items: [
          { text: 'Rayleigh-Bénard (2D)', link: '/examples/generated/RayleighBenard2D' },
          { text: 'Rayleigh-Bénard (3D)', link: '/examples/generated/RayleighBenard3D' },
          { text: 'Rayleigh-Taylor (2D)', link: '/examples/generated/RayleighTaylor2D' },
        ],
      },
    ],
  },
  "/manual/": {
    text: 'Manual',
    items: [
      {
        text: 'Equations',
        items: [
          { text: 'Incompressible Navier-Stokes equations', link: '/manual/ns' },
          { text: 'Spatial discretization', link: '/manual/spatial' },
          { text: 'Time discretization', link: '/manual/time' },
        ],
      },
      {
        text: 'API',
        items: [
          { text: 'Problem setup', link: '/manual/setup' },
          { text: 'Operators', link: '/manual/operators' },
          { text: 'Pressure solvers', link: '/manual/pressure' },
          { text: 'Solver', link: '/manual/solver' },
          { text: 'Utils', link: '/manual/utils' },
        ],
      },
      {
        text: 'Guide',
        items: [
          { text: 'Floating point precision', link: '/manual/precision' },
          { text: 'GPU support', link: '/manual/gpu' },
          { text: 'Differentiating code', link: '/manual/differentiability' },
          { text: 'Sparse matrices', link: '/manual/matrices' },
          { text: 'Temperature equation', link: '/manual/temperature' },
          { text: 'Large eddy simulation', link: '/manual/les' },
          { text: 'SciML', link: '/manual/sciml' },
        ],
      },
    ],
  },
  "/about/": {
    text: 'About',
    items: [
      { text: 'About', link: '/about/' },
      { text: 'License', link: '/about/license' },
      { text: 'Citing', link: '/about/citing' },
      { text: 'Local development', link: '/about/development' },
      { text: 'Contributing', link: '/about/contributing' },
      { text: 'Package versions', link: '/about/versions' },
    ],
  },
}

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS', // TODO: replace this in makedocs!
  title: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  description: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  lastUpdated: true,
  cleanUrls: true,
  outDir: 'REPLACE_ME_DOCUMENTER_VITEPRESS', // This is required for MarkdownVitepress to work correctly...
  head: [
    ['link', { rel: 'icon', href: 'REPLACE_ME_DOCUMENTER_VITEPRESS_FAVICON' }],
    ['script', {src: `${getBaseRepository(baseTemp.base)}versions.js`}],
    // ['script', {src: '/versions.js'], for custom domains, I guess if deploy_url is available.
    ['script', {src: `${baseTemp.base}siteinfo.js`}]
  ],
  ignoreDeadLinks: false,

  vite: {
    define: {
      __DEPLOY_ABSPATH__: JSON.stringify('REPLACE_ME_DOCUMENTER_VITEPRESS_DEPLOY_ABSPATH'),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '../components')
      }
    },
    optimizeDeps: {
      exclude: [ 
        '@nolebase/vitepress-plugin-enhanced-readabilities/client',
        'vitepress',
        '@nolebase/ui',
      ], 
    }, 
    ssr: { 
      noExternal: [ 
        // If there are other packages that need to be processed by Vite, you can add them here.
        '@nolebase/vitepress-plugin-enhanced-readabilities',
        '@nolebase/ui',
      ], 
    },
  },
  markdown: {
    math: true,
    config(md) {
      md.use(tabsMarkdownPlugin),
      md.use(mathjax3),
      md.use(footnote)
    },
    theme: {
      light: "github-light",
      dark: "github-dark"
    },
    codeTransformers: [transformerMetaWordHighlight(),],
  },

  themeConfig: {
    outline: 'deep',
    logo: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    // logo: {
    //   'light': '/logo.svg',
    //   'dark': '/logo.svg'
    // },
    search: {
      provider: 'local',
      options: {
        detailedView: true
      }
    },
    editLink: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    // socialLinks: [
    //   { icon: 'github', link: 'REPLACE_ME_DOCUMENTER_VITEPRESS' }
    // ],
    footer: {
      message: 'Made with <a href="https://luxdl.github.io/DocumenterVitepress.jl/dev/" target="_blank"><strong>DocumenterVitepress.jl</strong></a><br>',
      copyright: `© Copyright ${new Date().getUTCFullYear()}.`
    },
    nav,
    sidebar,
  },
})
