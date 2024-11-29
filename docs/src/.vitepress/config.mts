import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import mathjax3 from "markdown-it-mathjax3";
import footnote from "markdown-it-footnote";
import { transformerMetaWordHighlight } from '@shikijs/transformers';

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS', // TODO: replace this in makedocs!
  title: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  description: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  lastUpdated: true,
  cleanUrls: true,
  outDir: 'REPLACE_ME_DOCUMENTER_VITEPRESS', // This is required for MarkdownVitepress to work correctly...
  // head: [['link', { rel: 'icon', href: 'REPLACE_ME_DOCUMENTER_VITEPRESS_FAVICON' }]],

  ignoreDeadLinks: true, // Reexporting KernelAbstractions.CPU fails otherwise

  head: [
    ['link', { rel: 'apple-touch-icon', sizes: '180x180', href: '/apple-touch-icon.png' }],
    ['link', { rel: 'icon', type: 'image/png', sizes: '32x32', href: '/favicon-32x32.png' }],
    ['link', { rel: 'icon', type: 'image/png', sizes: '16x16', href: '/favicon-16x16.png' }],
    ['link', { rel: 'icon', href: '/favicon.ico' }],
    ['link', { rel: 'manifest', href: '/site.webmanifest' }],
  ],

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
    // https://vitepress.dev/reference/default-theme-config
    // logo: { src: '/logo.svg' },
    logo: {
      'light': '/logo.svg',
      'dark': '/logo.svg'
    },
    search: {
      provider: 'local',
      options: {
        detailedView: true
      }
    },
    editLink: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    socialLinks: [
      { icon: 'github', link: 'REPLACE_ME_DOCUMENTER_VITEPRESS' }
    ],
    footer: {
      message: 'Made with <a href="https://documenter.juliadocs.org/stable/" target="_blank"><strong>Documenter.jl</strong></a>, <a href="https://vitepress.dev" target="_blank"><strong>VitePress</strong></a> and <a href="https://luxdl.github.io/DocumenterVitepress.jl/stable/" target="_blank"><strong>DocumenterVitepress.jl</strong></a> <br>',
      copyright: `© Copyright ${new Date().getUTCFullYear()}.`
    },
    nav: [
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
              { text: 'Neural closure models', link: '/manual/closure' },
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
    ],
    sidebar: {
      "/examples/": {
        items: [
          { text: 'Examples gallery', link: '/examples/' },
          {
            text: 'Simple flows',
            items: [
              { text: 'Decaying Turbulunce (2D)', link: '/examples/generated/DecayingTurbulence2D' },
              { text: 'Decaying Turbulunce (3D)', link: '/examples/generated/DecayingTurbulence3D' },
              { text: 'Taylor-Green Vortex (2D)', link: '/examples/generated/TaylorGreenVortex2D' },
              { text: 'Taylor-Green Vortex (3D)', link: '/examples/generated/TaylorGreenVortex3D' },
              { text: 'Kolmogorov flow (2D)', link: '/examples/generated/Kolmogorov2D' },
              { text: 'Shear Layer (2D)', link: '/examples/generated/ShearLayer2D' },
              { text: 'Plane jets (2D)', link: '/examples/generated/PlaneJets2D' },
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
              { text: 'Lid-Driven Cavity (3D)', link: '/examples/generated/LidDrivenCavity3D' },
              { text: 'Multiple actuators (2D)', link: '/examples/generated/MultiActuator' },
              { text: 'Planar Mixing (2D)', link: '/examples/generated/PlanarMixing2D' },
            ],
          },
          {
            text: 'With temperature field',
            items: [
              { text: 'Rayleigh-Bénard (2D)', link: '/examples/generated/RayleighBenard2D' },
              { text: 'Rayleigh-Bénard (3D)', link: '/examples/generated/RayleighBenard3D' },
              { text: 'Rayleigh-Taylor (2D)', link: '/examples/generated/RayleighTaylor2D' },
              { text: 'Rayleigh-Taylor (3D)', link: '/examples/generated/RayleighTaylor3D' },
            ],
          },
          // {
          //   text: 'Neural closure models',
          //   items: [
          //     { text: 'Filter analysis', link: '/examples/generated/prioranalysis' },
          //     { text: 'CNN closures', link: '/examples/generated/postanalysis' },
          //     { text: 'Equivariant closures', link: '/examples/generated/symmetryanalysis' },
          //   ],
          // },
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
              { text: 'Neural closure models', link: '/manual/closure' },
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
    },
  },
})
