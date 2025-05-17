````@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: Navier-Stokes
  text: Experience 21st century turbulence
  tagline: Differentiable fluid solver written in Julia
  image:
    src: /logo.svg
    alt: IncompressibleNavierStokes
  actions:
    - theme: brand
      text: Getting started
      link: /getting_started
    - theme: alt
      text: View on GitHub
      link: https://github.com/agdestein/IncompressibleNavierStokes.jl
    - theme: alt
      text: Gallery
      link: /examples

features:
  - icon: <img width="64" height="64" src="https://raw.githubusercontent.com/JuliaGPU/CUDA.jl/3a2546c1ac1546b05b4c0d3d6ff26c268091a202/docs/src/assets/logo.png" alt="CUDA.jl"/>
    title: GPU acceleration
    details: Fast and efficient 2D/3D kernels for CPU and GPU with CUDA.jl
    link: /manual/gpu
  - icon: <img width="64" height="64" src="https://raw.githubusercontent.com/JuliaDiff/ChainRulesCore.jl/fa530b9865ec0cb3acff81ddef0967fdcc8c8214/docs/src/assets/logo.svg" alt="ChainRules.jl"/>
    title: Differentiable physics
    details: Backpropagate through the solver using Zygote.jl to optimize closure models
    link: /manual/differentiability
  - icon: 🌊
    title: Problems
    details: Solve for decaying turbulence, channel flows, actuator disks, Rayleigh-Bénard convection, and more
    link: /examples
---
````
