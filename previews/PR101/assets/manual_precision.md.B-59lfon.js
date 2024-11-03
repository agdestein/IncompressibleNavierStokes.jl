import{_ as o,c as i,a5 as s,o as r}from"./chunks/framework.D6R-qg6l.js";const f=JSON.parse('{"title":"Floating point precision","description":"","frontmatter":{},"headers":[],"relativePath":"manual/precision.md","filePath":"manual/precision.md","lastUpdated":null}'),a={name:"manual/precision.md"};function n(t,e,c,l,p,d){return r(),i("div",null,e[0]||(e[0]=[s('<h1 id="Floating-point-precision" tabindex="-1">Floating point precision <a class="header-anchor" href="#Floating-point-precision" aria-label="Permalink to &quot;Floating point precision {#Floating-point-precision}&quot;">​</a></h1><p>IncompressibleNavierStokes generates efficient code for different floating point precisions, such as</p><ul><li><p>Double precision (<code>Float64</code>)</p></li><li><p>Single precision (<code>Float32</code>)</p></li><li><p>Half precision (<code>Float16</code>)</p></li></ul><p>To use single or half precision, all user input floats should be converted to the desired type. Mixing different precisions causes unnecessary conversions and may break the code.</p><div class="tip custom-block"><p class="custom-block-title">GPU precision</p><p>For GPUs, single precision is preferred. <code>CUDA.jl</code>s <code>cu</code> converts to single precision.</p></div><div class="tip custom-block"><p class="custom-block-title">Pressure solvers</p><p><a href="https://github.com/JuliaSparse/SparseArrays.jl" target="_blank" rel="noreferrer"><code>SparseArrays.jl</code></a>s sparse matrix factorizations only support double precision. <a href="/IncompressibleNavierStokes.jl/previews/PR101/manual/pressure#IncompressibleNavierStokes.psolver_direct-Tuple{Any}"><code>psolver_direct</code></a> only works for <code>Float64</code>. Consider using an iterative solver such as <a href="/IncompressibleNavierStokes.jl/previews/PR101/manual/pressure#IncompressibleNavierStokes.psolver_cg-Tuple{Any}"><code>psolver_cg</code></a> when using single or half precision.</p></div>',6)]))}const m=o(a,[["render",n]]);export{f as __pageData,m as default};
