import{_ as e,c as o,o as s,a7 as i}from"./chunks/framework.DL9vbVCB.js";const m=JSON.parse('{"title":"Floating point precision","description":"","frontmatter":{},"headers":[],"relativePath":"manual/precision.md","filePath":"manual/precision.md","lastUpdated":null}'),r={name:"manual/precision.md"},a=i('<h1 id="Floating-point-precision" tabindex="-1">Floating point precision <a class="header-anchor" href="#Floating-point-precision" aria-label="Permalink to &quot;Floating point precision {#Floating-point-precision}&quot;">​</a></h1><p>IncompressibleNavierStokes generates efficient code for different floating point precisions, such as</p><ul><li><p>Double precision (<code>Float64</code>)</p></li><li><p>Single precision (<code>Float32</code>)</p></li><li><p>Half precision (<code>Float16</code>)</p></li></ul><p>To use single or half precision, all user input floats should be converted to the desired type. Mixing different precisions causes unnecessary conversions and may break the code.</p><div class="tip custom-block"><p class="custom-block-title">GPU precision</p><p>For GPUs, single precision is preferred. <code>CUDA.jl</code>s <code>cu</code> converts to single precision.</p></div><div class="tip custom-block"><p class="custom-block-title">Pressure solvers</p><p><a href="https://github.com/JuliaSparse/SparseArrays.jl" target="_blank" rel="noreferrer"><code>SparseArrays.jl</code></a>s sparse matrix factorizations only support double precision. <a href="/IncompressibleNavierStokes.jl/previews/PR76/manual/pressure#IncompressibleNavierStokes.psolver_direct"><code>psolver_direct</code></a> only works for <code>Float64</code>. Consider using an iterative solver such as <a href="/IncompressibleNavierStokes.jl/previews/PR76/manual/pressure#IncompressibleNavierStokes.psolver_cg"><code>psolver_cg</code></a> when using single or half precision.</p></div>',6),t=[a];function n(c,l,p,d,u,_){return s(),o("div",null,t)}const h=e(r,[["render",n]]);export{m as __pageData,h as default};
