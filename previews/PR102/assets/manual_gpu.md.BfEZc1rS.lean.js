import{_ as a,c as s,a5 as t,o as i}from"./chunks/framework.CUacQEhW.js";const c=JSON.parse('{"title":"GPU Support","description":"","frontmatter":{},"headers":[],"relativePath":"manual/gpu.md","filePath":"manual/gpu.md","lastUpdated":null}'),l={name:"manual/gpu.md"};function r(p,e,n,o,h,d){return i(),s("div",null,e[0]||(e[0]=[t(`<h1 id="GPU-Support" tabindex="-1">GPU Support <a class="header-anchor" href="#GPU-Support" aria-label="Permalink to &quot;GPU Support {#GPU-Support}&quot;">​</a></h1><p>IncompressibleNavierStokes supports various array types. The desired backend only has to be passed to the <a href="/IncompressibleNavierStokes.jl/previews/PR102/manual/setup#IncompressibleNavierStokes.Setup-Tuple{}"><code>Setup</code></a> function:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> CUDA</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">setup </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, backend </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> CUDABackend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">())</span></span></code></pre></div><p>All operators have been made are backend agnostic by using <a href="https://github.com/JuliaGPU/KernelAbstractions.jl/" target="_blank" rel="noreferrer">KernelAbstractions.jl</a>. Even if a GPU is not available, the operators are multithreaded if Julia is started with multiple threads (e.g. <code>julia -t 4</code>)</p><ul><li><p>This has been tested with CUDA compatible GPUs.</p></li><li><p>This has not been tested with other GPU interfaces, such as</p><ul><li><p><a href="https://github.com/JuliaGPU/AMDGPU.jl" target="_blank" rel="noreferrer">AMDGPU.jl</a></p></li><li><p><a href="https://github.com/JuliaGPU/Metal.jl" target="_blank" rel="noreferrer">Metal.jl</a></p></li><li><p><a href="https://github.com/JuliaGPU/oneAPI.jl" target="_blank" rel="noreferrer">oneAPI.jl</a></p></li></ul><p>If they start supporting sparse matrices and fast Fourier transforms they could also be used.</p></li></ul><div class="tip custom-block"><p class="custom-block-title"><code>psolver_direct</code> on CUDA</p><p>To use a specialized linear solver for CUDA, make sure to install and <code>using</code> CUDA.jl and CUDSS.jl. Then <code>psolver_direct</code> will automatically use the CUDSS solver.</p></div>`,6)]))}const k=a(l,[["render",r]]);export{c as __pageData,k as default};
