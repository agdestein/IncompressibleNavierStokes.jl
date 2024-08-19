import{_ as e,c as s,o as i,a7 as a}from"./chunks/framework.7bw2KgyR.js";const k=JSON.parse('{"title":"Boundary conditions","description":"","frontmatter":{},"headers":[],"relativePath":"manual/bc.md","filePath":"manual/bc.md","lastUpdated":null}'),t={name:"manual/bc.md"},o=a('<h1 id="Boundary-conditions" tabindex="-1">Boundary conditions <a class="header-anchor" href="#Boundary-conditions" aria-label="Permalink to &quot;Boundary conditions {#Boundary-conditions}&quot;">​</a></h1><p>Each boundary has exactly one type of boundary conditions. For periodic boundary conditions, the opposite boundary must also be periodic. The available boundary conditions are given below.</p><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="IncompressibleNavierStokes.AbstractBC" href="#IncompressibleNavierStokes.AbstractBC">#</a> <b><u>IncompressibleNavierStokes.AbstractBC</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">abstract type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> AbstractBC</span></span></code></pre></div><p>Boundary condition for one side of the domain.</p><p><strong>Fields</strong></p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/953afb3c0b5549f51a6f88f6d9d5a8f785033243/src/boundary_conditions.jl#L1" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="IncompressibleNavierStokes.PeriodicBC" href="#IncompressibleNavierStokes.PeriodicBC">#</a> <b><u>IncompressibleNavierStokes.PeriodicBC</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> PeriodicBC </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> IncompressibleNavierStokes.AbstractBC</span></span></code></pre></div><p>Periodic boundary conditions. Must be periodic on both sides.</p><p><strong>Fields</strong></p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/953afb3c0b5549f51a6f88f6d9d5a8f785033243/src/boundary_conditions.jl#L4" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="IncompressibleNavierStokes.DirichletBC" href="#IncompressibleNavierStokes.DirichletBC">#</a> <b><u>IncompressibleNavierStokes.DirichletBC</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> DirichletBC{U, DUDT} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> IncompressibleNavierStokes.AbstractBC</span></span></code></pre></div><p>Dirichlet boundary conditions for the velocity, where <code>u[1] = (x..., t) -&gt; u1_BC</code> up to <code>u[d] = (x..., t) -&gt; ud_BC</code>, where <code>d</code> is the dimension.</p><p>When <code>u</code> is <code>nothing</code>, then the boundary conditions are no slip boundary conditions, where all velocity components are zero.</p><p>To make the pressure the same order as velocity, also provide <code>dudt</code>.</p><p><strong>Fields</strong></p><ul><li><p><code>u</code>: Boundary condition</p></li><li><p><code>dudt</code>: Time derivative of boundary condition</p></li></ul><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/953afb3c0b5549f51a6f88f6d9d5a8f785033243/src/boundary_conditions.jl#L7" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="IncompressibleNavierStokes.SymmetricBC" href="#IncompressibleNavierStokes.SymmetricBC">#</a> <b><u>IncompressibleNavierStokes.SymmetricBC</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SymmetricBC </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> IncompressibleNavierStokes.AbstractBC</span></span></code></pre></div><p>Symmetric boundary conditions. The parallel velocity and pressure is the same at each side of the boundary. The normal velocity is zero.</p><p><strong>Fields</strong></p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/953afb3c0b5549f51a6f88f6d9d5a8f785033243/src/boundary_conditions.jl#L26" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="IncompressibleNavierStokes.PressureBC" href="#IncompressibleNavierStokes.PressureBC">#</a> <b><u>IncompressibleNavierStokes.PressureBC</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> PressureBC </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> IncompressibleNavierStokes.AbstractBC</span></span></code></pre></div><p>Pressure boundary conditions. The pressure is prescribed on the boundary (usually an &quot;outlet&quot;). The velocity has zero Neumann conditions.</p><p>Note: Currently, the pressure is prescribed with the constant value of zero on the entire boundary.</p><p><strong>Fields</strong></p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/953afb3c0b5549f51a6f88f6d9d5a8f785033243/src/boundary_conditions.jl#L33" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="IncompressibleNavierStokes.offset_p" href="#IncompressibleNavierStokes.offset_p">#</a> <b><u>IncompressibleNavierStokes.offset_p</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">offset_p</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(bc, isnormal, isright)</span></span></code></pre></div><p>Number of non-DOF pressure components at boundary.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/953afb3c0b5549f51a6f88f6d9d5a8f785033243/src/boundary_conditions.jl#L79" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="IncompressibleNavierStokes.offset_u" href="#IncompressibleNavierStokes.offset_u">#</a> <b><u>IncompressibleNavierStokes.offset_u</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">offset_u</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(bc, isnormal, isright)</span></span></code></pre></div><p>Number of non-DOF velocity components at boundary. If <code>isnormal</code>, then the velocity is normal to the boundary, else parallel. If <code>isright</code>, it is at the end/right/rear/top boundary, otherwise beginning.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/953afb3c0b5549f51a6f88f6d9d5a8f785033243/src/boundary_conditions.jl#L70" target="_blank" rel="noreferrer">source</a></p></div><br>',16),r=[o];function n(d,l,p,c,b,h){return i(),s("div",null,r)}const g=e(t,[["render",n]]);export{k as __pageData,g as default};
