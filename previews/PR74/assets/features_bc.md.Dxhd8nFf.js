import{_ as e,c as s,o as i,a7 as a}from"./chunks/framework.DV1fsDIA.js";const k=JSON.parse('{"title":"Boundary conditions","description":"","frontmatter":{},"headers":[],"relativePath":"features/bc.md","filePath":"features/bc.md","lastUpdated":null}'),o={name:"features/bc.md"},r=a('<h1 id="Boundary-conditions" tabindex="-1">Boundary conditions <a class="header-anchor" href="#Boundary-conditions" aria-label="Permalink to &quot;Boundary conditions {#Boundary-conditions}&quot;">​</a></h1><p>Each boundary has exactly one type of boundary conditions. For periodic boundary conditions, the opposite boundary must also be periodic. The available boundary conditions are given below.</p><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="IncompressibleNavierStokes.PeriodicBC" href="#IncompressibleNavierStokes.PeriodicBC">#</a> <b><u>IncompressibleNavierStokes.PeriodicBC</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">PeriodicBC</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><p>Periodic boundary conditions. Must be periodic on both sides.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/873dfce69af65c832b51a3399abe4e3a5e23b37a/src/boundary_conditions.jl#L3-L7" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="IncompressibleNavierStokes.DirichletBC" href="#IncompressibleNavierStokes.DirichletBC">#</a> <b><u>IncompressibleNavierStokes.DirichletBC</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">DirichletBC</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><p>No slip boundary conditions, where all velocity components are zero.</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>DirichletBC(u, dudt)</span></span></code></pre></div><p>Dirichlet boundary conditions for the velocity, where <code>u[1] = (x..., t) -&gt; u1_BC</code> up to <code>u[d] = (x..., t) -&gt; ud_BC</code>, where <code>d</code> is the dimension.</p><p>To make the pressure the same order as velocity, also provide <code>dudt</code>.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/873dfce69af65c832b51a3399abe4e3a5e23b37a/src/boundary_conditions.jl#L10-L21" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="IncompressibleNavierStokes.SymmetricBC" href="#IncompressibleNavierStokes.SymmetricBC">#</a> <b><u>IncompressibleNavierStokes.SymmetricBC</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SymmetricBC</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><p>Symmetric boundary conditions. The parallel velocity and pressure is the same at each side of the boundary. The normal velocity is zero.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/873dfce69af65c832b51a3399abe4e3a5e23b37a/src/boundary_conditions.jl#L30-L36" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="IncompressibleNavierStokes.PressureBC" href="#IncompressibleNavierStokes.PressureBC">#</a> <b><u>IncompressibleNavierStokes.PressureBC</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">PressureBC</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><p>Pressure boundary conditions. The pressure is prescribed on the boundary (usually an &quot;outlet&quot;). The velocity has zero Neumann conditions.</p><p>Note: Currently, the pressure is prescribed with the constant value of zero on the entire boundary.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/873dfce69af65c832b51a3399abe4e3a5e23b37a/src/boundary_conditions.jl#L39-L48" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="IncompressibleNavierStokes.offset_p" href="#IncompressibleNavierStokes.offset_p">#</a> <b><u>IncompressibleNavierStokes.offset_p</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">offset_p</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(bc)</span></span></code></pre></div><p>Number of non-DOF pressure components at boundary.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/873dfce69af65c832b51a3399abe4e3a5e23b37a/src/boundary_conditions.jl#L87-L91" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="IncompressibleNavierStokes.offset_u" href="#IncompressibleNavierStokes.offset_u">#</a> <b><u>IncompressibleNavierStokes.offset_u</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">offset_u</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(bc, isnormal, isright)</span></span></code></pre></div><p>Number of non-DOF velocity components at boundary. If <code>isnormal</code>, then the velocity is normal to the boundary, else parallel. If <code>isright</code>, it is at the end/right/rear/top boundary, otherwise beginning.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/873dfce69af65c832b51a3399abe4e3a5e23b37a/src/boundary_conditions.jl#L78-L84" target="_blank" rel="noreferrer">source</a></p></div><br>',14),t=[r];function n(d,l,c,p,b,h){return i(),s("div",null,t)}const y=e(o,[["render",n]]);export{k as __pageData,y as default};
