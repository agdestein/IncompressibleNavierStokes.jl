import{_ as e,c as s,o as t,a6 as o}from"./chunks/framework.B8Zg0AH8.js";const a="/agdestein.github.io/IncompressibleNavierStokes.jl/dev/assets/resolution.DAYTxiG0.png",b=JSON.parse('{"title":"Large eddy simulation","description":"","frontmatter":{},"headers":[],"relativePath":"features/les.md","filePath":"features/les.md","lastUpdated":null}'),i={name:"features/les.md"},r=o('<h1 id="Large-eddy-simulation" tabindex="-1">Large eddy simulation <a class="header-anchor" href="#Large-eddy-simulation" aria-label="Permalink to &quot;Large eddy simulation {#Large-eddy-simulation}&quot;">​</a></h1><p>Depending on the problem specification, a given grid resolution may not be sufficient to resolve all spatial features of the flow. Consider the following example:</p><p><img src="'+a+'" alt=""></p><p>On the left, the grid spacing is too large to capt the smallest eddies in the flow. These eddies create sub-grid stresses that also affect the large scale features. The grid must be refined if we want to compute these stresses exactly.</p><p>On the right, the smallest spatial feature of the flow is fully resolved, and there are no sub-grid stresses. The equations can be solved without worrying about errors from unresolved features. This is known as <em>Direct Numerical Simulation</em> (DNS).</p><p>If refining the grid is too costly, a closure model can be used to predict the sub-grid stresses. The models only give an estimate for these stresses, and may need to be calibrated to the given problem. When used correctly, they can predict the evolution of the large fluid motions without computing the sub-grid motions themselves. This is known as <em>Large Eddy Simulation</em> (LES).</p><p>Eddy viscosity models add a local contribution to the global baseline viscosity. The baseline viscosity models transfer of energy from resolved to atomic scales. The new turbulent viscosity on the other hand, models energy transfer from resolved to unresolved scales. This non-constant field is computed from the local velocity field.</p><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="IncompressibleNavierStokes.smagtensor!" href="#IncompressibleNavierStokes.smagtensor!">#</a> <b><u>IncompressibleNavierStokes.smagtensor!</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">smagtensor!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(σ, u, θ, setup)</span></span></code></pre></div><p>Compute Smagorinsky stress tensors <code>σ[I]</code>. The Smagorinsky constant <code>θ</code> should be a scalar between <code>0</code> and <code>1</code>.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/04ad11985c57ee41e49ccdf05cbc150df4311fe2/src/operators.jl#L1121-L1126" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="IncompressibleNavierStokes.smagorinsky_closure" href="#IncompressibleNavierStokes.smagorinsky_closure">#</a> <b><u>IncompressibleNavierStokes.smagorinsky_closure</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">m </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> smagorinsky_closure</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(setup)</span></span></code></pre></div><p>Create Smagorinsky closure model <code>m</code>. The model is called as <code>m(u, θ)</code>, where the Smagorinsky constant <code>θ</code> should be a scalar between <code>0</code> and <code>1</code> (for example <code>θ = 0.1</code>).</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/04ad11985c57ee41e49ccdf05cbc150df4311fe2/src/operators.jl#L1194-L1200" target="_blank" rel="noreferrer">source</a></p></div><br>',11),l=[r];function n(d,c,p,h,u,m){return t(),s("div",null,l)}const f=e(i,[["render",n]]);export{b as __pageData,f as default};
