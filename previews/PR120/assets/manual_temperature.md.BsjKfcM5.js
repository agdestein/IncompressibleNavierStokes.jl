import{_ as t,c as n,a5 as a,j as s,a as p,G as r,B as l,o}from"./chunks/framework.CBAIMBsr.js";const b=JSON.parse('{"title":"Temperature equation","description":"","frontmatter":{},"headers":[],"relativePath":"manual/temperature.md","filePath":"manual/temperature.md","lastUpdated":null}'),h={name:"manual/temperature.md"},d={class:"jldocstring custom-block",open:""};function k(u,e,c,E,m,g){const i=l("Badge");return o(),n("div",null,[e[3]||(e[3]=a(`<h1 id="Temperature-equation" tabindex="-1">Temperature equation <a class="header-anchor" href="#Temperature-equation" aria-label="Permalink to &quot;Temperature equation {#Temperature-equation}&quot;">​</a></h1><p>IncompressibleNavierStokes.jl supports adding a temperature equation, which is coupled back to the momentum equation through a gravity term [<a href="/IncompressibleNavierStokes.jl/previews/PR120/references#Sanderse2023">8</a>].</p><p>To enable the temperature equation, you need to set the <code>temperature</code> keyword in setup:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">setup </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    temperature </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> temperature_equation</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>where <code>temperature_equation</code> can be configured as follows:</p>`,5)),s("details",d,[s("summary",null,[e[0]||(e[0]=s("a",{id:"IncompressibleNavierStokes.temperature_equation-manual-temperature",href:"#IncompressibleNavierStokes.temperature_equation-manual-temperature"},[s("span",{class:"jlbinding"},"IncompressibleNavierStokes.temperature_equation")],-1)),e[1]||(e[1]=p()),r(i,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[2]||(e[2]=a(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">temperature_equation</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Pr,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Ra,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Ge,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dodissipation,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    boundary_conditions,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gdir,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    nondim_type</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create temperature equation setup (stored in a named tuple).</p><p>The equation is parameterized by three dimensionless numbers (Prandtl number, Rayleigh number, and Gebhart number), and requires separate boundary conditions for the <code>temperature</code> field. The <code>gdir</code> keyword specifies the direction gravity, while <code>non_dim_type</code> specifies the type of non-dimensionalization.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/558326e021454ecaf2ba1b317cafa75b6fafe39f/src/setup.jl#L48" target="_blank" rel="noreferrer">source</a></p>`,4))])])}const f=t(h,[["render",k]]);export{b as __pageData,f as default};
