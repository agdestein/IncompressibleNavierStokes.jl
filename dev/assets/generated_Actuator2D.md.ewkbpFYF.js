import{_ as s,c as a,o as t,a6 as n}from"./chunks/framework.DpS6FyRF.js";const i="/IncompressibleNavierStokes.jl/dev/assets/qbmyvwj.MCdy6iP9.png",p="/IncompressibleNavierStokes.jl/dev/assets/knfcqhn.DQ4Nvpb1.png",l="/IncompressibleNavierStokes.jl/dev/assets/wseqztv.BOckA-lV.png",e="/IncompressibleNavierStokes.jl/dev/assets/bsliugi.CFFxulAy.png",F=JSON.parse('{"title":"Unsteady actuator case - 2D","description":"","frontmatter":{},"headers":[],"relativePath":"generated/Actuator2D.md","filePath":"generated/Actuator2D.md","lastUpdated":null}'),h={name:"generated/Actuator2D.md"},k=n(`<h1 id="Unsteady-actuator-case-2D" tabindex="-1">Unsteady actuator case - 2D <a class="header-anchor" href="#Unsteady-actuator-case-2D" aria-label="Permalink to &quot;Unsteady actuator case - 2D {#Unsteady-actuator-case-2D}&quot;">​</a></h1><p>In this example, an unsteady inlet velocity profile at encounters a wind turbine blade in a wall-less domain. The blade is modeled as a uniform body force on a thin rectangle.</p><p>We start by loading packages. A <a href="https://github.com/JuliaPlots/Makie.jl" target="_blank" rel="noreferrer">Makie</a> plotting backend is needed for plotting. <code>GLMakie</code> creates an interactive window (useful for real-time plotting), but does not work when building this example on GitHub. <code>CairoMakie</code> makes high-quality static vector-graphics plots.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> CairoMakie</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> IncompressibleNavierStokes</span></span></code></pre></div><p>Output directory</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">output </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;output/Actuator2D&quot;</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>&quot;output/Actuator2D&quot;</span></span></code></pre></div><p>A 2D grid is a Cartesian product of two vectors</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">n </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 40</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LinRange</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">n </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LinRange</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">n </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">plotgrid</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, y)</span></span></code></pre></div><p><img src="`+i+`" alt=""></p><p>Boundary conditions</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">boundary_conditions </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # x left, x right</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Unsteady BC requires time derivatives</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DirichletBC</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (dim, x, y, t) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sin</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">π</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 6</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sin</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">π</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 6</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> t) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> π</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">dim</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">==</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (dim, x, y, t) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">π</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 6</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">^</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                cos</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">π</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 6</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> t) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                cos</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">π</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 6</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sin</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">π</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 6</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> t) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> π</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">dim</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">==</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        PressureBC</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ),</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # y rear, y front</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">PressureBC</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">PressureBC</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>((DirichletBC{Main.var&quot;#1#3&quot;, Main.var&quot;#2#4&quot;}(Main.var&quot;#1#3&quot;(), Main.var&quot;#2#4&quot;()), PressureBC()), (PressureBC(), PressureBC()))</span></span></code></pre></div><p>Actuator body force: A thrust coefficient <code>Cₜ</code> distributed over a thin rectangle</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">xc, yc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> # Disk center</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">D </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">           # Disk diameter</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">δ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.11</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">          # Disk thickness</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Cₜ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.2</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">          # Thrust coefficient</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">cₜ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Cₜ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (D </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> δ)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">inside</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> abs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xc) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">≤</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> δ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &amp;&amp;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> abs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> yc) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">≤</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> D </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">bodyforce</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dim, x, y, t) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dim</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">==</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> ?</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">cₜ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> inside</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>bodyforce (generic function with 1 method)</span></span></code></pre></div><p>Build setup and assemble operators</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">setup </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, y; Re </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, boundary_conditions, bodyforce);</span></span></code></pre></div><p>Initial conditions (extend inflow)</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ustart </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> create_initial_conditions</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(setup, (dim, x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dim</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">==</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> ?</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> :</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span></code></pre></div><p>Solve unsteady problem</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">state, outputs </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> solve_unsteady</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    setup,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ustart,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    tlims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">12.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    method </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> RKMethods</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">RK44P2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Δt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.05</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    processors </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        rtp </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> realtimeplotter</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; setup, plot </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> fieldplot, nupdate </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # ehist = realtimeplotter(; setup, plot = energy_history_plot, nupdate = 1),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # espec = realtimeplotter(; setup, plot = energy_spectrum_plot, nupdate = 1),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # anim = animator(; setup, path = &quot;$output/vorticity.mkv&quot;, nupdate = 20),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # vtk = vtk_writer(; setup, nupdate = 10, dir = &quot;$output&quot;, filename = &quot;solution&quot;),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # field = fieldsaver(; setup, nupdate = 10),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        log </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> timelogger</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; nupdate </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Iteration 1	t = 0.05	Δt = 0.05	umax = 1.02497</span></span>
<span class="line"><span>Iteration 2	t = 0.1	Δt = 0.05	umax = 1.0404</span></span>
<span class="line"><span>Iteration 3	t = 0.15	Δt = 0.05	umax = 1.04992</span></span>
<span class="line"><span>Iteration 4	t = 0.2	Δt = 0.05	umax = 1.05704</span></span>
<span class="line"><span>Iteration 5	t = 0.25	Δt = 0.05	umax = 1.06159</span></span>
<span class="line"><span>Iteration 6	t = 0.3	Δt = 0.05	umax = 1.06455</span></span>
<span class="line"><span>Iteration 7	t = 0.35	Δt = 0.05	umax = 1.06525</span></span>
<span class="line"><span>Iteration 8	t = 0.4	Δt = 0.05	umax = 1.06475</span></span>
<span class="line"><span>Iteration 9	t = 0.45	Δt = 0.05	umax = 1.06544</span></span>
<span class="line"><span>Iteration 10	t = 0.5	Δt = 0.05	umax = 1.0659</span></span>
<span class="line"><span>Iteration 11	t = 0.55	Δt = 0.05	umax = 1.06597</span></span>
<span class="line"><span>Iteration 12	t = 0.6	Δt = 0.05	umax = 1.06549</span></span>
<span class="line"><span>Iteration 13	t = 0.65	Δt = 0.05	umax = 1.06466</span></span>
<span class="line"><span>Iteration 14	t = 0.7	Δt = 0.05	umax = 1.06358</span></span>
<span class="line"><span>Iteration 15	t = 0.75	Δt = 0.05	umax = 1.0623</span></span>
<span class="line"><span>Iteration 16	t = 0.8	Δt = 0.05	umax = 1.06165</span></span>
<span class="line"><span>Iteration 17	t = 0.85	Δt = 0.05	umax = 1.06087</span></span>
<span class="line"><span>Iteration 18	t = 0.9	Δt = 0.05	umax = 1.05988</span></span>
<span class="line"><span>Iteration 19	t = 0.95	Δt = 0.05	umax = 1.05873</span></span>
<span class="line"><span>Iteration 20	t = 1	Δt = 0.05	umax = 1.05744</span></span>
<span class="line"><span>Iteration 21	t = 1.05	Δt = 0.05	umax = 1.05605</span></span>
<span class="line"><span>Iteration 22	t = 1.1	Δt = 0.05	umax = 1.05458</span></span>
<span class="line"><span>Iteration 23	t = 1.15	Δt = 0.05	umax = 1.05311</span></span>
<span class="line"><span>Iteration 24	t = 1.2	Δt = 0.05	umax = 1.05199</span></span>
<span class="line"><span>Iteration 25	t = 1.25	Δt = 0.05	umax = 1.05078</span></span>
<span class="line"><span>Iteration 26	t = 1.3	Δt = 0.05	umax = 1.04949</span></span>
<span class="line"><span>Iteration 27	t = 1.35	Δt = 0.05	umax = 1.04814</span></span>
<span class="line"><span>Iteration 28	t = 1.4	Δt = 0.05	umax = 1.04673</span></span>
<span class="line"><span>Iteration 29	t = 1.45	Δt = 0.05	umax = 1.04528</span></span>
<span class="line"><span>Iteration 30	t = 1.5	Δt = 0.05	umax = 1.04381</span></span>
<span class="line"><span>Iteration 31	t = 1.55	Δt = 0.05	umax = 1.04231</span></span>
<span class="line"><span>Iteration 32	t = 1.6	Δt = 0.05	umax = 1.04107</span></span>
<span class="line"><span>Iteration 33	t = 1.65	Δt = 0.05	umax = 1.03981</span></span>
<span class="line"><span>Iteration 34	t = 1.7	Δt = 0.05	umax = 1.03852</span></span>
<span class="line"><span>Iteration 35	t = 1.75	Δt = 0.05	umax = 1.03722</span></span>
<span class="line"><span>Iteration 36	t = 1.8	Δt = 0.05	umax = 1.03591</span></span>
<span class="line"><span>Iteration 37	t = 1.85	Δt = 0.05	umax = 1.03463</span></span>
<span class="line"><span>Iteration 38	t = 1.9	Δt = 0.05	umax = 1.03336</span></span>
<span class="line"><span>Iteration 39	t = 1.95	Δt = 0.05	umax = 1.03213</span></span>
<span class="line"><span>Iteration 40	t = 2	Δt = 0.05	umax = 1.03106</span></span>
<span class="line"><span>Iteration 41	t = 2.05	Δt = 0.05	umax = 1.03001</span></span>
<span class="line"><span>Iteration 42	t = 2.1	Δt = 0.05	umax = 1.02898</span></span>
<span class="line"><span>Iteration 43	t = 2.15	Δt = 0.05	umax = 1.02798</span></span>
<span class="line"><span>Iteration 44	t = 2.2	Δt = 0.05	umax = 1.02701</span></span>
<span class="line"><span>Iteration 45	t = 2.25	Δt = 0.05	umax = 1.02607</span></span>
<span class="line"><span>Iteration 46	t = 2.3	Δt = 0.05	umax = 1.02518</span></span>
<span class="line"><span>Iteration 47	t = 2.35	Δt = 0.05	umax = 1.02434</span></span>
<span class="line"><span>Iteration 48	t = 2.4	Δt = 0.05	umax = 1.02356</span></span>
<span class="line"><span>Iteration 49	t = 2.45	Δt = 0.05	umax = 1.02288</span></span>
<span class="line"><span>Iteration 50	t = 2.5	Δt = 0.05	umax = 1.02223</span></span>
<span class="line"><span>Iteration 51	t = 2.55	Δt = 0.05	umax = 1.02163</span></span>
<span class="line"><span>Iteration 52	t = 2.6	Δt = 0.05	umax = 1.02106</span></span>
<span class="line"><span>Iteration 53	t = 2.65	Δt = 0.05	umax = 1.02054</span></span>
<span class="line"><span>Iteration 54	t = 2.7	Δt = 0.05	umax = 1.02003</span></span>
<span class="line"><span>Iteration 55	t = 2.75	Δt = 0.05	umax = 1.01957</span></span>
<span class="line"><span>Iteration 56	t = 2.8	Δt = 0.05	umax = 1.01913</span></span>
<span class="line"><span>Iteration 57	t = 2.85	Δt = 0.05	umax = 1.01872</span></span>
<span class="line"><span>Iteration 58	t = 2.9	Δt = 0.05	umax = 1.01833</span></span>
<span class="line"><span>Iteration 59	t = 2.95	Δt = 0.05	umax = 1.01801</span></span>
<span class="line"><span>Iteration 60	t = 3	Δt = 0.05	umax = 1.01775</span></span>
<span class="line"><span>Iteration 61	t = 3.05	Δt = 0.05	umax = 1.0175</span></span>
<span class="line"><span>Iteration 62	t = 3.1	Δt = 0.05	umax = 1.01728</span></span>
<span class="line"><span>Iteration 63	t = 3.15	Δt = 0.05	umax = 1.01708</span></span>
<span class="line"><span>Iteration 64	t = 3.2	Δt = 0.05	umax = 1.0169</span></span>
<span class="line"><span>Iteration 65	t = 3.25	Δt = 0.05	umax = 1.01674</span></span>
<span class="line"><span>Iteration 66	t = 3.3	Δt = 0.05	umax = 1.01659</span></span>
<span class="line"><span>Iteration 67	t = 3.35	Δt = 0.05	umax = 1.01647</span></span>
<span class="line"><span>Iteration 68	t = 3.4	Δt = 0.05	umax = 1.01637</span></span>
<span class="line"><span>Iteration 69	t = 3.45	Δt = 0.05	umax = 1.01629</span></span>
<span class="line"><span>Iteration 70	t = 3.5	Δt = 0.05	umax = 1.01621</span></span>
<span class="line"><span>Iteration 71	t = 3.55	Δt = 0.05	umax = 1.01616</span></span>
<span class="line"><span>Iteration 72	t = 3.6	Δt = 0.05	umax = 1.01611</span></span>
<span class="line"><span>Iteration 73	t = 3.65	Δt = 0.05	umax = 1.0161</span></span>
<span class="line"><span>Iteration 74	t = 3.7	Δt = 0.05	umax = 1.01611</span></span>
<span class="line"><span>Iteration 75	t = 3.75	Δt = 0.05	umax = 1.01613</span></span>
<span class="line"><span>Iteration 76	t = 3.8	Δt = 0.05	umax = 1.01617</span></span>
<span class="line"><span>Iteration 77	t = 3.85	Δt = 0.05	umax = 1.01623</span></span>
<span class="line"><span>Iteration 78	t = 3.9	Δt = 0.05	umax = 1.01629</span></span>
<span class="line"><span>Iteration 79	t = 3.95	Δt = 0.05	umax = 1.01636</span></span>
<span class="line"><span>Iteration 80	t = 4	Δt = 0.05	umax = 1.01644</span></span>
<span class="line"><span>Iteration 81	t = 4.05	Δt = 0.05	umax = 1.01652</span></span>
<span class="line"><span>Iteration 82	t = 4.1	Δt = 0.05	umax = 1.01661</span></span>
<span class="line"><span>Iteration 83	t = 4.15	Δt = 0.05	umax = 1.01671</span></span>
<span class="line"><span>Iteration 84	t = 4.2	Δt = 0.05	umax = 1.0168</span></span>
<span class="line"><span>Iteration 85	t = 4.25	Δt = 0.05	umax = 1.01691</span></span>
<span class="line"><span>Iteration 86	t = 4.3	Δt = 0.05	umax = 1.01702</span></span>
<span class="line"><span>Iteration 87	t = 4.35	Δt = 0.05	umax = 1.01713</span></span>
<span class="line"><span>Iteration 88	t = 4.4	Δt = 0.05	umax = 1.01725</span></span>
<span class="line"><span>Iteration 89	t = 4.45	Δt = 0.05	umax = 1.01736</span></span>
<span class="line"><span>Iteration 90	t = 4.5	Δt = 0.05	umax = 1.01749</span></span>
<span class="line"><span>Iteration 91	t = 4.55	Δt = 0.05	umax = 1.01762</span></span>
<span class="line"><span>Iteration 92	t = 4.6	Δt = 0.05	umax = 1.01776</span></span>
<span class="line"><span>Iteration 93	t = 4.65	Δt = 0.05	umax = 1.01789</span></span>
<span class="line"><span>Iteration 94	t = 4.7	Δt = 0.05	umax = 1.01803</span></span>
<span class="line"><span>Iteration 95	t = 4.75	Δt = 0.05	umax = 1.01817</span></span>
<span class="line"><span>Iteration 96	t = 4.8	Δt = 0.05	umax = 1.0183</span></span>
<span class="line"><span>Iteration 97	t = 4.85	Δt = 0.05	umax = 1.01844</span></span>
<span class="line"><span>Iteration 98	t = 4.9	Δt = 0.05	umax = 1.01858</span></span>
<span class="line"><span>Iteration 99	t = 4.95	Δt = 0.05	umax = 1.01872</span></span>
<span class="line"><span>Iteration 100	t = 5	Δt = 0.05	umax = 1.01886</span></span>
<span class="line"><span>Iteration 101	t = 5.05	Δt = 0.05	umax = 1.01901</span></span>
<span class="line"><span>Iteration 102	t = 5.1	Δt = 0.05	umax = 1.01915</span></span>
<span class="line"><span>Iteration 103	t = 5.15	Δt = 0.05	umax = 1.01929</span></span>
<span class="line"><span>Iteration 104	t = 5.2	Δt = 0.05	umax = 1.01943</span></span>
<span class="line"><span>Iteration 105	t = 5.25	Δt = 0.05	umax = 1.01957</span></span>
<span class="line"><span>Iteration 106	t = 5.3	Δt = 0.05	umax = 1.01971</span></span>
<span class="line"><span>Iteration 107	t = 5.35	Δt = 0.05	umax = 1.01985</span></span>
<span class="line"><span>Iteration 108	t = 5.4	Δt = 0.05	umax = 1.01999</span></span>
<span class="line"><span>Iteration 109	t = 5.45	Δt = 0.05	umax = 1.02012</span></span>
<span class="line"><span>Iteration 110	t = 5.5	Δt = 0.05	umax = 1.02026</span></span>
<span class="line"><span>Iteration 111	t = 5.55	Δt = 0.05	umax = 1.0204</span></span>
<span class="line"><span>Iteration 112	t = 5.6	Δt = 0.05	umax = 1.02054</span></span>
<span class="line"><span>Iteration 113	t = 5.65	Δt = 0.05	umax = 1.02068</span></span>
<span class="line"><span>Iteration 114	t = 5.7	Δt = 0.05	umax = 1.02081</span></span>
<span class="line"><span>Iteration 115	t = 5.75	Δt = 0.05	umax = 1.02095</span></span>
<span class="line"><span>Iteration 116	t = 5.8	Δt = 0.05	umax = 1.02108</span></span>
<span class="line"><span>Iteration 117	t = 5.85	Δt = 0.05	umax = 1.02189</span></span>
<span class="line"><span>Iteration 118	t = 5.9	Δt = 0.05	umax = 1.02374</span></span>
<span class="line"><span>Iteration 119	t = 5.95	Δt = 0.05	umax = 1.02554</span></span>
<span class="line"><span>Iteration 120	t = 6	Δt = 0.05	umax = 1.02726</span></span>
<span class="line"><span>Iteration 121	t = 6.05	Δt = 0.05	umax = 1.0289</span></span>
<span class="line"><span>Iteration 122	t = 6.1	Δt = 0.05	umax = 1.03045</span></span>
<span class="line"><span>Iteration 123	t = 6.15	Δt = 0.05	umax = 1.0319</span></span>
<span class="line"><span>Iteration 124	t = 6.2	Δt = 0.05	umax = 1.03325</span></span>
<span class="line"><span>Iteration 125	t = 6.25	Δt = 0.05	umax = 1.03448</span></span>
<span class="line"><span>Iteration 126	t = 6.3	Δt = 0.05	umax = 1.03559</span></span>
<span class="line"><span>Iteration 127	t = 6.35	Δt = 0.05	umax = 1.03657</span></span>
<span class="line"><span>Iteration 128	t = 6.4	Δt = 0.05	umax = 1.03741</span></span>
<span class="line"><span>Iteration 129	t = 6.45	Δt = 0.05	umax = 1.03811</span></span>
<span class="line"><span>Iteration 130	t = 6.5	Δt = 0.05	umax = 1.03865</span></span>
<span class="line"><span>Iteration 131	t = 6.55	Δt = 0.05	umax = 1.03968</span></span>
<span class="line"><span>Iteration 132	t = 6.6	Δt = 0.05	umax = 1.04095</span></span>
<span class="line"><span>Iteration 133	t = 6.65	Δt = 0.05	umax = 1.04217</span></span>
<span class="line"><span>Iteration 134	t = 6.7	Δt = 0.05	umax = 1.04325</span></span>
<span class="line"><span>Iteration 135	t = 6.75	Δt = 0.05	umax = 1.04419</span></span>
<span class="line"><span>Iteration 136	t = 6.8	Δt = 0.05	umax = 1.04498</span></span>
<span class="line"><span>Iteration 137	t = 6.85	Δt = 0.05	umax = 1.04563</span></span>
<span class="line"><span>Iteration 138	t = 6.9	Δt = 0.05	umax = 1.04612</span></span>
<span class="line"><span>Iteration 139	t = 6.95	Δt = 0.05	umax = 1.04646</span></span>
<span class="line"><span>Iteration 140	t = 7	Δt = 0.05	umax = 1.04664</span></span>
<span class="line"><span>Iteration 141	t = 7.05	Δt = 0.05	umax = 1.04665</span></span>
<span class="line"><span>Iteration 142	t = 7.1	Δt = 0.05	umax = 1.0465</span></span>
<span class="line"><span>Iteration 143	t = 7.15	Δt = 0.05	umax = 1.04619</span></span>
<span class="line"><span>Iteration 144	t = 7.2	Δt = 0.05	umax = 1.04571</span></span>
<span class="line"><span>Iteration 145	t = 7.25	Δt = 0.05	umax = 1.04505</span></span>
<span class="line"><span>Iteration 146	t = 7.3	Δt = 0.05	umax = 1.04423</span></span>
<span class="line"><span>Iteration 147	t = 7.35	Δt = 0.05	umax = 1.04323</span></span>
<span class="line"><span>Iteration 148	t = 7.4	Δt = 0.05	umax = 1.04205</span></span>
<span class="line"><span>Iteration 149	t = 7.45	Δt = 0.05	umax = 1.0407</span></span>
<span class="line"><span>Iteration 150	t = 7.5	Δt = 0.05	umax = 1.03964</span></span>
<span class="line"><span>Iteration 151	t = 7.55	Δt = 0.05	umax = 1.03908</span></span>
<span class="line"><span>Iteration 152	t = 7.6	Δt = 0.05	umax = 1.03837</span></span>
<span class="line"><span>Iteration 153	t = 7.65	Δt = 0.05	umax = 1.0375</span></span>
<span class="line"><span>Iteration 154	t = 7.7	Δt = 0.05	umax = 1.03648</span></span>
<span class="line"><span>Iteration 155	t = 7.75	Δt = 0.05	umax = 1.0353</span></span>
<span class="line"><span>Iteration 156	t = 7.8	Δt = 0.05	umax = 1.03396</span></span>
<span class="line"><span>Iteration 157	t = 7.85	Δt = 0.05	umax = 1.03246</span></span>
<span class="line"><span>Iteration 158	t = 7.9	Δt = 0.05	umax = 1.03079</span></span>
<span class="line"><span>Iteration 159	t = 7.95	Δt = 0.05	umax = 1.02906</span></span>
<span class="line"><span>Iteration 160	t = 8	Δt = 0.05	umax = 1.02806</span></span>
<span class="line"><span>Iteration 161	t = 8.05	Δt = 0.05	umax = 1.02714</span></span>
<span class="line"><span>Iteration 162	t = 8.1	Δt = 0.05	umax = 1.02728</span></span>
<span class="line"><span>Iteration 163	t = 8.15	Δt = 0.05	umax = 1.02742</span></span>
<span class="line"><span>Iteration 164	t = 8.2	Δt = 0.05	umax = 1.02756</span></span>
<span class="line"><span>Iteration 165	t = 8.25	Δt = 0.05	umax = 1.0277</span></span>
<span class="line"><span>Iteration 166	t = 8.3	Δt = 0.05	umax = 1.02784</span></span>
<span class="line"><span>Iteration 167	t = 8.35	Δt = 0.05	umax = 1.02799</span></span>
<span class="line"><span>Iteration 168	t = 8.4	Δt = 0.05	umax = 1.02813</span></span>
<span class="line"><span>Iteration 169	t = 8.45	Δt = 0.05	umax = 1.02828</span></span>
<span class="line"><span>Iteration 170	t = 8.5	Δt = 0.05	umax = 1.02843</span></span>
<span class="line"><span>Iteration 171	t = 8.55	Δt = 0.05	umax = 1.02859</span></span>
<span class="line"><span>Iteration 172	t = 8.6	Δt = 0.05	umax = 1.02874</span></span>
<span class="line"><span>Iteration 173	t = 8.65	Δt = 0.05	umax = 1.0289</span></span>
<span class="line"><span>Iteration 174	t = 8.7	Δt = 0.05	umax = 1.02906</span></span>
<span class="line"><span>Iteration 175	t = 8.75	Δt = 0.05	umax = 1.02922</span></span>
<span class="line"><span>Iteration 176	t = 8.8	Δt = 0.05	umax = 1.02938</span></span>
<span class="line"><span>Iteration 177	t = 8.85	Δt = 0.05	umax = 1.02954</span></span>
<span class="line"><span>Iteration 178	t = 8.9	Δt = 0.05	umax = 1.02971</span></span>
<span class="line"><span>Iteration 179	t = 8.95	Δt = 0.05	umax = 1.02988</span></span>
<span class="line"><span>Iteration 180	t = 9	Δt = 0.05	umax = 1.03006</span></span>
<span class="line"><span>Iteration 181	t = 9.05	Δt = 0.05	umax = 1.03024</span></span>
<span class="line"><span>Iteration 182	t = 9.1	Δt = 0.05	umax = 1.03042</span></span>
<span class="line"><span>Iteration 183	t = 9.15	Δt = 0.05	umax = 1.03061</span></span>
<span class="line"><span>Iteration 184	t = 9.2	Δt = 0.05	umax = 1.0308</span></span>
<span class="line"><span>Iteration 185	t = 9.25	Δt = 0.05	umax = 1.031</span></span>
<span class="line"><span>Iteration 186	t = 9.3	Δt = 0.05	umax = 1.03121</span></span>
<span class="line"><span>Iteration 187	t = 9.35	Δt = 0.05	umax = 1.03142</span></span>
<span class="line"><span>Iteration 188	t = 9.4	Δt = 0.05	umax = 1.03164</span></span>
<span class="line"><span>Iteration 189	t = 9.45	Δt = 0.05	umax = 1.03186</span></span>
<span class="line"><span>Iteration 190	t = 9.5	Δt = 0.05	umax = 1.0321</span></span>
<span class="line"><span>Iteration 191	t = 9.55	Δt = 0.05	umax = 1.03234</span></span>
<span class="line"><span>Iteration 192	t = 9.6	Δt = 0.05	umax = 1.03258</span></span>
<span class="line"><span>Iteration 193	t = 9.65	Δt = 0.05	umax = 1.03284</span></span>
<span class="line"><span>Iteration 194	t = 9.7	Δt = 0.05	umax = 1.0331</span></span>
<span class="line"><span>Iteration 195	t = 9.75	Δt = 0.05	umax = 1.03335</span></span>
<span class="line"><span>Iteration 196	t = 9.8	Δt = 0.05	umax = 1.03363</span></span>
<span class="line"><span>Iteration 197	t = 9.85	Δt = 0.05	umax = 1.03383</span></span>
<span class="line"><span>Iteration 198	t = 9.9	Δt = 0.05	umax = 1.03393</span></span>
<span class="line"><span>Iteration 199	t = 9.95	Δt = 0.05	umax = 1.03391</span></span>
<span class="line"><span>Iteration 200	t = 10	Δt = 0.05	umax = 1.03376</span></span>
<span class="line"><span>Iteration 201	t = 10.05	Δt = 0.05	umax = 1.03334</span></span>
<span class="line"><span>Iteration 202	t = 10.1	Δt = 0.05	umax = 1.03283</span></span>
<span class="line"><span>Iteration 203	t = 10.15	Δt = 0.05	umax = 1.03211</span></span>
<span class="line"><span>Iteration 204	t = 10.2	Δt = 0.05	umax = 1.03113</span></span>
<span class="line"><span>Iteration 205	t = 10.25	Δt = 0.05	umax = 1.03011</span></span>
<span class="line"><span>Iteration 206	t = 10.3	Δt = 0.05	umax = 1.02878</span></span>
<span class="line"><span>Iteration 207	t = 10.35	Δt = 0.05	umax = 1.02736</span></span>
<span class="line"><span>Iteration 208	t = 10.4	Δt = 0.05	umax = 1.02571</span></span>
<span class="line"><span>Iteration 209	t = 10.45	Δt = 0.05	umax = 1.02389</span></span>
<span class="line"><span>Iteration 210	t = 10.5	Δt = 0.05	umax = 1.02194</span></span>
<span class="line"><span>Iteration 211	t = 10.55	Δt = 0.05	umax = 1.01978</span></span>
<span class="line"><span>Iteration 212	t = 10.6	Δt = 0.05	umax = 1.01755</span></span>
<span class="line"><span>Iteration 213	t = 10.65	Δt = 0.05	umax = 1.01509</span></span>
<span class="line"><span>Iteration 214	t = 10.7	Δt = 0.05	umax = 1.01261</span></span>
<span class="line"><span>Iteration 215	t = 10.75	Δt = 0.05	umax = 1.00993</span></span>
<span class="line"><span>Iteration 216	t = 10.8	Δt = 0.05	umax = 1.00722</span></span>
<span class="line"><span>Iteration 217	t = 10.85	Δt = 0.05	umax = 1.00437</span></span>
<span class="line"><span>Iteration 218	t = 10.9	Δt = 0.05	umax = 1.00146</span></span>
<span class="line"><span>Iteration 219	t = 10.95	Δt = 0.05	umax = 0.998497</span></span>
<span class="line"><span>Iteration 220	t = 11	Δt = 0.05	umax = 0.995456</span></span>
<span class="line"><span>Iteration 221	t = 11.05	Δt = 0.05	umax = 0.992388</span></span>
<span class="line"><span>Iteration 222	t = 11.1	Δt = 0.05	umax = 0.993041</span></span>
<span class="line"><span>Iteration 223	t = 11.15	Δt = 0.05	umax = 0.994971</span></span>
<span class="line"><span>Iteration 224	t = 11.2	Δt = 0.05	umax = 0.996932</span></span>
<span class="line"><span>Iteration 225	t = 11.25	Δt = 0.05	umax = 0.998921</span></span>
<span class="line"><span>Iteration 226	t = 11.3	Δt = 0.05	umax = 1.00093</span></span>
<span class="line"><span>Iteration 227	t = 11.35	Δt = 0.05	umax = 1.00297</span></span>
<span class="line"><span>Iteration 228	t = 11.4	Δt = 0.05	umax = 1.00502</span></span>
<span class="line"><span>Iteration 229	t = 11.45	Δt = 0.05	umax = 1.00708</span></span>
<span class="line"><span>Iteration 230	t = 11.5	Δt = 0.05	umax = 1.00914</span></span>
<span class="line"><span>Iteration 231	t = 11.55	Δt = 0.05	umax = 1.0112</span></span>
<span class="line"><span>Iteration 232	t = 11.6	Δt = 0.05	umax = 1.01326</span></span>
<span class="line"><span>Iteration 233	t = 11.65	Δt = 0.05	umax = 1.0153</span></span>
<span class="line"><span>Iteration 234	t = 11.7	Δt = 0.05	umax = 1.01731</span></span>
<span class="line"><span>Iteration 235	t = 11.75	Δt = 0.05	umax = 1.0193</span></span>
<span class="line"><span>Iteration 236	t = 11.8	Δt = 0.05	umax = 1.02125</span></span>
<span class="line"><span>Iteration 237	t = 11.85	Δt = 0.05	umax = 1.02315</span></span>
<span class="line"><span>Iteration 238	t = 11.9	Δt = 0.05	umax = 1.02499</span></span>
<span class="line"><span>Iteration 239	t = 11.95	Δt = 0.05	umax = 1.02677</span></span>
<span class="line"><span>Iteration 240	t = 12	Δt = 0.05	umax = 1.02848</span></span></code></pre></div><h2 id="post-process" tabindex="-1">Post-process <a class="header-anchor" href="#post-process" aria-label="Permalink to &quot;Post-process&quot;">​</a></h2><p>We may visualize or export the computed fields <code>(u, p)</code>.</p><p>Export to VTK</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">save_vtk</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(setup, state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">u, state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">t, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">$output</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">/solution&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>1-element Vector{String}:</span></span>
<span class="line"><span> &quot;output/Actuator2D/solution.vtr&quot;</span></span></code></pre></div><p>We create a box to visualize the actuator.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">box </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    [xc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> δ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, xc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> δ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, xc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> δ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, xc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> δ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, xc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> δ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    [yc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> D </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, yc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> D </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, yc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> D </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, yc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> D </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, yc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> D </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>([1.945, 1.945, 2.055, 2.055, 1.945], [0.5, -0.5, -0.5, 0.5, 0.5])</span></span></code></pre></div><p>Plot pressure</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">fig </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> fieldplot</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(state; setup, fieldname </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> :pressure</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lines!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(box</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; color </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> :red</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">fig</span></span></code></pre></div><p><img src="`+p+`" alt=""></p><p>Plot velocity</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">fig </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> fieldplot</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(state; setup, fieldname </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> :velocitynorm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lines!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(box</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; color </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> :red</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">fig</span></span></code></pre></div><p><img src="`+l+`" alt=""></p><p>Plot vorticity</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">fig </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> fieldplot</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(state; setup, fieldname </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> :vorticity</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lines!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(box</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; color </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> :red</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">fig</span></span></code></pre></div><p><img src="`+e+'" alt=""></p><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>',42),r=[k];function d(o,c,E,g,u,y){return t(),a("div",null,r)}const x=s(h,[["render",d]]);export{F as __pageData,x as default};
