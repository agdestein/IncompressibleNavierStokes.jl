import{_ as a,c as t,a5 as o,o as n}from"./chunks/framework.C9D4Xw9z.js";const u=JSON.parse('{"title":"Local development","description":"","frontmatter":{},"headers":[],"relativePath":"about/development.md","filePath":"about/development.md","lastUpdated":null}'),i={name:"about/development.md"};function s(l,e,d,r,c,p){return n(),t("div",null,e[0]||(e[0]=[o(`<h1 id="Local-development" tabindex="-1">Local development <a class="header-anchor" href="#Local-development" aria-label="Permalink to &quot;Local development {#Local-development}&quot;">​</a></h1><h2 id="Use-Juliaup" tabindex="-1">Use Juliaup <a class="header-anchor" href="#Use-Juliaup" aria-label="Permalink to &quot;Use Juliaup {#Use-Juliaup}&quot;">​</a></h2><p>Install Julia using the <a href="https://julialang.org/downloads/" target="_blank" rel="noreferrer">juliaup</a> version manager. This allows for choosing the Julia version, e.g. v1.11, by running</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>juliaup add 1.11</span></span>
<span class="line"><span>juliaup default 1.11</span></span></code></pre></div><h2 id="revise" tabindex="-1">Revise <a class="header-anchor" href="#revise" aria-label="Permalink to &quot;Revise&quot;">​</a></h2><p>It is recommended to use <a href="https://github.com/timholy/Revise.jl" target="_blank" rel="noreferrer">Revise.jl</a> for interactive development. Add it to your global environment with</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>julia -e &#39;using Pkg; Pkg.add(&quot;Revise&quot;)&#39;</span></span></code></pre></div><p>and load it in the startup file (create the file and folder if it is not already there) at <code>~/.julia/config/startup.jl</code> with</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>using Revise</span></span></code></pre></div><p>Then changes to the IncompressibleNavierStokes modules are detected and reloaded live.</p><h2 id="environments" tabindex="-1">Environments <a class="header-anchor" href="#environments" aria-label="Permalink to &quot;Environments&quot;">​</a></h2><p>To keep dependencies sparse, there are multiple <code>Project.toml</code> files in this repository, specifying environments. For example, the <a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/main/docs/Project.toml" target="_blank" rel="noreferrer">docs</a> environment contains packages that are required to build documentation, but not needed to run the simulations. To add local packages to an environment and be detectable by Revise, they need to be <code>Pkg.develop</code>ed. For example, the package <a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/main/lib/NeuralClosure/Project.toml" target="_blank" rel="noreferrer">NeuralClosure</a> depends on IncompressibleNavierStokes, and IncompressibleNavierStokes needs to be <code>dev</code>ed with</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>julia --project=lib/NeuralClosure -e &#39;using Pkg; Pkg.develop(PackageSpec(; path = &quot;.&quot;))&#39;</span></span></code></pre></div><p>Run this from the repository root, where <code>&quot;.&quot;</code> is the path to IncompressibleNavierStokes.</p><p>On Julia v1.11, this linking is automatic, with the dedicated <code>[sources]</code> sections in the <code>Project.toml</code> files. In that case, an environment can be instantiated with</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>julia --project=lib/NeuralClosure -e &#39;using Pkg; Pkg.instantiate()&#39;</span></span></code></pre></div><p>etc., or interactively from the REPL with <code>] instantiate</code>.</p><h3 id="vscode" tabindex="-1">VSCode <a class="header-anchor" href="#vscode" aria-label="Permalink to &quot;VSCode&quot;">​</a></h3><p>In VSCode, you can choose an active environment by clicking on the <code>Julia env:</code> button in the status bar, or press <code>ctrl</code>/<code>cmd</code> + <code>shift</code> + <code>p</code> and start typing <code>environment</code>:</p><ul><li><p><code>Julia: Activate this environment</code> activates the one of the current open file</p></li><li><p><code>Julia: Change current environment</code> otherwise</p></li></ul><p>Then scripts will be run from the selected environment.</p><h3 id="Environment-vs-package" tabindex="-1">Environment vs package <a class="header-anchor" href="#Environment-vs-package" aria-label="Permalink to &quot;Environment vs package {#Environment-vs-package}&quot;">​</a></h3><p>If a <code>Project.toml</code> has a header with a <code>name</code> and <code>uuid</code>, then it is a package with the module <code>src/ModuleName.jl</code>, and can be depended on in other projects (by <code>add</code> or <code>dev</code>).</p>`,23)]))}const v=a(i,[["render",s]]);export{u as __pageData,v as default};
