import{_ as t,c as s,a5 as o,o as a}from"./chunks/framework.CWYYQxkc.js";const i="/IncompressibleNavierStokes.jl/previews/PR89/assets/resolution.DAYTxiG0.png",f=JSON.parse('{"title":"Large eddy simulation","description":"","frontmatter":{},"headers":[],"relativePath":"manual/les.md","filePath":"manual/les.md","lastUpdated":null}'),r={name:"manual/les.md"};function l(n,e,d,c,m,u){return a(),s("div",null,e[0]||(e[0]=[o('<h1 id="Large-eddy-simulation" tabindex="-1">Large eddy simulation <a class="header-anchor" href="#Large-eddy-simulation" aria-label="Permalink to &quot;Large eddy simulation {#Large-eddy-simulation}&quot;">​</a></h1><p>Depending on the problem specification, a given grid resolution may not be sufficient to resolve all spatial features of the flow. Consider the following example:</p><p><img src="'+i+'" alt=""></p><p>On the left, the grid spacing is too large to capt the smallest eddies in the flow. These eddies create sub-grid stresses that also affect the large scale features. The grid must be refined if we want to compute these stresses exactly.</p><p>On the right, the smallest spatial feature of the flow is fully resolved, and there are no sub-grid stresses. The equations can be solved without worrying about errors from unresolved features. This is known as <em>Direct Numerical Simulation</em> (DNS).</p><p>If refining the grid is too costly, a closure model can be used to predict the sub-grid stresses. The models only give an estimate for these stresses, and may need to be calibrated to the given problem. When used correctly, they can predict the evolution of the large fluid motions without computing the sub-grid motions themselves. This is known as <em>Large Eddy Simulation</em> (LES).</p><p>Eddy viscosity models add a local contribution to the global baseline viscosity. The baseline viscosity models transfer of energy from resolved to atomic scales. The new turbulent viscosity on the other hand, models energy transfer from resolved to unresolved scales. This non-constant field is computed from the local velocity field.</p>',7)]))}const p=t(r,[["render",l]]);export{f as __pageData,p as default};
