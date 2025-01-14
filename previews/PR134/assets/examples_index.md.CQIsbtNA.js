import{d as l,o as r,c as n,j as e,k as h,g as m,t as g,_ as u,F as f,C as y,b as D,K as b,a5 as x,G as s,a as p}from"./chunks/framework.CfPBTMA0.js";const _={class:"img-box"},v=["href"],w=["src"],k={class:"transparent-box1"},T={class:"caption"},B={class:"transparent-box2"},S={class:"subcaption"},F={class:"opacity-low"},G=l({__name:"GalleryImage",props:{href:{},src:{},caption:{},desc:{}},setup(d){return(a,i)=>(r(),n("div",_,[e("a",{href:a.href},[e("img",{src:h(m)(a.src),height:"100px",alt:""},null,8,w),e("div",k,[e("div",T,[e("h2",null,g(a.caption),1)])]),e("div",B,[e("div",S,[e("p",F,g(a.desc),1)])])],8,v)]))}}),R=u(G,[["__scopeId","data-v-f778be06"]]),P={class:"gallery-image"},q=l({__name:"Gallery",props:{images:{}},setup(d){return(a,i)=>(r(),n("div",P,[(r(!0),n(f,null,y(a.images,o=>(r(),D(R,b({ref_for:!0},o),null,16))),256))]))}}),c=u(q,[["__scopeId","data-v-9f22d17b"]]),A=JSON.parse('{"title":"Examples gallery","description":"","frontmatter":{},"headers":[],"relativePath":"examples/index.md","filePath":"examples/index.md","lastUpdated":null}'),C={name:"examples/index.md"},E=l({...C,setup(d){const a=[{href:"generated/DecayingTurbulence2D",src:"../DecayingTurbulence2D.gif",caption:"Decaying turbulence (2D)",desc:"Decaying turbulence in a periodic 2D-box"},{href:"generated/DecayingTurbulence3D",src:"../DecayingTurbulence3D.png",caption:"Decaying turbulence (3D)",desc:"Decaying turbulence in a periodic 3D-box"},{href:"generated/TaylorGreenVortex2D",src:"../TaylorGreenVortex2D.png",caption:"Taylor-Green vortex (2D)",desc:"Decaying vortex structures in a periodic 2D-box"},{href:"generated/TaylorGreenVortex3D",src:"../TaylorGreenVortex3D.png",caption:"Taylor-Green vortex (3D)",desc:"Decaying vortex structures in a periodic 3D-box"},{href:"generated/Kolmogorov2D",src:"../logo.svg",caption:"Kolmogorov flow (2D)",desc:"Initiate a flow through a sinusoidal force"},{href:"generated/ShearLayer2D",src:"../logo.svg",caption:"Shear-layer (2D)",desc:"Two layers with different velocities"},{href:"generated/PlanarMixing2D",src:"../logo.svg",caption:"Planar mixing (2D)",desc:"Planar mixing layers"}],i=[{href:"generated/Actuator2D",src:"../Actuator2D.gif",caption:"Actuator (2D)",desc:"Unsteady inflow around an actuator disk"},{href:"generated/Actuator3D",src:"../Actuator3D.png",caption:"Actuator (3D)",desc:"Unsteady inflow around an actuator disk"},{href:"generated/BackwardFacingStep2D",src:"../BackwardFacingStep2D.png",caption:"Backward Facing Step (2D)",desc:"Flow past a backward facing step"},{href:"generated/BackwardFacingStep3D",src:"../BackwardFacingStep3D.png",caption:"Backward Facing Step (3D)",desc:"Flow past a backward facing step"},{href:"generated/LidDrivenCavity2D",src:"../logo.svg",caption:"Lid-driven cavity (2D)",desc:"Generate a flow caused by a moving lid"},{href:"generated/LidDrivenCavity3D",src:"../logo.svg",caption:"Lid-driven cavity (3D)",desc:"Generate a flow caused by a moving lid"},{href:"generated/MultiActuator",src:"../logo.svg",desc:"Multi-actuator (2D)",caption:"Unsteady inflow around multiple actuator disks"}],o=[{href:"generated/RayleighBenard2D",src:"../RayleighBenard2D.gif",caption:"Rayleigh-Bénard convection (2D)",desc:"Convection generated by a temperature gradient between a hot and a cold plate"},{href:"generated/RayleighBenard3D",src:"../RayleighBenard3D.gif",caption:"Rayleigh-Bénard convection (3D)",desc:"Convection generated by a temperature gradient between a hot and a cold plate"},{href:"generated/RayleighTaylor2D",src:"../RayleighTaylor2D.gif",caption:"Rayleigh-Taylor instability (2D)",desc:"Convection generated by a temperature gradient between hot and cold fluids"},{href:"generated/RayleighTaylor3D",src:"../logo.svg",caption:"Rayleigh-Taylor instability (3D)",desc:"Convection generated by a temperature gradient between hot and cold fluids"}];return(L,t)=>(r(),n("div",null,[t[0]||(t[0]=x('<h1 id="Examples-gallery" tabindex="-1">Examples gallery <a class="header-anchor" href="#Examples-gallery" aria-label="Permalink to &quot;Examples gallery {#Examples-gallery}&quot;">​</a></h1><p>Here is a gallery of selected commented example simulations. The outputs are generated with <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a> and displayed inline. Copy-pasteable code is available at the bottom of each example. The raw Julia source scripts can be found in the <a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/tree/main/examples" target="_blank" rel="noreferrer">examples</a> folder.</p><h2 id="Simple-flows" tabindex="-1">Simple flows <a class="header-anchor" href="#Simple-flows" aria-label="Permalink to &quot;Simple flows {#Simple-flows}&quot;">​</a></h2>',3)),s(c,{images:a}),t[1]||(t[1]=e("h2",{id:"Flows-with-mixed-boundary-conditions",tabindex:"-1"},[p("Flows with mixed boundary conditions "),e("a",{class:"header-anchor",href:"#Flows-with-mixed-boundary-conditions","aria-label":'Permalink to "Flows with mixed boundary conditions {#Flows-with-mixed-boundary-conditions}"'},"​")],-1)),s(c,{images:i}),t[2]||(t[2]=e("h2",{id:"With-a-temperature-equation",tabindex:"-1"},[p("With a temperature equation "),e("a",{class:"header-anchor",href:"#With-a-temperature-equation","aria-label":'Permalink to "With a temperature equation {#With-a-temperature-equation}"'},"​")],-1)),s(c,{images:o})]))}});export{A as __pageData,E as default};
