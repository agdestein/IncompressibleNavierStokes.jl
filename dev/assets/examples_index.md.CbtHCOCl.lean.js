import{d,o as r,c as s,j as e,k as h,g as m,t as p,_ as u,F as y,E as D,b as f,M as _,I as n,a as t}from"./chunks/framework.NoRTQVDn.js";const b={class:"img-box"},w=["href"],x=["src"],v={class:"transparent-box1"},k={class:"caption"},T={class:"transparent-box2"},N={class:"subcaption"},S={class:"opacity-low"},B=d({__name:"GalleryImage",props:{href:{},src:{},caption:{},desc:{}},setup(l){return(a,o)=>(r(),s("div",b,[e("a",{href:a.href},[e("img",{src:h(m)(a.src),height:"150px",alt:""},null,8,x),e("div",v,[e("div",k,[e("h2",null,p(a.caption),1)])]),e("div",T,[e("div",N,[e("p",S,p(a.desc),1)])])],8,w)]))}}),F=u(B,[["__scopeId","data-v-d6324755"]]),C={class:"gallery-image"},G=d({__name:"Gallery",props:{images:{}},setup(l){return(a,o)=>(r(),s("div",C,[(r(!0),s(y,null,D(a.images,c=>(r(),f(F,_({ref_for:!0},c),null,16))),256))]))}}),i=u(G,[["__scopeId","data-v-ca4fdbe5"]]),P=e("h1",{id:"examples",tabindex:"-1"},[t("Examples "),e("a",{class:"header-anchor",href:"#examples","aria-label":'Permalink to "Examples"'},"​")],-1),R=e("h2",{id:"Simple-flows",tabindex:"-1"},[t("Simple flows "),e("a",{class:"header-anchor",href:"#Simple-flows","aria-label":'Permalink to "Simple flows {#Simple-flows}"'},"​")],-1),L=e("h2",{id:"Flows-with-mixed-boundary-conditions",tabindex:"-1"},[t("Flows with mixed boundary conditions "),e("a",{class:"header-anchor",href:"#Flows-with-mixed-boundary-conditions","aria-label":'Permalink to "Flows with mixed boundary conditions {#Flows-with-mixed-boundary-conditions}"'},"​")],-1),A=e("h2",{id:"With-a-temperature-equation",tabindex:"-1"},[t("With a temperature equation "),e("a",{class:"header-anchor",href:"#With-a-temperature-equation","aria-label":'Permalink to "With a temperature equation {#With-a-temperature-equation}"'},"​")],-1),$=e("h2",{id:"Neural-network-closure-models",tabindex:"-1"},[t("Neural network closure models "),e("a",{class:"header-anchor",href:"#Neural-network-closure-models","aria-label":'Permalink to "Neural network closure models {#Neural-network-closure-models}"'},"​")],-1),I=JSON.parse('{"title":"Examples","description":"","frontmatter":{},"headers":[],"relativePath":"examples/index.md","filePath":"examples/index.md","lastUpdated":null}'),V={name:"examples/index.md"},W=d({...V,setup(l){const a=[{href:"generated/DecayingTurbulence2D",src:"../DecayingTurbulence2D.mp4",caption:"Decaying turbulence (2D)",desc:"Decaying turbulence in a periodic 2D-box"},{href:"generated/DecayingTurbulence3D",src:"../DecayingTurbulence3D.png",caption:"Decaying turbulence (3D)",desc:"Decaying turbulence in a periodic 3D-box"},{href:"generated/TaylorGreenVortex2D",src:"../TaylorGreenVortex2D.png",caption:"Taylor-Green vortex (2D)",desc:"Decaying vortex structures in a periodic 2D-box"},{href:"generated/TaylorGreenVortex3D",src:"../TaylorGreenVortex3D.png",caption:"Taylor-Green vortex (3D)",desc:"Decaying vortex structures in a periodic 3D-box"},{href:"generated/ShearLayer2D",src:"../ShearLayer2D.png",caption:"Shear-layer (2D)",desc:"Two layers with different velocities"},{href:"generated/PlanarMixing2D",src:"../PlanarMixing2D.png",caption:"Planar mixing (2D)",desc:"Planar mixing layers"}],o=[{href:"generated/Actuator2D",src:"../Actuator2D.mp4",caption:"Actuator (2D)",desc:"Unsteady inflow around an actuator disk"},{href:"generated/Actuator3D",src:"../Actuator3D.png",caption:"Actuator (3D)",desc:"Unsteady inflow around an actuator disk"},{href:"generated/BackwardFacingStep2D",src:"../BackwardFacingStep2D.png",caption:"Backward Facing Step (2D)",desc:"Flow past a backward facing step"},{href:"generated/BackwardFacingStep3D",src:"../BackwardFacingStep3D.png",caption:"Backward Facing Step (3D)",desc:"Flow past a backward facing step"},{href:"generated/LidDrivenCavity2D",src:"../LidDrivenCavity2D.png",caption:"Lid-driven cavity (2D)",desc:"Generate a flow caused by a moving lid"},{href:"generated/LidDrivenCavity3D",src:"../LidDrivenCavity3D.png",caption:"Lid-driven cavity (3D)",desc:"Generate a flow caused by a moving lid"},{href:"generated/MultiActuator",src:"../MultiActuator.png",desc:"Multi-actuator (2D)",caption:"Unsteady inflow around multiple actuator disks"}],c=[{href:"generated/RayleighBenard2D",src:"../RayleighBenard2D.mp4",caption:"Rayleigh-Bénard convection (2D)",desc:"Convection generated by a temperature gradient between a hot and a cold plate"},{href:"generated/RayleighBenard3D",src:"../RayleighBenard3D.png",caption:"Rayleigh-Bénard convection (3D)",desc:"Convection generated by a temperature gradient between a hot and a cold plate"},{href:"generated/RayleighTaylor2D",src:"../RayleighTaylor2D.mp4",caption:"Rayleigh-Taylor instability (2D)",desc:"Convection generated by a temperature gradient between hot and cold fluids"},{href:"generated/RayleighTaylor3D",src:"../RayleighTaylor3D.png",caption:"Rayleigh-Taylor instability (3D)",desc:"Convection generated by a temperature gradient between hot and cold fluids"}],g=[{href:"generated/prioranalysis",src:"../prioranalysis.png",caption:"Filter analysis",desc:"Compare discrete filters and their properties with DNS data"},{href:"generated/postanalysis",src:"../postanalysis.png",caption:"CNN closure models",desc:"Compare CNN closure models for different filters, grid sizes, and projection orders"},{href:"generated/symmetryanalysis",src:"../symmetryanalysis.png",caption:"Symmetry-preserving closure models",desc:"Train group equivariant CNNs and compare to normal CNNs"}];return(q,M)=>(r(),s("div",null,[P,R,n(i,{images:a}),L,n(i,{images:o}),A,n(i,{images:c}),$,n(i,{images:g})]))}});export{I as __pageData,W as default};
