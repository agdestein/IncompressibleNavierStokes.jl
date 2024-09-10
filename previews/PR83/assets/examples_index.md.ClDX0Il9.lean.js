import{d as l,o as n,c as s,j as e,k as m,g as y,t as g,_ as u,F as f,C as D,b,K as w,a as t,G as i}from"./chunks/framework.B4r91vhS.js";const x={class:"img-box"},v=["href"],_=["src"],k={class:"transparent-box1"},T={class:"caption"},N={class:"transparent-box2"},S={class:"subcaption"},B={class:"opacity-low"},C=l({__name:"GalleryImage",props:{href:{},src:{},caption:{},desc:{}},setup(p){return(r,c)=>(n(),s("div",x,[e("a",{href:r.href},[e("img",{src:m(y)(r.src),height:"150px",alt:""},null,8,_),e("div",k,[e("div",T,[e("h2",null,g(r.caption),1)])]),e("div",N,[e("div",S,[e("p",B,g(r.desc),1)])])],8,v)]))}}),F=u(C,[["__scopeId","data-v-d6324755"]]),G={class:"gallery-image"},P=l({__name:"Gallery",props:{images:{}},setup(p){return(r,c)=>(n(),s("div",G,[(n(!0),s(f,null,D(r.images,d=>(n(),b(F,w({ref_for:!0},d),null,16))),256))]))}}),o=u(P,[["__scopeId","data-v-ca4fdbe5"]]),V=JSON.parse('{"title":"Examples","description":"","frontmatter":{},"headers":[],"relativePath":"examples/index.md","filePath":"examples/index.md","lastUpdated":null}'),R={name:"examples/index.md"},q=l({...R,setup(p){const r=[{href:"generated/DecayingTurbulence2D",src:"../DecayingTurbulence2D.gif",caption:"Decaying turbulence (2D)",desc:"Decaying turbulence in a periodic 2D-box"},{href:"generated/DecayingTurbulence3D",src:"../DecayingTurbulence3D.png",caption:"Decaying turbulence (3D)",desc:"Decaying turbulence in a periodic 3D-box"},{href:"generated/TaylorGreenVortex2D",src:"../TaylorGreenVortex2D.png",caption:"Taylor-Green vortex (2D)",desc:"Decaying vortex structures in a periodic 2D-box"},{href:"generated/TaylorGreenVortex3D",src:"../TaylorGreenVortex3D.png",caption:"Taylor-Green vortex (3D)",desc:"Decaying vortex structures in a periodic 3D-box"},{href:"generated/Kolmogorov2D",src:"../Kolmogorov2D.png",caption:"Kolmogorov flow (2D)",desc:"Initiate a flow through a sinusoidal force"},{href:"generated/ShearLayer2D",src:"../ShearLayer2D.png",caption:"Shear-layer (2D)",desc:"Two layers with different velocities"},{href:"generated/PlanarMixing2D",src:"../PlanarMixing2D.png",caption:"Planar mixing (2D)",desc:"Planar mixing layers"}],c=[{href:"generated/Actuator2D",src:"../Actuator2D.gif",caption:"Actuator (2D)",desc:"Unsteady inflow around an actuator disk"},{href:"generated/Actuator3D",src:"../Actuator3D.png",caption:"Actuator (3D)",desc:"Unsteady inflow around an actuator disk"},{href:"generated/BackwardFacingStep2D",src:"../BackwardFacingStep2D.png",caption:"Backward Facing Step (2D)",desc:"Flow past a backward facing step"},{href:"generated/BackwardFacingStep3D",src:"../BackwardFacingStep3D.png",caption:"Backward Facing Step (3D)",desc:"Flow past a backward facing step"},{href:"generated/LidDrivenCavity2D",src:"../LidDrivenCavity2D.png",caption:"Lid-driven cavity (2D)",desc:"Generate a flow caused by a moving lid"},{href:"generated/LidDrivenCavity3D",src:"../LidDrivenCavity3D.png",caption:"Lid-driven cavity (3D)",desc:"Generate a flow caused by a moving lid"},{href:"generated/MultiActuator",src:"../MultiActuator.png",desc:"Multi-actuator (2D)",caption:"Unsteady inflow around multiple actuator disks"}],d=[{href:"generated/RayleighBenard2D",src:"../RayleighBenard2D.gif",caption:"Rayleigh-Bénard convection (2D)",desc:"Convection generated by a temperature gradient between a hot and a cold plate"},{href:"generated/RayleighBenard3D",src:"../RayleighBenard3D.gif",caption:"Rayleigh-Bénard convection (3D)",desc:"Convection generated by a temperature gradient between a hot and a cold plate"},{href:"generated/RayleighTaylor2D",src:"../RayleighTaylor2D.gif",caption:"Rayleigh-Taylor instability (2D)",desc:"Convection generated by a temperature gradient between hot and cold fluids"},{href:"generated/RayleighTaylor3D",src:"../RayleighTaylor3D.png",caption:"Rayleigh-Taylor instability (3D)",desc:"Convection generated by a temperature gradient between hot and cold fluids"}],h=[{href:"generated/prioranalysis",src:"../prioranalysis.png",caption:"Filter analysis",desc:"Compare discrete filters and their properties with DNS data"},{href:"generated/postanalysis",src:"../postanalysis.png",caption:"CNN closure models",desc:"Compare CNN closure models for different filters, grid sizes, and projection orders"},{href:"generated/symmetryanalysis",src:"../symmetryanalysis.png",caption:"Symmetry-preserving closure models",desc:"Train group equivariant CNNs and compare to normal CNNs"}];return(L,a)=>(n(),s("div",null,[a[0]||(a[0]=e("h1",{id:"examples",tabindex:"-1"},[t("Examples "),e("a",{class:"header-anchor",href:"#examples","aria-label":'Permalink to "Examples"'},"​")],-1)),a[1]||(a[1]=e("h2",{id:"Simple-flows",tabindex:"-1"},[t("Simple flows "),e("a",{class:"header-anchor",href:"#Simple-flows","aria-label":'Permalink to "Simple flows {#Simple-flows}"'},"​")],-1)),i(o,{images:r}),a[2]||(a[2]=e("h2",{id:"Flows-with-mixed-boundary-conditions",tabindex:"-1"},[t("Flows with mixed boundary conditions "),e("a",{class:"header-anchor",href:"#Flows-with-mixed-boundary-conditions","aria-label":'Permalink to "Flows with mixed boundary conditions {#Flows-with-mixed-boundary-conditions}"'},"​")],-1)),i(o,{images:c}),a[3]||(a[3]=e("h2",{id:"With-a-temperature-equation",tabindex:"-1"},[t("With a temperature equation "),e("a",{class:"header-anchor",href:"#With-a-temperature-equation","aria-label":'Permalink to "With a temperature equation {#With-a-temperature-equation}"'},"​")],-1)),i(o,{images:d}),a[4]||(a[4]=e("h2",{id:"Neural-network-closure-models",tabindex:"-1"},[t("Neural network closure models "),e("a",{class:"header-anchor",href:"#Neural-network-closure-models","aria-label":'Permalink to "Neural network closure models {#Neural-network-closure-models}"'},"​")],-1)),i(o,{images:h})]))}});export{V as __pageData,q as default};
