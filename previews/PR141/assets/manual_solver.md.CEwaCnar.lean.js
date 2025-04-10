import{_ as r,C as d,c as o,o as p,j as e,a as i,ai as l,G as a,w as n}from"./chunks/framework.Clo3yZS-.js";const B=JSON.parse('{"title":"Solvers","description":"","frontmatter":{},"headers":[],"relativePath":"manual/solver.md","filePath":"manual/solver.md","lastUpdated":null}'),h={name:"manual/solver.md"},k={class:"jldocstring custom-block",open:""},T={class:"jldocstring custom-block",open:""},c={class:"jldocstring custom-block",open:""},m={class:"jldocstring custom-block",open:""},g={class:"jldocstring custom-block",open:""},u={class:"jldocstring custom-block",open:""},Q={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},y={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.09ex"},xmlns:"http://www.w3.org/2000/svg",width:"5.703ex",height:"1.636ex",role:"img",focusable:"false",viewBox:"0 -683 2520.6 723","aria-hidden":"true"},E={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},b={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-2.428ex"},xmlns:"http://www.w3.org/2000/svg",width:"26.783ex",height:"5.507ex",role:"img",focusable:"false",viewBox:"0 -1361 11837.9 2434.1","aria-hidden":"true"},f={class:"jldocstring custom-block",open:""},v={class:"jldocstring custom-block",open:""},_={class:"jldocstring custom-block",open:""},j={class:"jldocstring custom-block",open:""},C={class:"jldocstring custom-block",open:""},S={class:"jldocstring custom-block",open:""},x={class:"jldocstring custom-block",open:""},F={class:"jldocstring custom-block",open:""},I={class:"jldocstring custom-block",open:""},w={class:"jldocstring custom-block",open:""};function A(L,s,N,H,V,D){const t=d("Badge");return p(),o("div",null,[s[70]||(s[70]=e("h1",{id:"solvers",tabindex:"-1"},[i("Solvers "),e("a",{class:"header-anchor",href:"#solvers","aria-label":'Permalink to "Solvers"'},"​")],-1)),s[71]||(s[71]=e("h2",{id:"Solvers-2",tabindex:"-1"},[i("Solvers "),e("a",{class:"header-anchor",href:"#Solvers-2","aria-label":'Permalink to "Solvers {#Solvers-2}"'},"​")],-1)),e("details",k,[e("summary",null,[s[0]||(s[0]=e("a",{id:"IncompressibleNavierStokes.get_cfl_timestep!-Tuple{Any, Any, Any}",href:"#IncompressibleNavierStokes.get_cfl_timestep!-Tuple{Any, Any, Any}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.get_cfl_timestep!")],-1)),s[1]||(s[1]=i()),a(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[3]||(s[3]=l("",2)),a(t,{type:"info",class:"source-link",text:"source"},{default:n(()=>s[2]||(s[2]=[e("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fd1f8e19740acedf01928865d15a70ab4a3a4686/src/solver.jl#L100",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),e("details",T,[e("summary",null,[s[4]||(s[4]=e("a",{id:"IncompressibleNavierStokes.get_state-Tuple{Any}",href:"#IncompressibleNavierStokes.get_state-Tuple{Any}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.get_state")],-1)),s[5]||(s[5]=i()),a(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[7]||(s[7]=l("",2)),a(t,{type:"info",class:"source-link",text:"source"},{default:n(()=>s[6]||(s[6]=[e("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fd1f8e19740acedf01928865d15a70ab4a3a4686/src/solver.jl#L94",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),e("details",c,[e("summary",null,[s[8]||(s[8]=e("a",{id:"IncompressibleNavierStokes.solve_unsteady-Tuple{}",href:"#IncompressibleNavierStokes.solve_unsteady-Tuple{}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.solve_unsteady")],-1)),s[9]||(s[9]=i()),a(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[11]||(s[11]=l("",6)),a(t,{type:"info",class:"source-link",text:"source"},{default:n(()=>s[10]||(s[10]=[e("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fd1f8e19740acedf01928865d15a70ab4a3a4686/src/solver.jl#L1",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s[72]||(s[72]=e("h2",{id:"processors",tabindex:"-1"},[i("Processors "),e("a",{class:"header-anchor",href:"#processors","aria-label":'Permalink to "Processors"'},"​")],-1)),s[73]||(s[73]=e("p",null,[i("Processors can be used to process the solution in "),e("a",{href:"/IncompressibleNavierStokes.jl/previews/PR141/manual/solver#IncompressibleNavierStokes.solve_unsteady-Tuple{}"},[e("code",null,"solve_unsteady")]),i(" after every time step.")],-1)),e("details",m,[e("summary",null,[s[12]||(s[12]=e("a",{id:"IncompressibleNavierStokes.animator",href:"#IncompressibleNavierStokes.animator"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.animator")],-1)),s[13]||(s[13]=i()),a(t,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),s[15]||(s[15]=l("",3)),a(t,{type:"info",class:"source-link",text:"source"},{default:n(()=>s[14]||(s[14]=[e("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fd1f8e19740acedf01928865d15a70ab4a3a4686/src/processors.jl#L350-L362",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),e("details",g,[e("summary",null,[s[16]||(s[16]=e("a",{id:"IncompressibleNavierStokes.energy_history_plot",href:"#IncompressibleNavierStokes.energy_history_plot"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.energy_history_plot")],-1)),s[17]||(s[17]=i()),a(t,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),s[19]||(s[19]=e("p",null,"Create energy history plot.",-1)),a(t,{type:"info",class:"source-link",text:"source"},{default:n(()=>s[18]||(s[18]=[e("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fd1f8e19740acedf01928865d15a70ab4a3a4686/src/processors.jl#L410",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),e("details",u,[e("summary",null,[s[20]||(s[20]=e("a",{id:"IncompressibleNavierStokes.energy_spectrum_plot",href:"#IncompressibleNavierStokes.energy_spectrum_plot"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.energy_spectrum_plot")],-1)),s[21]||(s[21]=i()),a(t,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e("p",null,[s[24]||(s[24]=i("Create energy spectrum plot. The energy at a scalar wavenumber level ")),e("mjx-container",Q,[(p(),o("svg",y,s[22]||(s[22]=[l("",1)]))),s[23]||(s[23]=e("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[e("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[e("mi",null,"κ"),e("mo",null,"∈"),e("mrow",{"data-mjx-texclass":"ORD"},[e("mi",{mathvariant:"double-struck"},"N")])])],-1))]),s[25]||(s[25]=i(" is defined by"))]),e("mjx-container",E,[(p(),o("svg",b,s[26]||(s[26]=[l("",1)]))),s[27]||(s[27]=e("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[e("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[e("mrow",{"data-mjx-texclass":"ORD"},[e("mover",null,[e("mi",null,"e"),e("mo",{stretchy:"false"},"^")])]),e("mo",{stretchy:"false"},"("),e("mi",null,"κ"),e("mo",{stretchy:"false"},")"),e("mo",null,"="),e("msub",null,[e("mo",{"data-mjx-texclass":"OP"},"∫"),e("mrow",{"data-mjx-texclass":"ORD"},[e("mi",null,"κ"),e("mo",null,"≤"),e("mo",{"data-mjx-texclass":"ORD"},"∥"),e("mi",null,"k"),e("msub",null,[e("mo",{"data-mjx-texclass":"ORD"},"∥"),e("mn",null,"2")]),e("mo",null,"<"),e("mi",null,"κ"),e("mo",null,"+"),e("mn",null,"1")])]),e("mo",{"data-mjx-texclass":"ORD",stretchy:"false"},"|"),e("mrow",{"data-mjx-texclass":"ORD"},[e("mover",null,[e("mi",null,"e"),e("mo",{stretchy:"false"},"^")])]),e("mo",{stretchy:"false"},"("),e("mi",null,"k"),e("mo",{stretchy:"false"},")"),e("mo",{"data-mjx-texclass":"ORD",stretchy:"false"},"|"),e("mrow",{"data-mjx-texclass":"ORD"},[e("mi",{mathvariant:"normal"},"d")]),e("mi",null,"k"),e("mo",null,",")])],-1))]),s[29]||(s[29]=l("",3)),a(t,{type:"info",class:"source-link",text:"source"},{default:n(()=>s[28]||(s[28]=[e("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fd1f8e19740acedf01928865d15a70ab4a3a4686/src/processors.jl#L413-L428",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),e("details",f,[e("summary",null,[s[30]||(s[30]=e("a",{id:"IncompressibleNavierStokes.fieldplot",href:"#IncompressibleNavierStokes.fieldplot"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.fieldplot")],-1)),s[31]||(s[31]=i()),a(t,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),s[33]||(s[33]=l("",8)),a(t,{type:"info",class:"source-link",text:"source"},{default:n(()=>s[32]||(s[32]=[e("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fd1f8e19740acedf01928865d15a70ab4a3a4686/src/processors.jl#L384-L407",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),e("details",v,[e("summary",null,[s[34]||(s[34]=e("a",{id:"IncompressibleNavierStokes.fieldsaver-Tuple{}",href:"#IncompressibleNavierStokes.fieldsaver-Tuple{}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.fieldsaver")],-1)),s[35]||(s[35]=i()),a(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[37]||(s[37]=l("",2)),a(t,{type:"info",class:"source-link",text:"source"},{default:n(()=>s[36]||(s[36]=[e("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fd1f8e19740acedf01928865d15a70ab4a3a4686/src/processors.jl#L291",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),e("details",_,[e("summary",null,[s[38]||(s[38]=e("a",{id:"IncompressibleNavierStokes.observefield-Tuple{Any}",href:"#IncompressibleNavierStokes.observefield-Tuple{Any}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.observefield")],-1)),s[39]||(s[39]=i()),a(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[41]||(s[41]=l("",2)),a(t,{type:"info",class:"source-link",text:"source"},{default:n(()=>s[40]||(s[40]=[e("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fd1f8e19740acedf01928865d15a70ab4a3a4686/src/processors.jl#L74",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),e("details",j,[e("summary",null,[s[42]||(s[42]=e("a",{id:"IncompressibleNavierStokes.observespectrum-Tuple{Any}",href:"#IncompressibleNavierStokes.observespectrum-Tuple{Any}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.observespectrum")],-1)),s[43]||(s[43]=i()),a(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[45]||(s[45]=l("",2)),a(t,{type:"info",class:"source-link",text:"source"},{default:n(()=>s[44]||(s[44]=[e("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fd1f8e19740acedf01928865d15a70ab4a3a4686/src/processors.jl#L306",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),e("details",C,[e("summary",null,[s[46]||(s[46]=e("a",{id:"IncompressibleNavierStokes.processor",href:"#IncompressibleNavierStokes.processor"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.processor")],-1)),s[47]||(s[47]=i()),a(t,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),s[49]||(s[49]=l("",7)),a(t,{type:"info",class:"source-link",text:"source"},{default:n(()=>s[48]||(s[48]=[e("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fd1f8e19740acedf01928865d15a70ab4a3a4686/src/processors.jl#L1",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),e("details",S,[e("summary",null,[s[50]||(s[50]=e("a",{id:"IncompressibleNavierStokes.realtimeplotter",href:"#IncompressibleNavierStokes.realtimeplotter"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.realtimeplotter")],-1)),s[51]||(s[51]=i()),a(t,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),s[53]||(s[53]=l("",4)),a(t,{type:"info",class:"source-link",text:"source"},{default:n(()=>s[52]||(s[52]=[e("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fd1f8e19740acedf01928865d15a70ab4a3a4686/src/processors.jl#L365-L381",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),e("details",x,[e("summary",null,[s[54]||(s[54]=e("a",{id:"IncompressibleNavierStokes.save_vtk-Tuple{Any}",href:"#IncompressibleNavierStokes.save_vtk-Tuple{Any}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.save_vtk")],-1)),s[55]||(s[55]=i()),a(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[57]||(s[57]=l("",3)),a(t,{type:"info",class:"source-link",text:"source"},{default:n(()=>s[56]||(s[56]=[e("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fd1f8e19740acedf01928865d15a70ab4a3a4686/src/processors.jl#L252",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),e("details",F,[e("summary",null,[s[58]||(s[58]=e("a",{id:"IncompressibleNavierStokes.snapshotsaver-Tuple{Any}",href:"#IncompressibleNavierStokes.snapshotsaver-Tuple{Any}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.snapshotsaver")],-1)),s[59]||(s[59]=i()),a(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[61]||(s[61]=l("",2)),a(t,{type:"info",class:"source-link",text:"source"},{default:n(()=>s[60]||(s[60]=[e("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fd1f8e19740acedf01928865d15a70ab4a3a4686/src/processors.jl#L204",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),e("details",I,[e("summary",null,[s[62]||(s[62]=e("a",{id:"IncompressibleNavierStokes.timelogger-Tuple{}",href:"#IncompressibleNavierStokes.timelogger-Tuple{}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.timelogger")],-1)),s[63]||(s[63]=i()),a(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[65]||(s[65]=l("",2)),a(t,{type:"info",class:"source-link",text:"source"},{default:n(()=>s[64]||(s[64]=[e("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fd1f8e19740acedf01928865d15a70ab4a3a4686/src/processors.jl#L42",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),e("details",w,[e("summary",null,[s[66]||(s[66]=e("a",{id:"IncompressibleNavierStokes.vtk_writer-Tuple{}",href:"#IncompressibleNavierStokes.vtk_writer-Tuple{}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.vtk_writer")],-1)),s[67]||(s[67]=i()),a(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[69]||(s[69]=l("",2)),a(t,{type:"info",class:"source-link",text:"source"},{default:n(()=>s[68]||(s[68]=[e("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fd1f8e19740acedf01928865d15a70ab4a3a4686/src/processors.jl#L264",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})])])}const P=r(h,[["render",A]]);export{B as __pageData,P as default};
