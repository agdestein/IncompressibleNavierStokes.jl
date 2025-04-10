import{_ as o,C as d,c as p,o as r,ai as t,j as a,a as i,G as e,w as l}from"./chunks/framework.DzksKGI_.js";const V=JSON.parse('{"title":"Sparse matrices","description":"","frontmatter":{},"headers":[],"relativePath":"manual/matrices.md","filePath":"manual/matrices.md","lastUpdated":null}'),h={name:"manual/matrices.md"},k={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},c={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"4.986ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 2204 1000","aria-hidden":"true"},g={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},u={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"4.801ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 2122 1000","aria-hidden":"true"},m={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},E={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"7.53ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 3328.4 1000","aria-hidden":"true"},T={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},y={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"10.17ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 4495.2 1000","aria-hidden":"true"},b={class:"jldocstring custom-block",open:""},Q={class:"jldocstring custom-block",open:""},v={class:"jldocstring custom-block",open:""},f={class:"jldocstring custom-block",open:""},C={class:"jldocstring custom-block",open:""},x={class:"jldocstring custom-block",open:""},_={class:"jldocstring custom-block",open:""},F={class:"jldocstring custom-block",open:""},S={class:"jldocstring custom-block",open:""},j={class:"jldocstring custom-block",open:""};function A(w,s,I,D,B,N){const n=d("Badge");return r(),p("div",null,[s[60]||(s[60]=t("",14)),a("p",null,[s[4]||(s[4]=i("Note the sparsity pattern with matrix concatenation of two scalar operators for operators acting on or producing vector fields. The ")),s[5]||(s[5]=a("code",null,"pressuregradient_mat",-1)),s[6]||(s[6]=i(" converts a scalar field to a vector field, and is thus the vertical concatenation of the matrices for ")),a("mjx-container",k,[(r(),p("svg",c,s[0]||(s[0]=[t("",1)]))),s[1]||(s[1]=a("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[a("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[a("mi",null,"∂"),a("mrow",{"data-mjx-texclass":"ORD"},[a("mo",null,"/")]),a("mi",null,"∂"),a("mi",null,"x")])],-1))]),s[7]||(s[7]=i(" and ")),a("mjx-container",g,[(r(),p("svg",u,s[2]||(s[2]=[t("",1)]))),s[3]||(s[3]=a("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[a("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[a("mi",null,"∂"),a("mrow",{"data-mjx-texclass":"ORD"},[a("mo",null,"/")]),a("mi",null,"∂"),a("mi",null,"y")])],-1))]),s[8]||(s[8]=i(", while the ")),s[9]||(s[9]=a("code",null,"divergence_mat",-1)),s[10]||(s[10]=i(" is a horizontal concatenation of two similar matrices. The periodic boundary conditions are not included in the operators above, and are implemented via their own matrix. The periodic extension is visible:"))]),s[61]||(s[61]=t("",5)),a("p",null,[s[15]||(s[15]=i("Matrices only work on flattened fields ")),s[16]||(s[16]=a("code",null,"u[:]",-1)),s[17]||(s[17]=i(", while the kernels work on ")),a("mjx-container",m,[(r(),p("svg",E,s[11]||(s[11]=[t("",1)]))),s[12]||(s[12]=a("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[a("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[a("mo",{stretchy:"false"},"("),a("mi",null,"D"),a("mo",null,"+"),a("mn",null,"1"),a("mo",{stretchy:"false"},")")])],-1))]),s[18]||(s[18]=i("-array-shaped fields for a dimension ")),a("mjx-container",T,[(r(),p("svg",y,s[13]||(s[13]=[t("",1)]))),s[14]||(s[14]=a("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[a("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[a("mi",null,"D"),a("mo",null,"∈"),a("mo",{fence:"false",stretchy:"false"},"{"),a("mn",null,"2"),a("mo",null,","),a("mn",null,"3"),a("mo",{fence:"false",stretchy:"false"},"}")])],-1))]),s[19]||(s[19]=i("."))]),s[62]||(s[62]=t("",14)),a("details",b,[a("summary",null,[s[20]||(s[20]=a("a",{id:"IncompressibleNavierStokes.bc_p_mat",href:"#IncompressibleNavierStokes.bc_p_mat"},[a("span",{class:"jlbinding"},"IncompressibleNavierStokes.bc_p_mat")],-1)),s[21]||(s[21]=i()),e(n,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),s[23]||(s[23]=a("p",null,[i("Matrix for applying boundary conditions to pressure fields "),a("code",null,"p"),i(".")],-1)),e(n,{type:"info",class:"source-link",text:"source"},{default:l(()=>s[22]||(s[22]=[a("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fb876fe1b887251a590eaa43ee28dadedd57fd67/src/matrices.jl#L61",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),a("details",Q,[a("summary",null,[s[24]||(s[24]=a("a",{id:"IncompressibleNavierStokes.bc_temp_mat",href:"#IncompressibleNavierStokes.bc_temp_mat"},[a("span",{class:"jlbinding"},"IncompressibleNavierStokes.bc_temp_mat")],-1)),s[25]||(s[25]=i()),e(n,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),s[27]||(s[27]=a("p",null,[i("Matrix for applying boundary conditions to temperature fields "),a("code",null,"temp"),i(".")],-1)),e(n,{type:"info",class:"source-link",text:"source"},{default:l(()=>s[26]||(s[26]=[a("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fb876fe1b887251a590eaa43ee28dadedd57fd67/src/matrices.jl#L64",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),a("details",v,[a("summary",null,[s[28]||(s[28]=a("a",{id:"IncompressibleNavierStokes.bc_u_mat",href:"#IncompressibleNavierStokes.bc_u_mat"},[a("span",{class:"jlbinding"},"IncompressibleNavierStokes.bc_u_mat")],-1)),s[29]||(s[29]=i()),e(n,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),s[31]||(s[31]=a("p",null,[i("Create matrix for applying boundary conditions to velocity fields "),a("code",null,"u"),i(". This matrix only applies the boundary conditions depending on "),a("code",null,"u"),i(" itself (e.g. "),a("a",{href:"/IncompressibleNavierStokes.jl/dev/manual/setup#IncompressibleNavierStokes.PeriodicBC"},[a("code",null,"PeriodicBC")]),i("). It does not apply constant boundary conditions (e.g. non-zero "),a("a",{href:"/IncompressibleNavierStokes.jl/dev/manual/setup#IncompressibleNavierStokes.DirichletBC"},[a("code",null,"DirichletBC")]),i(").")],-1)),e(n,{type:"info",class:"source-link",text:"source"},{default:l(()=>s[30]||(s[30]=[a("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fb876fe1b887251a590eaa43ee28dadedd57fd67/src/matrices.jl#L54",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),a("details",f,[a("summary",null,[s[32]||(s[32]=a("a",{id:"IncompressibleNavierStokes.diffusion_mat-Tuple{Any}",href:"#IncompressibleNavierStokes.diffusion_mat-Tuple{Any}"},[a("span",{class:"jlbinding"},"IncompressibleNavierStokes.diffusion_mat")],-1)),s[33]||(s[33]=i()),e(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[35]||(s[35]=t("",2)),e(n,{type:"info",class:"source-link",text:"source"},{default:l(()=>s[34]||(s[34]=[a("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fb876fe1b887251a590eaa43ee28dadedd57fd67/src/matrices.jl#L494",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),a("details",C,[a("summary",null,[s[36]||(s[36]=a("a",{id:"IncompressibleNavierStokes.divergence_mat-Tuple{Any}",href:"#IncompressibleNavierStokes.divergence_mat-Tuple{Any}"},[a("span",{class:"jlbinding"},"IncompressibleNavierStokes.divergence_mat")],-1)),s[37]||(s[37]=i()),e(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[39]||(s[39]=t("",2)),e(n,{type:"info",class:"source-link",text:"source"},{default:l(()=>s[38]||(s[38]=[a("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fb876fe1b887251a590eaa43ee28dadedd57fd67/src/matrices.jl#L388",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),a("details",x,[a("summary",null,[s[40]||(s[40]=a("a",{id:"IncompressibleNavierStokes.laplacian_mat-Tuple{Any}",href:"#IncompressibleNavierStokes.laplacian_mat-Tuple{Any}"},[a("span",{class:"jlbinding"},"IncompressibleNavierStokes.laplacian_mat")],-1)),s[41]||(s[41]=i()),e(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[43]||(s[43]=t("",2)),e(n,{type:"info",class:"source-link",text:"source"},{default:l(()=>s[42]||(s[42]=[a("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fb876fe1b887251a590eaa43ee28dadedd57fd67/src/matrices.jl#L480",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),a("details",_,[a("summary",null,[s[44]||(s[44]=a("a",{id:"IncompressibleNavierStokes.pad_scalarfield_mat-Tuple{Any}",href:"#IncompressibleNavierStokes.pad_scalarfield_mat-Tuple{Any}"},[a("span",{class:"jlbinding"},"IncompressibleNavierStokes.pad_scalarfield_mat")],-1)),s[45]||(s[45]=i()),e(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[47]||(s[47]=t("",3)),e(n,{type:"info",class:"source-link",text:"source"},{default:l(()=>s[46]||(s[46]=[a("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fb876fe1b887251a590eaa43ee28dadedd57fd67/src/matrices.jl#L16",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),a("details",F,[a("summary",null,[s[48]||(s[48]=a("a",{id:"IncompressibleNavierStokes.pad_vectorfield_mat-Tuple{Any}",href:"#IncompressibleNavierStokes.pad_vectorfield_mat-Tuple{Any}"},[a("span",{class:"jlbinding"},"IncompressibleNavierStokes.pad_vectorfield_mat")],-1)),s[49]||(s[49]=i()),e(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[51]||(s[51]=t("",2)),e(n,{type:"info",class:"source-link",text:"source"},{default:l(()=>s[50]||(s[50]=[a("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fb876fe1b887251a590eaa43ee28dadedd57fd67/src/matrices.jl#L34",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),a("details",S,[a("summary",null,[s[52]||(s[52]=a("a",{id:"IncompressibleNavierStokes.pressuregradient_mat-Tuple{Any}",href:"#IncompressibleNavierStokes.pressuregradient_mat-Tuple{Any}"},[a("span",{class:"jlbinding"},"IncompressibleNavierStokes.pressuregradient_mat")],-1)),s[53]||(s[53]=i()),e(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[55]||(s[55]=t("",2)),e(n,{type:"info",class:"source-link",text:"source"},{default:l(()=>s[54]||(s[54]=[a("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fb876fe1b887251a590eaa43ee28dadedd57fd67/src/matrices.jl#L429",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),a("details",j,[a("summary",null,[s[56]||(s[56]=a("a",{id:"IncompressibleNavierStokes.volume_mat-Tuple{Any}",href:"#IncompressibleNavierStokes.volume_mat-Tuple{Any}"},[a("span",{class:"jlbinding"},"IncompressibleNavierStokes.volume_mat")],-1)),s[57]||(s[57]=i()),e(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[59]||(s[59]=t("",2)),e(n,{type:"info",class:"source-link",text:"source"},{default:l(()=>s[58]||(s[58]=[a("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/fb876fe1b887251a590eaa43ee28dadedd57fd67/src/matrices.jl#L470",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})])])}const L=o(h,[["render",A]]);export{V as __pageData,L as default};
