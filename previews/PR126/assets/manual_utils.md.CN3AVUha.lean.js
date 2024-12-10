import{_ as r,c as n,j as i,a as e,G as l,a5 as t,B as o,o as p}from"./chunks/framework.BSoZtefh.js";const F=JSON.parse('{"title":"Utils","description":"","frontmatter":{},"headers":[],"relativePath":"manual/utils.md","filePath":"manual/utils.md","lastUpdated":null}'),d={name:"manual/utils.md"},k={class:"jldocstring custom-block",open:""},h={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},g={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.489ex"},xmlns:"http://www.w3.org/2000/svg",width:"6.779ex",height:"1.995ex",role:"img",focusable:"false",viewBox:"0 -666 2996.4 882","aria-hidden":"true"},u={class:"jldocstring custom-block",open:""},b={class:"jldocstring custom-block",open:""},c={class:"jldocstring custom-block",open:""},m={class:"jldocstring custom-block",open:""},E={class:"jldocstring custom-block",open:""},y={class:"jldocstring custom-block",open:""};function T(f,s,v,Q,j,x){const a=o("Badge");return p(),n("div",null,[s[32]||(s[32]=i("h1",{id:"utils",tabindex:"-1"},[e("Utils "),i("a",{class:"header-anchor",href:"#utils","aria-label":'Permalink to "Utils"'},"​")],-1)),i("details",k,[i("summary",null,[s[0]||(s[0]=i("a",{id:"IncompressibleNavierStokes.get_lims",href:"#IncompressibleNavierStokes.get_lims"},[i("span",{class:"jlbinding"},"IncompressibleNavierStokes.get_lims")],-1)),s[1]||(s[1]=e()),l(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),s[12]||(s[12]=t(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get_lims</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Tuple{Any, Any}</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get_lims</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, n) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Tuple{Any, Any}</span></span></code></pre></div>`,1)),i("p",null,[s[4]||(s[4]=e("Get approximate lower and upper limits of a field ")),s[5]||(s[5]=i("code",null,"x",-1)),s[6]||(s[6]=e(" based on the mean and standard deviation (")),i("mjx-container",h,[(p(),n("svg",g,s[2]||(s[2]=[t('<g stroke="currentColor" fill="currentColor" stroke-width="0" transform="scale(1,-1)"><g data-mml-node="math"><g data-mml-node="mi"><path data-c="1D707" d="M58 -216Q44 -216 34 -208T23 -186Q23 -176 96 116T173 414Q186 442 219 442Q231 441 239 435T249 423T251 413Q251 401 220 279T187 142Q185 131 185 107V99Q185 26 252 26Q261 26 270 27T287 31T302 38T315 45T327 55T338 65T348 77T356 88T365 100L372 110L408 253Q444 395 448 404Q461 431 491 431Q504 431 512 424T523 412T525 402L449 84Q448 79 448 68Q448 43 455 35T476 26Q485 27 496 35Q517 55 537 131Q543 151 547 152Q549 153 557 153H561Q580 153 580 144Q580 138 575 117T555 63T523 13Q510 0 491 -8Q483 -10 467 -10Q446 -10 429 -4T402 11T385 29T376 44T374 51L368 45Q362 39 350 30T324 12T288 -4T246 -11Q199 -11 153 12L129 -85Q108 -167 104 -180T92 -202Q76 -216 58 -216Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(825.2,0)"><path data-c="B1" d="M56 320T56 333T70 353H369V502Q369 651 371 655Q376 666 388 666Q402 666 405 654T409 596V500V353H707Q722 345 722 333Q722 320 707 313H409V40H707Q722 32 722 20T707 0H70Q56 7 56 20T70 40H369V313H70Q56 320 56 333Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(1825.4,0)"><path data-c="1D45B" d="M21 287Q22 293 24 303T36 341T56 388T89 425T135 442Q171 442 195 424T225 390T231 369Q231 367 232 367L243 378Q304 442 382 442Q436 442 469 415T503 336T465 179T427 52Q427 26 444 26Q450 26 453 27Q482 32 505 65T540 145Q542 153 560 153Q580 153 580 145Q580 144 576 130Q568 101 554 73T508 17T439 -10Q392 -10 371 17T350 73Q350 92 386 193T423 345Q423 404 379 404H374Q288 404 229 303L222 291L189 157Q156 26 151 16Q138 -11 108 -11Q95 -11 87 -5T76 7T74 17Q74 30 112 180T152 343Q153 348 153 366Q153 405 129 405Q91 405 66 305Q60 285 60 284Q58 278 41 278H27Q21 284 21 287Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(2425.4,0)"><path data-c="1D70E" d="M184 -11Q116 -11 74 34T31 147Q31 247 104 333T274 430Q275 431 414 431H552Q553 430 555 429T559 427T562 425T565 422T567 420T569 416T570 412T571 407T572 401Q572 357 507 357Q500 357 490 357T476 358H416L421 348Q439 310 439 263Q439 153 359 71T184 -11ZM361 278Q361 358 276 358Q152 358 115 184Q114 180 114 178Q106 141 106 117Q106 67 131 47T188 26Q242 26 287 73Q316 103 334 153T356 233T361 278Z" style="stroke-width:3;"></path></g></g></g>',1)]))),s[3]||(s[3]=i("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[i("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[i("mi",null,"μ"),i("mo",null,"±"),i("mi",null,"n"),i("mi",null,"σ")])],-1))]),s[7]||(s[7]=e("). If ")),s[8]||(s[8]=i("code",null,"x",-1)),s[9]||(s[9]=e(" is constant, a margin of ")),s[10]||(s[10]=i("code",null,"1e-4",-1)),s[11]||(s[11]=e(" is enforced. This is required for contour plotting functions that require a certain range."))]),s[13]||(s[13]=i("p",null,[i("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/505e79d2b6bbbc243eed9b17029223fcc45d6236/src/utils.jl#L13",target:"_blank",rel:"noreferrer"},"source")],-1))]),i("details",u,[i("summary",null,[s[14]||(s[14]=i("a",{id:"IncompressibleNavierStokes.get_spectrum-Tuple{Any}",href:"#IncompressibleNavierStokes.get_spectrum-Tuple{Any}"},[i("span",{class:"jlbinding"},"IncompressibleNavierStokes.get_spectrum")],-1)),s[15]||(s[15]=e()),l(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[16]||(s[16]=t(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get_spectrum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    setup;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    npoint,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    a</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> NamedTuple{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:κ</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:masks</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:K</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Tuple{Any, Any, Any}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div><p>Get energy spectrum of velocity field.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/505e79d2b6bbbc243eed9b17029223fcc45d6236/src/utils.jl#L96" target="_blank" rel="noreferrer">source</a></p>`,3))]),i("details",b,[i("summary",null,[s[17]||(s[17]=i("a",{id:"IncompressibleNavierStokes.getoffset-Tuple{Any}",href:"#IncompressibleNavierStokes.getoffset-Tuple{Any}"},[i("span",{class:"jlbinding"},"IncompressibleNavierStokes.getoffset")],-1)),s[18]||(s[18]=e()),l(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[19]||(s[19]=t('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">getoffset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(I) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Any</span></span></code></pre></div><p>Get offset from <code>CartesianIndices</code> <code>I</code>.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/505e79d2b6bbbc243eed9b17029223fcc45d6236/src/utils.jl#L4" target="_blank" rel="noreferrer">source</a></p>',3))]),i("details",c,[i("summary",null,[s[20]||(s[20]=i("a",{id:"IncompressibleNavierStokes.getval-Union{Tuple{Val{x}}, Tuple{x}} where x",href:"#IncompressibleNavierStokes.getval-Union{Tuple{Val{x}}, Tuple{x}} where x"},[i("span",{class:"jlbinding"},"IncompressibleNavierStokes.getval")],-1)),s[21]||(s[21]=e()),l(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[22]||(s[22]=t('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">getval</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(_</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val{x}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Any</span></span></code></pre></div><p>Get value contained in <code>Val</code>.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/505e79d2b6bbbc243eed9b17029223fcc45d6236/src/utils.jl#L1" target="_blank" rel="noreferrer">source</a></p>',3))]),i("details",m,[i("summary",null,[s[23]||(s[23]=i("a",{id:"IncompressibleNavierStokes.plotgrid",href:"#IncompressibleNavierStokes.plotgrid"},[i("span",{class:"jlbinding"},"IncompressibleNavierStokes.plotgrid")],-1)),s[24]||(s[24]=e()),l(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),s[25]||(s[25]=t(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">plotgrid</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, y; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">plotgrid</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, y, z)</span></span></code></pre></div><p>Plot nonuniform Cartesian grid.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/505e79d2b6bbbc243eed9b17029223fcc45d6236/src/utils.jl#L26" target="_blank" rel="noreferrer">source</a></p>`,3))]),i("details",E,[i("summary",null,[s[26]||(s[26]=i("a",{id:"IncompressibleNavierStokes.spectral_stuff-Tuple{Any}",href:"#IncompressibleNavierStokes.spectral_stuff-Tuple{Any}"},[i("span",{class:"jlbinding"},"IncompressibleNavierStokes.spectral_stuff")],-1)),s[27]||(s[27]=e()),l(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[28]||(s[28]=t(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">spectral_stuff</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    setup;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    npoint,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    a</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> NamedTuple{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:inds</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:κ</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:K</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Tuple{Any, Any, Any}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div><p>Get utilities to compute energy spectrum.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/505e79d2b6bbbc243eed9b17029223fcc45d6236/src/utils.jl#L34" target="_blank" rel="noreferrer">source</a></p>`,3))]),i("details",y,[i("summary",null,[s[29]||(s[29]=i("a",{id:"IncompressibleNavierStokes.splitseed-Tuple{Any, Any}",href:"#IncompressibleNavierStokes.splitseed-Tuple{Any, Any}"},[i("span",{class:"jlbinding"},"IncompressibleNavierStokes.splitseed")],-1)),s[30]||(s[30]=e()),l(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[31]||(s[31]=t('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">splitseed</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(seed, n) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> AbstractArray</span></span></code></pre></div><p>Split random number generator seed into <code>n</code> new seeds.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/505e79d2b6bbbc243eed9b17029223fcc45d6236/src/utils.jl#L10" target="_blank" rel="noreferrer">source</a></p>',3))])])}const A=r(d,[["render",T]]);export{F as __pageData,A as default};
