import{_ as n,c as t,j as i,a,a5 as e,o as l}from"./chunks/framework.B4r91vhS.js";const m=JSON.parse('{"title":"Utils","description":"","frontmatter":{},"headers":[],"relativePath":"manual/utils.md","filePath":"manual/utils.md","lastUpdated":null}'),p={name:"manual/utils.md"},r={style:{"border-width":"1px","border-style":"solid","border-color":"black",padding:"1em","border-radius":"25px"}},d={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},o={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.489ex"},xmlns:"http://www.w3.org/2000/svg",width:"6.779ex",height:"1.995ex",role:"img",focusable:"false",viewBox:"0 -666 2996.4 882","aria-hidden":"true"};function h(k,s,g,u,E,b){return l(),t("div",null,[s[12]||(s[12]=i("h1",{id:"utils",tabindex:"-1"},[a("Utils "),i("a",{class:"header-anchor",href:"#utils","aria-label":'Permalink to "Utils"'},"​")],-1)),i("div",r,[s[10]||(s[10]=e(`<a id="IncompressibleNavierStokes.get_lims" href="#IncompressibleNavierStokes.get_lims">#</a> <b><u>IncompressibleNavierStokes.get_lims</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get_lims</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Tuple{Any, Any}</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get_lims</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, n) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Tuple{Any, Any}</span></span></code></pre></div>`,7)),i("p",null,[s[2]||(s[2]=a("Get approximate lower and upper limits of a field ")),s[3]||(s[3]=i("code",null,"x",-1)),s[4]||(s[4]=a(" based on the mean and standard deviation (")),i("mjx-container",d,[(l(),t("svg",o,s[0]||(s[0]=[e('<g stroke="currentColor" fill="currentColor" stroke-width="0" transform="scale(1,-1)"><g data-mml-node="math"><g data-mml-node="mi"><path data-c="1D707" d="M58 -216Q44 -216 34 -208T23 -186Q23 -176 96 116T173 414Q186 442 219 442Q231 441 239 435T249 423T251 413Q251 401 220 279T187 142Q185 131 185 107V99Q185 26 252 26Q261 26 270 27T287 31T302 38T315 45T327 55T338 65T348 77T356 88T365 100L372 110L408 253Q444 395 448 404Q461 431 491 431Q504 431 512 424T523 412T525 402L449 84Q448 79 448 68Q448 43 455 35T476 26Q485 27 496 35Q517 55 537 131Q543 151 547 152Q549 153 557 153H561Q580 153 580 144Q580 138 575 117T555 63T523 13Q510 0 491 -8Q483 -10 467 -10Q446 -10 429 -4T402 11T385 29T376 44T374 51L368 45Q362 39 350 30T324 12T288 -4T246 -11Q199 -11 153 12L129 -85Q108 -167 104 -180T92 -202Q76 -216 58 -216Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(825.2,0)"><path data-c="B1" d="M56 320T56 333T70 353H369V502Q369 651 371 655Q376 666 388 666Q402 666 405 654T409 596V500V353H707Q722 345 722 333Q722 320 707 313H409V40H707Q722 32 722 20T707 0H70Q56 7 56 20T70 40H369V313H70Q56 320 56 333Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(1825.4,0)"><path data-c="1D45B" d="M21 287Q22 293 24 303T36 341T56 388T89 425T135 442Q171 442 195 424T225 390T231 369Q231 367 232 367L243 378Q304 442 382 442Q436 442 469 415T503 336T465 179T427 52Q427 26 444 26Q450 26 453 27Q482 32 505 65T540 145Q542 153 560 153Q580 153 580 145Q580 144 576 130Q568 101 554 73T508 17T439 -10Q392 -10 371 17T350 73Q350 92 386 193T423 345Q423 404 379 404H374Q288 404 229 303L222 291L189 157Q156 26 151 16Q138 -11 108 -11Q95 -11 87 -5T76 7T74 17Q74 30 112 180T152 343Q153 348 153 366Q153 405 129 405Q91 405 66 305Q60 285 60 284Q58 278 41 278H27Q21 284 21 287Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(2425.4,0)"><path data-c="1D70E" d="M184 -11Q116 -11 74 34T31 147Q31 247 104 333T274 430Q275 431 414 431H552Q553 430 555 429T559 427T562 425T565 422T567 420T569 416T570 412T571 407T572 401Q572 357 507 357Q500 357 490 357T476 358H416L421 348Q439 310 439 263Q439 153 359 71T184 -11ZM361 278Q361 358 276 358Q152 358 115 184Q114 180 114 178Q106 141 106 117Q106 67 131 47T188 26Q242 26 287 73Q316 103 334 153T356 233T361 278Z" style="stroke-width:3;"></path></g></g></g>',1)]))),s[1]||(s[1]=i("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[i("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[i("mi",null,"μ"),i("mo",null,"±"),i("mi",null,"n"),i("mi",null,"σ")])],-1))]),s[5]||(s[5]=a("). If ")),s[6]||(s[6]=i("code",null,"x",-1)),s[7]||(s[7]=a(" is constant, a margin of ")),s[8]||(s[8]=i("code",null,"1e-4",-1)),s[9]||(s[9]=a(" is enforced. This is required for contour plotting functions that require a certain range."))]),s[11]||(s[11]=i("p",null,[i("a",{href:"https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/4a01a5afdfa96d7ae0f7b7303ca70af498f0cb9e/src/utils.jl#L1",target:"_blank",rel:"noreferrer"},"source")],-1))]),s[13]||(s[13]=e(`<br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="IncompressibleNavierStokes.get_spectrum-Tuple{Any}" href="#IncompressibleNavierStokes.get_spectrum-Tuple{Any}">#</a> <b><u>IncompressibleNavierStokes.get_spectrum</u></b> — <i>Method</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get_spectrum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    setup;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    npoint,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    a</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> NamedTuple{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:κ</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:masks</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:K</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Tuple{Any, Any, Any}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div><p>Get energy spectrum of velocity field.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/4a01a5afdfa96d7ae0f7b7303ca70af498f0cb9e/src/utils.jl#L120" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="IncompressibleNavierStokes.plotgrid" href="#IncompressibleNavierStokes.plotgrid">#</a> <b><u>IncompressibleNavierStokes.plotgrid</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">plotgrid</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, y; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">plotgrid</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, y, z)</span></span></code></pre></div><p>Plot nonuniform Cartesian grid.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/4a01a5afdfa96d7ae0f7b7303ca70af498f0cb9e/src/utils.jl#L14" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="IncompressibleNavierStokes.spectral_stuff-Tuple{Any}" href="#IncompressibleNavierStokes.spectral_stuff-Tuple{Any}">#</a> <b><u>IncompressibleNavierStokes.spectral_stuff</u></b> — <i>Method</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">spectral_stuff</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    setup;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    npoint,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    a</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> NamedTuple{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:A</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:κ</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:K</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Tuple{Any, Any, Any}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div><p>Get utilities to compute energy spectrum.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/4a01a5afdfa96d7ae0f7b7303ca70af498f0cb9e/src/utils.jl#L65" target="_blank" rel="noreferrer">source</a></p></div><br>`,7))])}const y=n(p,[["render",h]]);export{m as __pageData,y as default};