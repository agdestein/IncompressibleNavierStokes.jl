import{_ as r,c as n,j as e,a as i,a5 as t,G as l,B as o,o as p}from"./chunks/framework.Cs5qBNw-.js";const I=JSON.parse('{"title":"Pressure solvers","description":"","frontmatter":{},"headers":[],"relativePath":"manual/pressure.md","filePath":"manual/pressure.md","lastUpdated":null}'),d={name:"manual/pressure.md"},h={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},k={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"15.194ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 6715.6 1000","aria-hidden":"true"},c={class:"jldocstring custom-block",open:""},u={class:"jldocstring custom-block",open:""},g={class:"jldocstring custom-block",open:""},b={class:"jldocstring custom-block",open:""},Q={class:"jldocstring custom-block",open:""},m={class:"jldocstring custom-block",open:""},T={class:"jldocstring custom-block",open:""},y={class:"jldocstring custom-block",open:""},v={class:"jldocstring custom-block",open:""},E={class:"jldocstring custom-block",open:""},f={class:"jldocstring custom-block",open:""};function j(C,s,N,F,S,x){const a=o("Badge");return p(),n("div",null,[s[35]||(s[35]=e("h1",{id:"Pressure-solvers",tabindex:"-1"},[i("Pressure solvers "),e("a",{class:"header-anchor",href:"#Pressure-solvers","aria-label":'Permalink to "Pressure solvers {#Pressure-solvers}"'},"​")],-1)),s[36]||(s[36]=e("p",null,"The discrete pressure Poisson equation",-1)),e("mjx-container",h,[(p(),n("svg",k,s[0]||(s[0]=[t('<g stroke="currentColor" fill="currentColor" stroke-width="0" transform="scale(1,-1)"><g data-mml-node="math"><g data-mml-node="mi"><path data-c="1D43F" d="M228 637Q194 637 192 641Q191 643 191 649Q191 673 202 682Q204 683 217 683Q271 680 344 680Q485 680 506 683H518Q524 677 524 674T522 656Q517 641 513 637H475Q406 636 394 628Q387 624 380 600T313 336Q297 271 279 198T252 88L243 52Q243 48 252 48T311 46H328Q360 46 379 47T428 54T478 72T522 106T564 161Q580 191 594 228T611 270Q616 273 628 273H641Q647 264 647 262T627 203T583 83T557 9Q555 4 553 3T537 0T494 -1Q483 -1 418 -1T294 0H116Q32 0 32 10Q32 17 34 24Q39 43 44 45Q48 46 59 46H65Q92 46 125 49Q139 52 144 61Q147 65 216 339T285 628Q285 635 228 637Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(681,0)"><path data-c="1D45D" d="M23 287Q24 290 25 295T30 317T40 348T55 381T75 411T101 433T134 442Q209 442 230 378L240 387Q302 442 358 442Q423 442 460 395T497 281Q497 173 421 82T249 -10Q227 -10 210 -4Q199 1 187 11T168 28L161 36Q160 35 139 -51T118 -138Q118 -144 126 -145T163 -148H188Q194 -155 194 -157T191 -175Q188 -187 185 -190T172 -194Q170 -194 161 -194T127 -193T65 -192Q-5 -192 -24 -194H-32Q-39 -187 -39 -183Q-37 -156 -26 -148H-6Q28 -147 33 -136Q36 -130 94 103T155 350Q156 355 156 364Q156 405 131 405Q109 405 94 377T71 316T59 280Q57 278 43 278H29Q23 284 23 287ZM178 102Q200 26 252 26Q282 26 310 49T356 107Q374 141 392 215T411 325V331Q411 405 350 405Q339 405 328 402T306 393T286 380T269 365T254 350T243 336T235 326L232 322Q232 321 229 308T218 264T204 212Q178 106 178 102Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(1461.8,0)"><path data-c="3D" d="M56 347Q56 360 70 367H707Q722 359 722 347Q722 336 708 328L390 327H72Q56 332 56 347ZM56 153Q56 168 72 173H708Q722 163 722 153Q722 140 707 133H70Q56 140 56 153Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(2517.6,0)"><path data-c="1D44A" d="M436 683Q450 683 486 682T553 680Q604 680 638 681T677 682Q695 682 695 674Q695 670 692 659Q687 641 683 639T661 637Q636 636 621 632T600 624T597 615Q597 603 613 377T629 138L631 141Q633 144 637 151T649 170T666 200T690 241T720 295T759 362Q863 546 877 572T892 604Q892 619 873 628T831 637Q817 637 817 647Q817 650 819 660Q823 676 825 679T839 682Q842 682 856 682T895 682T949 681Q1015 681 1034 683Q1048 683 1048 672Q1048 666 1045 655T1038 640T1028 637Q1006 637 988 631T958 617T939 600T927 584L923 578L754 282Q586 -14 585 -15Q579 -22 561 -22Q546 -22 542 -17Q539 -14 523 229T506 480L494 462Q472 425 366 239Q222 -13 220 -15T215 -19Q210 -22 197 -22Q178 -22 176 -15Q176 -12 154 304T131 622Q129 631 121 633T82 637H58Q51 644 51 648Q52 671 64 683H76Q118 680 176 680Q301 680 313 683H323Q329 677 329 674T327 656Q322 641 318 637H297Q236 634 232 620Q262 160 266 136L501 550L499 587Q496 629 489 632Q483 636 447 637Q428 637 422 639T416 648Q416 650 418 660Q419 664 420 669T421 676T424 680T428 682T436 683Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(3565.6,0)"><path data-c="1D440" d="M289 629Q289 635 232 637Q208 637 201 638T194 648Q194 649 196 659Q197 662 198 666T199 671T201 676T203 679T207 681T212 683T220 683T232 684Q238 684 262 684T307 683Q386 683 398 683T414 678Q415 674 451 396L487 117L510 154Q534 190 574 254T662 394Q837 673 839 675Q840 676 842 678T846 681L852 683H948Q965 683 988 683T1017 684Q1051 684 1051 673Q1051 668 1048 656T1045 643Q1041 637 1008 637Q968 636 957 634T939 623Q936 618 867 340T797 59Q797 55 798 54T805 50T822 48T855 46H886Q892 37 892 35Q892 19 885 5Q880 0 869 0Q864 0 828 1T736 2Q675 2 644 2T609 1Q592 1 592 11Q592 13 594 25Q598 41 602 43T625 46Q652 46 685 49Q699 52 704 61Q706 65 742 207T813 490T848 631L654 322Q458 10 453 5Q451 4 449 3Q444 0 433 0Q418 0 415 7Q413 11 374 317L335 624L267 354Q200 88 200 79Q206 46 272 46H282Q288 41 289 37T286 19Q282 3 278 1Q274 0 267 0Q265 0 255 0T221 1T157 2Q127 2 95 1T58 0Q43 0 39 2T35 11Q35 13 38 25T43 40Q45 46 65 46Q135 46 154 86Q158 92 223 354T289 629Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(4616.6,0)"><path data-c="1D439" d="M48 1Q31 1 31 11Q31 13 34 25Q38 41 42 43T65 46Q92 46 125 49Q139 52 144 61Q146 66 215 342T285 622Q285 629 281 629Q273 632 228 634H197Q191 640 191 642T193 659Q197 676 203 680H742Q749 676 749 669Q749 664 736 557T722 447Q720 440 702 440H690Q683 445 683 453Q683 454 686 477T689 530Q689 560 682 579T663 610T626 626T575 633T503 634H480Q398 633 393 631Q388 629 386 623Q385 622 352 492L320 363H375Q378 363 398 363T426 364T448 367T472 374T489 386Q502 398 511 419T524 457T529 475Q532 480 548 480H560Q567 475 567 470Q567 467 536 339T502 207Q500 200 482 200H470Q463 206 463 212Q463 215 468 234T473 274Q473 303 453 310T364 317H309L277 190Q245 66 245 60Q245 46 334 46H359Q365 40 365 39T363 19Q359 6 353 0H336Q295 2 185 2Q120 2 86 2T48 1Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(5365.6,0)"><path data-c="28" d="M94 250Q94 319 104 381T127 488T164 576T202 643T244 695T277 729T302 750H315H319Q333 750 333 741Q333 738 316 720T275 667T226 581T184 443T167 250T184 58T225 -81T274 -167T316 -220T333 -241Q333 -250 318 -250H315H302L274 -226Q180 -141 137 -14T94 250Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(5754.6,0)"><path data-c="1D462" d="M21 287Q21 295 30 318T55 370T99 420T158 442Q204 442 227 417T250 358Q250 340 216 246T182 105Q182 62 196 45T238 27T291 44T328 78L339 95Q341 99 377 247Q407 367 413 387T427 416Q444 431 463 431Q480 431 488 421T496 402L420 84Q419 79 419 68Q419 43 426 35T447 26Q469 29 482 57T512 145Q514 153 532 153Q551 153 551 144Q550 139 549 130T540 98T523 55T498 17T462 -8Q454 -10 438 -10Q372 -10 347 46Q345 45 336 36T318 21T296 6T267 -6T233 -11Q189 -11 155 7Q103 38 103 113Q103 170 138 262T173 379Q173 380 173 381Q173 390 173 393T169 400T158 404H154Q131 404 112 385T82 344T65 302T57 280Q55 278 41 278H27Q21 284 21 287Z" style="stroke-width:3;"></path></g><g data-mml-node="mo" transform="translate(6326.6,0)"><path data-c="29" d="M60 749L64 750Q69 750 74 750H86L114 726Q208 641 251 514T294 250Q294 182 284 119T261 12T224 -76T186 -143T145 -194T113 -227T90 -246Q87 -249 86 -250H74Q66 -250 63 -250T58 -247T55 -238Q56 -237 66 -225Q221 -64 221 250T66 725Q56 737 55 738Q55 746 60 749Z" style="stroke-width:3;"></path></g></g></g>',1)]))),s[1]||(s[1]=e("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[e("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[e("mi",null,"L"),e("mi",null,"p"),e("mo",null,"="),e("mi",null,"W"),e("mi",null,"M"),e("mi",null,"F"),e("mo",{stretchy:"false"},"("),e("mi",null,"u"),e("mo",{stretchy:"false"},")")])],-1))]),s[37]||(s[37]=e("p",null,"enforces divergence freeness. There are multiple options for solving this system.",-1)),e("details",c,[e("summary",null,[s[2]||(s[2]=e("a",{id:"IncompressibleNavierStokes.default_psolver-Tuple{Any}",href:"#IncompressibleNavierStokes.default_psolver-Tuple{Any}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.default_psolver")],-1)),s[3]||(s[3]=i()),l(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[4]||(s[4]=t(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_psolver</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    setup</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Union{IncompressibleNavierStokes</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">var&quot;#psolve!#101&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">{Bool}, IncompressibleNavierStokes</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">var&quot;#psolve!#124&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div><p>Get default Poisson solver from setup.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/c25f63c9eeb0d3cc8f6db8c8112822c173f74c51/src/pressure.jl#L84" target="_blank" rel="noreferrer">source</a></p>`,3))]),e("details",u,[e("summary",null,[s[5]||(s[5]=e("a",{id:"IncompressibleNavierStokes.poisson!-Tuple{Any, Any}",href:"#IncompressibleNavierStokes.poisson!-Tuple{Any, Any}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.poisson!")],-1)),s[6]||(s[6]=i()),l(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[7]||(s[7]=t('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">poisson!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(psolver, f) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Any</span></span></code></pre></div><p>Solve the Poisson equation for the pressure (in-place version).</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/c25f63c9eeb0d3cc8f6db8c8112822c173f74c51/src/pressure.jl#L21" target="_blank" rel="noreferrer">source</a></p>',3))]),e("details",g,[e("summary",null,[s[8]||(s[8]=e("a",{id:"IncompressibleNavierStokes.poisson-Tuple{Any, Any}",href:"#IncompressibleNavierStokes.poisson-Tuple{Any, Any}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.poisson")],-1)),s[9]||(s[9]=i()),l(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[10]||(s[10]=t('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">poisson</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(psolver, f) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Any</span></span></code></pre></div><p>Solve the Poisson equation for the pressure with right hand side <code>f</code> at time <code>t</code>. For periodic and no-slip BC, the sum of <code>f</code> should be zero.</p><p>Differentiable version.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/c25f63c9eeb0d3cc8f6db8c8112822c173f74c51/src/pressure.jl#L9" target="_blank" rel="noreferrer">source</a></p>',4))]),e("details",b,[e("summary",null,[s[11]||(s[11]=e("a",{id:"IncompressibleNavierStokes.pressure!-NTuple{5, Any}",href:"#IncompressibleNavierStokes.pressure!-NTuple{5, Any}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.pressure!")],-1)),s[12]||(s[12]=i()),l(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[13]||(s[13]=t('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">pressure!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(p, u, temp, t, setup; psolver, F)</span></span></code></pre></div><p>Compute pressure from velocity field (in-place version).</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/c25f63c9eeb0d3cc8f6db8c8112822c173f74c51/src/pressure.jl#L40" target="_blank" rel="noreferrer">source</a></p>',3))]),e("details",Q,[e("summary",null,[s[14]||(s[14]=e("a",{id:"IncompressibleNavierStokes.pressure-NTuple{4, Any}",href:"#IncompressibleNavierStokes.pressure-NTuple{4, Any}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.pressure")],-1)),s[15]||(s[15]=i()),l(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[16]||(s[16]=t('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">pressure</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(u, temp, t, setup; psolver)</span></span></code></pre></div><p>Compute pressure from velocity field. This makes the pressure compatible with the velocity field, resulting in same order pressure as velocity.</p><p>Differentiable version.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/c25f63c9eeb0d3cc8f6db8c8112822c173f74c51/src/pressure.jl#L24" target="_blank" rel="noreferrer">source</a></p>',4))]),e("details",m,[e("summary",null,[s[17]||(s[17]=e("a",{id:"IncompressibleNavierStokes.project!-Tuple{Any, Any}",href:"#IncompressibleNavierStokes.project!-Tuple{Any, Any}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.project!")],-1)),s[18]||(s[18]=i()),l(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[19]||(s[19]=t('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">project!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(u, setup; psolver, p)</span></span></code></pre></div><p>Project velocity field onto divergence-free space (in-place version).</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/c25f63c9eeb0d3cc8f6db8c8112822c173f74c51/src/pressure.jl#L68" target="_blank" rel="noreferrer">source</a></p>',3))]),e("details",T,[e("summary",null,[s[20]||(s[20]=e("a",{id:"IncompressibleNavierStokes.project-Tuple{Any, Any}",href:"#IncompressibleNavierStokes.project-Tuple{Any, Any}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.project")],-1)),s[21]||(s[21]=i()),l(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[22]||(s[22]=t('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">project</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(u, setup; psolver)</span></span></code></pre></div><p>Project velocity field onto divergence-free space (differentiable version).</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/c25f63c9eeb0d3cc8f6db8c8112822c173f74c51/src/pressure.jl#L51" target="_blank" rel="noreferrer">source</a></p>',3))]),e("details",y,[e("summary",null,[s[23]||(s[23]=e("a",{id:"IncompressibleNavierStokes.psolver_cg-Tuple{Any}",href:"#IncompressibleNavierStokes.psolver_cg-Tuple{Any}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.psolver_cg")],-1)),s[24]||(s[24]=i()),l(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[25]||(s[25]=t(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">psolver_cg</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    setup;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    abstol,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    reltol,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    maxiter,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    preconditioner</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> IncompressibleNavierStokes</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">var&quot;#psolve!#111&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">{_A, _B, _C, IncompressibleNavierStokes</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">var&quot;#laplace_diag#109&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">{ndrange, workgroupsize}} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {_A, _B, _C, ndrange, workgroupsize}</span></span></code></pre></div><p>Conjugate gradients iterative Poisson solver.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/c25f63c9eeb0d3cc8f6db8c8112822c173f74c51/src/pressure.jl#L208" target="_blank" rel="noreferrer">source</a></p>`,3))]),e("details",v,[e("summary",null,[s[26]||(s[26]=e("a",{id:"IncompressibleNavierStokes.psolver_cg_matrix-Tuple{Any}",href:"#IncompressibleNavierStokes.psolver_cg_matrix-Tuple{Any}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.psolver_cg_matrix")],-1)),s[27]||(s[27]=i()),l(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[28]||(s[28]=t(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">psolver_cg_matrix</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    setup;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> IncompressibleNavierStokes</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">var&quot;#psolve!#105&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">{Base</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Pairs{Symbol, Union{}, Tuple{}, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">{}}}</span></span></code></pre></div><p>Conjugate gradients iterative Poisson solver. The <code>kwargs</code> are passed to the <code>cg!</code> function from IterativeSolvers.jl.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/c25f63c9eeb0d3cc8f6db8c8112822c173f74c51/src/pressure.jl#L156" target="_blank" rel="noreferrer">source</a></p>`,3))]),e("details",E,[e("summary",null,[s[29]||(s[29]=e("a",{id:"IncompressibleNavierStokes.psolver_direct-Tuple{Any}",href:"#IncompressibleNavierStokes.psolver_direct-Tuple{Any}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.psolver_direct")],-1)),s[30]||(s[30]=i()),l(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[31]||(s[31]=t(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">psolver_direct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    setup</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> IncompressibleNavierStokes</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">var&quot;#psolve!#101&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">{Bool}</span></span></code></pre></div><p>Create direct Poisson solver using an appropriate matrix decomposition.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/c25f63c9eeb0d3cc8f6db8c8112822c173f74c51/src/pressure.jl#L100" target="_blank" rel="noreferrer">source</a></p>`,3))]),e("details",f,[e("summary",null,[s[32]||(s[32]=e("a",{id:"IncompressibleNavierStokes.psolver_spectral-Tuple{Any}",href:"#IncompressibleNavierStokes.psolver_spectral-Tuple{Any}"},[e("span",{class:"jlbinding"},"IncompressibleNavierStokes.psolver_spectral")],-1)),s[33]||(s[33]=i()),l(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[34]||(s[34]=t(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">psolver_spectral</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    setup</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> IncompressibleNavierStokes</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">var&quot;#psolve!#124&quot;</span></span></code></pre></div><p>Create spectral Poisson solver from setup.</p><p><a href="https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/c25f63c9eeb0d3cc8f6db8c8112822c173f74c51/src/pressure.jl#L288" target="_blank" rel="noreferrer">source</a></p>`,3))])])}const H=r(d,[["render",j]]);export{I as __pageData,H as default};
