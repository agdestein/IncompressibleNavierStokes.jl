import{_ as n,c as a,a5 as p,o as l}from"./chunks/framework.C2KJk1yw.js";const b=JSON.parse('{"title":"Package versions","description":"","frontmatter":{},"headers":[],"relativePath":"about/versions.md","filePath":"about/versions.md","lastUpdated":null}'),e={name:"about/versions.md"};function i(c,s,v,t,r,d){return l(),a("div",null,s[0]||(s[0]=[p(`<h1 id="Package-versions" tabindex="-1">Package versions <a class="header-anchor" href="#Package-versions" aria-label="Permalink to &quot;Package versions {#Package-versions}&quot;">​</a></h1><h2 id="Julia-version" tabindex="-1">Julia version <a class="header-anchor" href="#Julia-version" aria-label="Permalink to &quot;Julia version {#Julia-version}&quot;">​</a></h2><p>The examples were generated with the following Julia version:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.11.1</span></span>
<span class="line"><span>Commit 8f5b7ca12ad (2024-10-16 10:53 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 4 × AMD EPYC 7763 64-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LLVM: libLLVM-16.0.6 (ORCJIT, znver3)</span></span>
<span class="line"><span>Threads: 1 default, 0 interactive, 1 GC (on 4 virtual cores)</span></span></code></pre></div><h2 id="Package-versions-2" tabindex="-1">Package versions <a class="header-anchor" href="#Package-versions-2" aria-label="Permalink to &quot;Package versions {#Package-versions-2}&quot;">​</a></h2><p>The examples were generated with the following package versions:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Pkg</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Pkg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">status</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; mode </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Pkg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">PKGMODE_MANIFEST)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Status \`~/work/IncompressibleNavierStokes.jl/IncompressibleNavierStokes.jl/docs/Manifest.toml\`</span></span>
<span class="line"><span>  [47edcb42] ADTypes v1.10.0</span></span>
<span class="line"><span>  [a4c015fc] ANSIColoredPrinters v0.0.1</span></span>
<span class="line"><span>  [621f4979] AbstractFFTs v1.5.0</span></span>
<span class="line"><span>  [1520ce14] AbstractTrees v0.4.5</span></span>
<span class="line"><span>  [7d9f7c33] Accessors v0.1.38</span></span>
<span class="line"><span>  [79e6a3ab] Adapt v4.1.1</span></span>
<span class="line"><span>  [35492f91] AdaptivePredicates v1.2.0</span></span>
<span class="line"><span>  [66dad0bd] AliasTables v1.1.3</span></span>
<span class="line"><span>  [27a7e980] Animations v0.4.2</span></span>
<span class="line"><span>  [dce04be8] ArgCheck v2.3.0</span></span>
<span class="line"><span>  [4fba245c] ArrayInterface v7.17.1</span></span>
<span class="line"><span>  [a9b6321e] Atomix v0.1.0</span></span>
<span class="line"><span>  [67c07d97] Automa v1.1.0</span></span>
<span class="line"><span>  [13072b0f] AxisAlgorithms v1.1.0</span></span>
<span class="line"><span>  [39de3d68] AxisArrays v0.4.7</span></span>
<span class="line"><span>  [198e06fe] BangBang v0.4.3</span></span>
<span class="line"><span>  [9718e550] Baselet v0.1.1</span></span>
<span class="line"><span>  [2027ae74] BibInternal v0.3.7</span></span>
<span class="line"><span>  [13533e5b] BibParser v0.2.2</span></span>
<span class="line"><span>⌅ [f1be7e48] Bibliography v0.2.20</span></span>
<span class="line"><span>  [62783981] BitTwiddlingConvenienceFunctions v0.1.6</span></span>
<span class="line"><span>  [fa961155] CEnum v0.5.0</span></span>
<span class="line"><span>  [2a0fbf3d] CPUSummary v0.2.6</span></span>
<span class="line"><span>  [159f3aea] Cairo v1.1.1</span></span>
<span class="line"><span>  [13f3f980] CairoMakie v0.12.16</span></span>
<span class="line"><span>  [082447d4] ChainRules v1.72.1</span></span>
<span class="line"><span>  [d360d2e6] ChainRulesCore v1.25.0</span></span>
<span class="line"><span>  [fb6a15b2] CloseOpenIntervals v0.1.13</span></span>
<span class="line"><span>  [944b1d66] CodecZlib v0.7.6</span></span>
<span class="line"><span>  [a2cac450] ColorBrewer v0.4.0</span></span>
<span class="line"><span>  [35d6a980] ColorSchemes v3.27.1</span></span>
<span class="line"><span>⌅ [3da002f7] ColorTypes v0.11.5</span></span>
<span class="line"><span>⌃ [c3611d14] ColorVectorSpace v0.10.0</span></span>
<span class="line"><span>⌅ [5ae59095] Colors v0.12.11</span></span>
<span class="line"><span>  [bbf7d656] CommonSubexpressions v0.3.1</span></span>
<span class="line"><span>  [f70d9fcc] CommonWorldInvalidations v1.0.0</span></span>
<span class="line"><span>  [34da2185] Compat v4.16.0</span></span>
<span class="line"><span>  [b0b7db55] ComponentArrays v0.15.19</span></span>
<span class="line"><span>  [a33af91c] CompositionsBase v0.1.2</span></span>
<span class="line"><span>  [2569d6c7] ConcreteStructs v0.2.3</span></span>
<span class="line"><span>  [187b0558] ConstructionBase v1.5.8</span></span>
<span class="line"><span>  [6add18c4] ContextVariablesX v0.1.3</span></span>
<span class="line"><span>  [d38c429a] Contour v0.6.3</span></span>
<span class="line"><span>  [adafc99b] CpuId v0.3.1</span></span>
<span class="line"><span>  [9a962f9c] DataAPI v1.16.0</span></span>
<span class="line"><span>  [864edb3b] DataStructures v0.18.20</span></span>
<span class="line"><span>  [e2d170a0] DataValueInterfaces v1.0.0</span></span>
<span class="line"><span>  [244e2a9f] DefineSingletons v0.1.2</span></span>
<span class="line"><span>  [927a84f5] DelaunayTriangulation v1.6.1</span></span>
<span class="line"><span>  [8bb1440f] DelimitedFiles v1.9.1</span></span>
<span class="line"><span>  [163ba53b] DiffResults v1.1.0</span></span>
<span class="line"><span>  [b552c78f] DiffRules v1.15.1</span></span>
<span class="line"><span>  [8d63f2c5] DispatchDoctor v0.4.17</span></span>
<span class="line"><span>  [31c24e10] Distributions v0.25.113</span></span>
<span class="line"><span>  [ffbed154] DocStringExtensions v0.9.3</span></span>
<span class="line"><span>  [e30172f5] Documenter v1.8.0</span></span>
<span class="line"><span>  [daee34ce] DocumenterCitations v1.3.5</span></span>
<span class="line"><span>  [4710194d] DocumenterVitepress v0.1.3</span></span>
<span class="line"><span>  [4e289a0a] EnumX v1.0.4</span></span>
<span class="line"><span>  [f151be2c] EnzymeCore v0.8.6</span></span>
<span class="line"><span>  [429591f6] ExactPredicates v2.2.8</span></span>
<span class="line"><span>  [318dbb63] Examples v1.0.0 \`../examples\`</span></span>
<span class="line"><span>  [411431e0] Extents v0.1.4</span></span>
<span class="line"><span>  [7a1cc6ca] FFTW v1.8.0</span></span>
<span class="line"><span>  [cc61a311] FLoops v0.2.2</span></span>
<span class="line"><span>  [b9860ae5] FLoopsBase v0.1.1</span></span>
<span class="line"><span>  [9aa1b823] FastClosures v0.3.2</span></span>
<span class="line"><span>  [5789e2e9] FileIO v1.16.5</span></span>
<span class="line"><span>  [8fc22ac5] FilePaths v0.8.3</span></span>
<span class="line"><span>  [48062228] FilePathsBase v0.9.22</span></span>
<span class="line"><span>  [1a297f60] FillArrays v1.13.0</span></span>
<span class="line"><span>  [53c48c17] FixedPointNumbers v0.8.5</span></span>
<span class="line"><span>  [1fa38f19] Format v1.3.7</span></span>
<span class="line"><span>  [f6369f11] ForwardDiff v0.10.38</span></span>
<span class="line"><span>  [b38be410] FreeType v4.1.1</span></span>
<span class="line"><span>  [663a7486] FreeTypeAbstraction v0.10.5</span></span>
<span class="line"><span>  [d9f16b24] Functors v0.5.1</span></span>
<span class="line"><span>  [f7f18e0c] GLFW v3.4.3</span></span>
<span class="line"><span>  [e9467ef8] GLMakie v0.10.16</span></span>
<span class="line"><span>⌃ [0c68f7d7] GPUArrays v10.3.1</span></span>
<span class="line"><span>⌅ [46192b85] GPUArraysCore v0.1.6</span></span>
<span class="line"><span>  [68eda718] GeoFormatTypes v0.4.2</span></span>
<span class="line"><span>  [cf35fbd7] GeoInterface v1.3.8</span></span>
<span class="line"><span>⌅ [5c1252a2] GeometryBasics v0.4.11</span></span>
<span class="line"><span>  [d7ba0133] Git v1.3.1</span></span>
<span class="line"><span>  [a2bd30eb] Graphics v1.1.3</span></span>
<span class="line"><span>  [3955a311] GridLayoutBase v0.11.0</span></span>
<span class="line"><span>  [42e2da0e] Grisu v1.0.2</span></span>
<span class="line"><span>  [0e44f5e4] Hwloc v3.3.0</span></span>
<span class="line"><span>  [34004b35] HypergeometricFunctions v0.3.25</span></span>
<span class="line"><span>  [b5f81e59] IOCapture v0.2.5</span></span>
<span class="line"><span>  [7869d1d1] IRTools v0.4.14</span></span>
<span class="line"><span>  [615f187c] IfElse v0.1.1</span></span>
<span class="line"><span>  [2803e5a7] ImageAxes v0.6.12</span></span>
<span class="line"><span>  [c817782e] ImageBase v0.1.7</span></span>
<span class="line"><span>  [a09fc81d] ImageCore v0.10.5</span></span>
<span class="line"><span>  [82e4d734] ImageIO v0.6.9</span></span>
<span class="line"><span>  [bc367c6b] ImageMetadata v0.9.10</span></span>
<span class="line"><span>  [5e318141] IncompressibleNavierStokes v2.0.1 \`..\`</span></span>
<span class="line"><span>  [9b13fd28] IndirectArrays v1.0.0</span></span>
<span class="line"><span>  [d25df0c9] Inflate v0.1.5</span></span>
<span class="line"><span>  [22cec73e] InitialValues v0.3.1</span></span>
<span class="line"><span>  [a98d9a8b] Interpolations v0.15.1</span></span>
<span class="line"><span>  [d1acc4aa] IntervalArithmetic v0.22.19</span></span>
<span class="line"><span>  [8197267c] IntervalSets v0.7.10</span></span>
<span class="line"><span>  [3587e190] InverseFunctions v0.1.17</span></span>
<span class="line"><span>  [92d709cd] IrrationalConstants v0.2.2</span></span>
<span class="line"><span>  [f1662d9f] Isoband v0.1.1</span></span>
<span class="line"><span>  [c8e1da08] IterTools v1.10.0</span></span>
<span class="line"><span>  [42fd0dbc] IterativeSolvers v0.9.4</span></span>
<span class="line"><span>  [82899510] IteratorInterfaceExtensions v1.0.0</span></span>
<span class="line"><span>  [033835bb] JLD2 v0.5.8</span></span>
<span class="line"><span>  [692b3bcd] JLLWrappers v1.6.1</span></span>
<span class="line"><span>  [682c06a0] JSON v0.21.4</span></span>
<span class="line"><span>  [0f8b85d8] JSON3 v1.14.1</span></span>
<span class="line"><span>  [7d188eb4] JSONSchema v1.4.1</span></span>
<span class="line"><span>  [b835a17e] JpegTurbo v0.1.5</span></span>
<span class="line"><span>  [b14d175d] JuliaVariables v0.2.4</span></span>
<span class="line"><span>  [63c18a36] KernelAbstractions v0.9.29</span></span>
<span class="line"><span>  [5ab0869b] KernelDensity v0.6.9</span></span>
<span class="line"><span>  [929cbde3] LLVM v9.1.3</span></span>
<span class="line"><span>  [b964fa9f] LaTeXStrings v1.4.0</span></span>
<span class="line"><span>  [10f19ff3] LayoutPointers v0.1.17</span></span>
<span class="line"><span>  [0e77f7df] LazilyInitializedFields v1.3.0</span></span>
<span class="line"><span>  [8cdb02fc] LazyModules v0.3.1</span></span>
<span class="line"><span>  [9c8b4983] LightXML v0.9.1</span></span>
<span class="line"><span>  [98b081ad] Literate v2.20.1</span></span>
<span class="line"><span>  [2ab3a3ac] LogExpFunctions v0.3.28</span></span>
<span class="line"><span>  [b2108857] Lux v1.3.3</span></span>
<span class="line"><span>  [bb33d45b] LuxCore v1.2.0</span></span>
<span class="line"><span>  [82251201] LuxLib v1.3.8</span></span>
<span class="line"><span>  [7e8f7934] MLDataDevices v1.6.2</span></span>
<span class="line"><span>  [d8e11817] MLStyle v0.4.17</span></span>
<span class="line"><span>  [f1d291b0] MLUtils v0.4.4</span></span>
<span class="line"><span>  [1914dd2f] MacroTools v0.5.13</span></span>
<span class="line"><span>  [ee78f7c6] Makie v0.21.16</span></span>
<span class="line"><span>  [20f20a25] MakieCore v0.8.10</span></span>
<span class="line"><span>  [d125e4d3] ManualMemory v0.1.8</span></span>
<span class="line"><span>  [dbb5928d] MappedArrays v0.4.2</span></span>
<span class="line"><span>  [d0879d2d] MarkdownAST v0.1.2</span></span>
<span class="line"><span>  [0a4f8689] MathTeXEngine v0.6.2</span></span>
<span class="line"><span>  [7269a6da] MeshIO v0.4.13</span></span>
<span class="line"><span>  [128add7d] MicroCollections v0.2.0</span></span>
<span class="line"><span>  [e1d29d7a] Missings v1.2.0</span></span>
<span class="line"><span>  [66fc600b] ModernGL v1.1.7</span></span>
<span class="line"><span>  [e94cdb99] MosaicViews v0.3.4</span></span>
<span class="line"><span>  [872c559c] NNlib v0.9.24</span></span>
<span class="line"><span>  [77ba4419] NaNMath v1.0.2</span></span>
<span class="line"><span>  [71a1bf82] NameResolution v0.1.5</span></span>
<span class="line"><span>  [f09324ee] Netpbm v1.1.1</span></span>
<span class="line"><span>  [099dac27] NeuralClosure v1.0.0 \`../lib/NeuralClosure\`</span></span>
<span class="line"><span>  [510215fc] Observables v0.5.5</span></span>
<span class="line"><span>  [6fe1bfb0] OffsetArrays v1.14.1</span></span>
<span class="line"><span>  [52e1d378] OpenEXR v0.3.3</span></span>
<span class="line"><span>  [3bd65402] Optimisers v0.4.1</span></span>
<span class="line"><span>  [bac558e1] OrderedCollections v1.6.3</span></span>
<span class="line"><span>  [90014a1f] PDMats v0.11.31</span></span>
<span class="line"><span>  [f57f5aa1] PNGFiles v0.4.3</span></span>
<span class="line"><span>  [19eb6ba3] Packing v0.5.0</span></span>
<span class="line"><span>  [5432bcbf] PaddedViews v0.5.12</span></span>
<span class="line"><span>  [69de0a69] Parsers v2.8.1</span></span>
<span class="line"><span>  [eebad327] PkgVersion v0.3.3</span></span>
<span class="line"><span>  [995b91a9] PlotUtils v1.4.3</span></span>
<span class="line"><span>  [f517fe37] Polyester v0.7.16</span></span>
<span class="line"><span>  [1d0040c9] PolyesterWeave v0.2.2</span></span>
<span class="line"><span>  [647866c9] PolygonOps v0.1.2</span></span>
<span class="line"><span>  [aea7be01] PrecompileTools v1.2.1</span></span>
<span class="line"><span>  [21216c6a] Preferences v1.4.3</span></span>
<span class="line"><span>  [8162dcfd] PrettyPrint v0.2.0</span></span>
<span class="line"><span>  [92933f4c] ProgressMeter v1.10.2</span></span>
<span class="line"><span>  [43287f4e] PtrArrays v1.2.1</span></span>
<span class="line"><span>  [4b34888f] QOI v1.0.1</span></span>
<span class="line"><span>  [1fd47b50] QuadGK v2.11.1</span></span>
<span class="line"><span>  [b3c3ace0] RangeArrays v0.3.2</span></span>
<span class="line"><span>  [c84ed2f1] Ratios v0.4.5</span></span>
<span class="line"><span>  [c1ae055f] RealDot v0.1.0</span></span>
<span class="line"><span>  [3cdcf5f2] RecipesBase v1.3.4</span></span>
<span class="line"><span>  [189a3867] Reexport v1.2.2</span></span>
<span class="line"><span>  [2792f1a3] RegistryInstances v0.1.0</span></span>
<span class="line"><span>  [05181044] RelocatableFolders v1.0.1</span></span>
<span class="line"><span>  [ae029012] Requires v1.3.0</span></span>
<span class="line"><span>  [79098fc4] Rmath v0.8.0</span></span>
<span class="line"><span>  [5eaf0fd0] RoundingEmulator v0.2.1</span></span>
<span class="line"><span>  [fdea26ae] SIMD v3.7.0</span></span>
<span class="line"><span>  [94e857df] SIMDTypes v0.1.0</span></span>
<span class="line"><span>  [6c6a2e73] Scratch v1.2.1</span></span>
<span class="line"><span>  [efcf1570] Setfield v1.1.1</span></span>
<span class="line"><span>  [65257c39] ShaderAbstractions v0.4.1</span></span>
<span class="line"><span>  [605ecd9f] ShowCases v0.1.0</span></span>
<span class="line"><span>  [992d4aef] Showoff v1.0.3</span></span>
<span class="line"><span>  [73760f76] SignedDistanceFields v0.4.0</span></span>
<span class="line"><span>  [699a6c99] SimpleTraits v0.9.4</span></span>
<span class="line"><span>  [45858cf5] Sixel v0.1.3</span></span>
<span class="line"><span>  [a2af1166] SortingAlgorithms v1.2.1</span></span>
<span class="line"><span>  [dc90abb0] SparseInverseSubset v0.1.2</span></span>
<span class="line"><span>  [276daf66] SpecialFunctions v2.4.0</span></span>
<span class="line"><span>  [171d559e] SplittablesBase v0.1.15</span></span>
<span class="line"><span>  [860ef19b] StableRNGs v1.0.2</span></span>
<span class="line"><span>  [cae243ae] StackViews v0.1.1</span></span>
<span class="line"><span>  [aedffcd0] Static v1.1.1</span></span>
<span class="line"><span>  [0d7ed370] StaticArrayInterface v1.8.0</span></span>
<span class="line"><span>  [90137ffa] StaticArrays v1.9.8</span></span>
<span class="line"><span>  [1e83bf80] StaticArraysCore v1.4.3</span></span>
<span class="line"><span>  [10745b16] Statistics v1.11.1</span></span>
<span class="line"><span>  [82ae8749] StatsAPI v1.7.0</span></span>
<span class="line"><span>  [2913bbd2] StatsBase v0.34.3</span></span>
<span class="line"><span>  [4c63d2b9] StatsFuns v1.3.2</span></span>
<span class="line"><span>  [7792a7ef] StrideArraysCore v0.5.7</span></span>
<span class="line"><span>  [69024149] StringEncodings v0.3.7</span></span>
<span class="line"><span>⌅ [09ab397b] StructArrays v0.6.18</span></span>
<span class="line"><span>  [856f2bd8] StructTypes v1.11.0</span></span>
<span class="line"><span>  [3783bdb8] TableTraits v1.0.1</span></span>
<span class="line"><span>  [bd369af6] Tables v1.12.0</span></span>
<span class="line"><span>  [62fd8b95] TensorCore v0.1.1</span></span>
<span class="line"><span>  [1c621080] TestItems v1.0.0</span></span>
<span class="line"><span>  [8290d209] ThreadingUtilities v0.5.2</span></span>
<span class="line"><span>  [731e570b] TiffImages v0.11.1</span></span>
<span class="line"><span>  [3bb67fe8] TranscodingStreams v0.11.3</span></span>
<span class="line"><span>  [28d57a85] Transducers v0.4.84</span></span>
<span class="line"><span>  [981d1d27] TriplotBase v0.1.0</span></span>
<span class="line"><span>  [5c2747f8] URIs v1.5.1</span></span>
<span class="line"><span>  [1cfade01] UnicodeFun v0.4.1</span></span>
<span class="line"><span>  [1986cc42] Unitful v1.21.0</span></span>
<span class="line"><span>  [013be700] UnsafeAtomics v0.2.1</span></span>
<span class="line"><span>  [d80eeb9a] UnsafeAtomicsLLVM v0.2.1</span></span>
<span class="line"><span>  [4004b06d] VTKBase v1.0.1</span></span>
<span class="line"><span>  [e3aaa7dc] WebP v0.1.3</span></span>
<span class="line"><span>  [d49dbf32] WeightInitializers v1.0.4</span></span>
<span class="line"><span>  [efce3f68] WoodburyMatrices v1.0.0</span></span>
<span class="line"><span>  [64499a7a] WriteVTK v1.21.1</span></span>
<span class="line"><span>  [ddb6d928] YAML v0.4.12</span></span>
<span class="line"><span>  [e88e6eb3] Zygote v0.6.73</span></span>
<span class="line"><span>  [700de1a5] ZygoteRules v0.2.5</span></span>
<span class="line"><span>  [6e34b625] Bzip2_jll v1.0.8+2</span></span>
<span class="line"><span>  [4e9b3aee] CRlibm_jll v1.0.1+0</span></span>
<span class="line"><span>  [83423d85] Cairo_jll v1.18.2+1</span></span>
<span class="line"><span>  [ee1fde0b] Dbus_jll v1.14.10+0</span></span>
<span class="line"><span>  [5ae413db] EarCut_jll v2.2.4+0</span></span>
<span class="line"><span>  [2702e6a9] EpollShim_jll v0.0.20230411+0</span></span>
<span class="line"><span>  [2e619515] Expat_jll v2.6.4+0</span></span>
<span class="line"><span>  [b22a6f82] FFMPEG_jll v6.1.2+0</span></span>
<span class="line"><span>  [f5851436] FFTW_jll v3.3.10+1</span></span>
<span class="line"><span>  [a3f928ae] Fontconfig_jll v2.13.96+0</span></span>
<span class="line"><span>  [d7e528f0] FreeType2_jll v2.13.2+0</span></span>
<span class="line"><span>  [559328eb] FriBidi_jll v1.0.14+0</span></span>
<span class="line"><span>  [0656b61e] GLFW_jll v3.4.0+1</span></span>
<span class="line"><span>  [78b55507] Gettext_jll v0.21.0+0</span></span>
<span class="line"><span>  [59f7168a] Giflib_jll v5.2.2+0</span></span>
<span class="line"><span>  [f8c6e375] Git_jll v2.46.2+0</span></span>
<span class="line"><span>  [7746bdde] Glib_jll v2.80.5+0</span></span>
<span class="line"><span>  [3b182d85] Graphite2_jll v1.3.14+0</span></span>
<span class="line"><span>  [2e76f6c2] HarfBuzz_jll v8.3.1+0</span></span>
<span class="line"><span>  [e33a78d0] Hwloc_jll v2.11.2+1</span></span>
<span class="line"><span>  [905a6f67] Imath_jll v3.1.11+0</span></span>
<span class="line"><span>  [1d5cc7b8] IntelOpenMP_jll v2024.2.1+0</span></span>
<span class="line"><span>  [aacddb02] JpegTurbo_jll v3.0.4+0</span></span>
<span class="line"><span>  [c1c5ebd0] LAME_jll v3.100.2+0</span></span>
<span class="line"><span>  [88015f11] LERC_jll v4.0.0+0</span></span>
<span class="line"><span>  [dad2f222] LLVMExtra_jll v0.0.34+0</span></span>
<span class="line"><span>  [1d63c593] LLVMOpenMP_jll v18.1.7+0</span></span>
<span class="line"><span>  [dd4b983a] LZO_jll v2.10.2+1</span></span>
<span class="line"><span>⌅ [e9f186c6] Libffi_jll v3.2.2+1</span></span>
<span class="line"><span>  [d4300ac3] Libgcrypt_jll v1.11.0+0</span></span>
<span class="line"><span>  [7e76a0d4] Libglvnd_jll v1.6.0+0</span></span>
<span class="line"><span>  [7add5ba3] Libgpg_error_jll v1.50.0+0</span></span>
<span class="line"><span>  [94ce4f54] Libiconv_jll v1.17.0+1</span></span>
<span class="line"><span>  [4b2f31a3] Libmount_jll v2.40.1+0</span></span>
<span class="line"><span>  [89763e89] Libtiff_jll v4.7.0+0</span></span>
<span class="line"><span>  [38a345b3] Libuuid_jll v2.40.1+0</span></span>
<span class="line"><span>  [856f044c] MKL_jll v2024.2.0+0</span></span>
<span class="line"><span>  [c7aee132] NodeJS_20_jll v20.12.2+0</span></span>
<span class="line"><span>  [e7412a2a] Ogg_jll v1.3.5+1</span></span>
<span class="line"><span>  [18a262bb] OpenEXR_jll v3.2.4+0</span></span>
<span class="line"><span>  [458c3c95] OpenSSL_jll v3.0.15+1</span></span>
<span class="line"><span>  [efe28fd5] OpenSpecFun_jll v0.5.5+0</span></span>
<span class="line"><span>  [91d4177d] Opus_jll v1.3.3+0</span></span>
<span class="line"><span>  [36c8627f] Pango_jll v1.54.1+0</span></span>
<span class="line"><span>  [30392449] Pixman_jll v0.43.4+0</span></span>
<span class="line"><span>  [f50d1b31] Rmath_jll v0.5.1+0</span></span>
<span class="line"><span>  [a2964d1f] Wayland_jll v1.21.0+1</span></span>
<span class="line"><span>  [2381bf8a] Wayland_protocols_jll v1.31.0+0</span></span>
<span class="line"><span>  [02c8fc9c] XML2_jll v2.13.5+0</span></span>
<span class="line"><span>  [aed1982a] XSLT_jll v1.1.41+0</span></span>
<span class="line"><span>  [ffd25f8a] XZ_jll v5.6.3+0</span></span>
<span class="line"><span>  [4f6342f7] Xorg_libX11_jll v1.8.6+0</span></span>
<span class="line"><span>  [0c0b7dd1] Xorg_libXau_jll v1.0.11+0</span></span>
<span class="line"><span>  [935fb764] Xorg_libXcursor_jll v1.2.0+4</span></span>
<span class="line"><span>  [a3789734] Xorg_libXdmcp_jll v1.1.4+0</span></span>
<span class="line"><span>  [1082639a] Xorg_libXext_jll v1.3.6+0</span></span>
<span class="line"><span>  [d091e8ba] Xorg_libXfixes_jll v5.0.3+4</span></span>
<span class="line"><span>  [a51aa0fd] Xorg_libXi_jll v1.7.10+4</span></span>
<span class="line"><span>  [d1454406] Xorg_libXinerama_jll v1.1.4+4</span></span>
<span class="line"><span>  [ec84b674] Xorg_libXrandr_jll v1.5.2+4</span></span>
<span class="line"><span>  [ea2f1a96] Xorg_libXrender_jll v0.9.11+0</span></span>
<span class="line"><span>  [14d82f49] Xorg_libpthread_stubs_jll v0.1.1+0</span></span>
<span class="line"><span>  [c7cfdc94] Xorg_libxcb_jll v1.17.0+0</span></span>
<span class="line"><span>  [cc61e674] Xorg_libxkbfile_jll v1.1.2+0</span></span>
<span class="line"><span>  [35661453] Xorg_xkbcomp_jll v1.4.6+0</span></span>
<span class="line"><span>  [33bec58e] Xorg_xkeyboard_config_jll v2.39.0+0</span></span>
<span class="line"><span>  [c5fb5394] Xorg_xtrans_jll v1.5.0+0</span></span>
<span class="line"><span>  [3161d3a3] Zstd_jll v1.5.6+1</span></span>
<span class="line"><span>  [9a68df92] isoband_jll v0.2.3+0</span></span>
<span class="line"><span>  [a4ae2306] libaom_jll v3.9.0+0</span></span>
<span class="line"><span>  [0ac62f75] libass_jll v0.15.2+0</span></span>
<span class="line"><span>  [1183f4f0] libdecor_jll v0.2.2+0</span></span>
<span class="line"><span>  [f638f0a6] libfdk_aac_jll v2.0.3+0</span></span>
<span class="line"><span>  [b53b4c65] libpng_jll v1.6.44+0</span></span>
<span class="line"><span>  [075b6546] libsixel_jll v1.10.3+1</span></span>
<span class="line"><span>  [f27f6e37] libvorbis_jll v1.3.7+2</span></span>
<span class="line"><span>  [c5f90fcd] libwebp_jll v1.4.0+0</span></span>
<span class="line"><span>  [1317d2d5] oneTBB_jll v2021.12.0+0</span></span>
<span class="line"><span>  [1270edf5] x264_jll v10164.0.0+0</span></span>
<span class="line"><span>⌅ [dfaa095f] x265_jll v3.6.0+0</span></span>
<span class="line"><span>  [d8fb68d0] xkbcommon_jll v1.4.1+1</span></span>
<span class="line"><span>  [0dad84c5] ArgTools v1.1.2</span></span>
<span class="line"><span>  [56f22d72] Artifacts v1.11.0</span></span>
<span class="line"><span>  [2a0f44e3] Base64 v1.11.0</span></span>
<span class="line"><span>  [8bf52ea8] CRC32c v1.11.0</span></span>
<span class="line"><span>  [ade2ca70] Dates v1.11.0</span></span>
<span class="line"><span>  [8ba89e20] Distributed v1.11.0</span></span>
<span class="line"><span>  [f43a241f] Downloads v1.6.0</span></span>
<span class="line"><span>  [7b1f6079] FileWatching v1.11.0</span></span>
<span class="line"><span>  [9fa8497b] Future v1.11.0</span></span>
<span class="line"><span>  [b77e0a4c] InteractiveUtils v1.11.0</span></span>
<span class="line"><span>  [4af54fe1] LazyArtifacts v1.11.0</span></span>
<span class="line"><span>  [b27032c2] LibCURL v0.6.4</span></span>
<span class="line"><span>  [76f85450] LibGit2 v1.11.0</span></span>
<span class="line"><span>  [8f399da3] Libdl v1.11.0</span></span>
<span class="line"><span>  [37e2e46d] LinearAlgebra v1.11.0</span></span>
<span class="line"><span>  [56ddb016] Logging v1.11.0</span></span>
<span class="line"><span>  [d6f4376e] Markdown v1.11.0</span></span>
<span class="line"><span>  [a63ad114] Mmap v1.11.0</span></span>
<span class="line"><span>  [ca575930] NetworkOptions v1.2.0</span></span>
<span class="line"><span>  [44cfe95a] Pkg v1.11.0</span></span>
<span class="line"><span>  [de0858da] Printf v1.11.0</span></span>
<span class="line"><span>  [3fa0cd96] REPL v1.11.0</span></span>
<span class="line"><span>  [9a3f8284] Random v1.11.0</span></span>
<span class="line"><span>  [ea8e919c] SHA v0.7.0</span></span>
<span class="line"><span>  [9e88b42a] Serialization v1.11.0</span></span>
<span class="line"><span>  [1a1011a3] SharedArrays v1.11.0</span></span>
<span class="line"><span>  [6462fe0b] Sockets v1.11.0</span></span>
<span class="line"><span>  [2f01184e] SparseArrays v1.11.0</span></span>
<span class="line"><span>  [f489334b] StyledStrings v1.11.0</span></span>
<span class="line"><span>  [4607b0f0] SuiteSparse</span></span>
<span class="line"><span>  [fa267f1f] TOML v1.0.3</span></span>
<span class="line"><span>  [a4e569a6] Tar v1.10.0</span></span>
<span class="line"><span>  [8dfed614] Test v1.11.0</span></span>
<span class="line"><span>  [cf7118a7] UUIDs v1.11.0</span></span>
<span class="line"><span>  [4ec0a83e] Unicode v1.11.0</span></span>
<span class="line"><span>  [e66e0078] CompilerSupportLibraries_jll v1.1.1+0</span></span>
<span class="line"><span>  [deac9b47] LibCURL_jll v8.6.0+0</span></span>
<span class="line"><span>  [e37daf67] LibGit2_jll v1.7.2+0</span></span>
<span class="line"><span>  [29816b5a] LibSSH2_jll v1.11.0+1</span></span>
<span class="line"><span>  [c8ffd9c3] MbedTLS_jll v2.28.6+0</span></span>
<span class="line"><span>  [14a3606d] MozillaCACerts_jll v2023.12.12</span></span>
<span class="line"><span>  [4536629a] OpenBLAS_jll v0.3.27+1</span></span>
<span class="line"><span>  [05823500] OpenLibm_jll v0.8.1+2</span></span>
<span class="line"><span>  [efcefdf7] PCRE2_jll v10.42.0+1</span></span>
<span class="line"><span>  [bea87d4a] SuiteSparse_jll v7.7.0+0</span></span>
<span class="line"><span>  [83775a58] Zlib_jll v1.2.13+1</span></span>
<span class="line"><span>  [8e850b90] libblastrampoline_jll v5.11.0+0</span></span>
<span class="line"><span>  [8e850ede] nghttp2_jll v1.59.0+0</span></span>
<span class="line"><span>  [3f19e933] p7zip_jll v17.4.0+2</span></span>
<span class="line"><span>Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use \`status --outdated -m\`</span></span></code></pre></div>`,9)]))}const f=n(e,[["render",i]]);export{b as __pageData,f as default};
