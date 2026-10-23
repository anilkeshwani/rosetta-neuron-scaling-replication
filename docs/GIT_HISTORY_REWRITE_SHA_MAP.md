# Commit SHA map for the 2026-10-24 history rewrites

The history was rewritten twice on 2026-10-24: first to replace a work email in the two scaffold commits with the GitHub noreply address, then to remove details of the machine the runs used. File contents changed only in the second rewrite, and only in wording about that machine. Run manifests (`run_manifest.json`, field `repo_sha`) written before the rewrites carry the original SHAs; map them with this table.

| Original SHA | After the first rewrite | Final SHA | Subject |
|---|---|---|---|
| (none) | `e47f313` | `27ad657e8391` | Map old to new commit SHAs after the author-email rewrite |
| `9e2bea2cb266` | `3ea6bb2bfea8` | `af8c249e6317` | Record the GitHub remote in the handoff |
| `e608d635e9bf` | `8c1bb12989ed` | `eb793e6441bc` | Summarize phase 0 and phase 1 results so far |
| `4236535cf265` | `639f0d3c7a27` | `822356ab9590` | Add the Pythia-1.4B/OPT-1.3B count and the first three-point fit |
| `7cc0e4fc10f4` | `88686461417b` | `439cdc6175ad` | Add phase 0 and partial phase 1 results and a status note to the handoff |
| `3007e023007b` | `1c7af30d5e51` | `b9679e06d2ed` | Record the paper's per-pair Pythia-OPT counts decoded from Fig. 25 |
| `170a86548895` | `8862b13c27f0` | `8a5bb803df14` | Halve the batch size for the 1.4B pairs to fit in GPU memory |
| `0d26459ab070` | `54e9b7e47220` | `e76fb447f161` | Use expandable CUDA allocator segments on the local GPUs |
| `962679a434a1` | `1c512ea926ae` | `484ab64fe428` | Tick off phase 0 checks and record the smoke-test result |
| `19b579210a79` | `1298dbd18dac` | `ab477ef6c72e` | Record first measured timings |
| `9b2d8c3da383` | `bf64724c05ed` | `cebbf4523ae8` | Document the GPU-count dependence of upstream's token sample |
| `c2ea7abfc448` | `44f9ac830c7b` | `a1fefe2917f1` | Add the local GPU schedule for phase 1 small pairs and the null |
| `c95beb3b0871` | `b1f1c1678aba` | `50dcc2814364` | Skip superseded run dirs when collecting Rosetta counts |
| `45e30df7b5b3` | `aa4b12854497` | `521a72546192` | Write a provenance sidecar next to each built token cache |
| `c94039cea964` | `10f59e0cf978` | `f8a0b95087c5` | Start forced reruns and retries from an empty run directory |
| `d6160e24300c` | `a231c29a6610` | `5967fe091fce` | Split reference fit keys on the first period only |
| `d46d02eef3d2` | `b2405fbd9825` | `b7b02477ef8c` | Reject power-law fits with fewer than two distinct x values |
| `a83305d04751` | `57e05b38bcf1` | `42ff54a03b9b` | Count only the k values a run's manifest asked for |
| `3614eba88492` | `f3b3a5244b71` | `e7487ac8295c` | Use the authors' 8-rank token sample for phase 1 and the random-init null |
| `282f452e315b` | `6a5b5e42249c` | `fdab621c4173` | Let experiments draw their token sample as an N-rank run would |
| `81dec7942653` | `ea3bac87357a` | `0ae70a1eb76f` | Build token cache ranks in parallel and create Pile stats if missing |
| `9c183f3c37f9` | `b65ed2485b6b` | `9513cd415ba0` | Record TOKEN_CACHE, GPU selection and port in run manifests |
| `095ef1fe3be3` | `deb6e189f72b` | `26b713615570` | Add CPU builder for upstream's token cache at a chosen world size |
| `6978b27ef896` | `d1121f932445` | `c3b77cc4f9e4` | Add best-buddy pair-list comparison script |
| `0e32faa2767e` | `0af841c5522d` | `269d81a74cb8` | Ignore driver and download logs |
| `108d7d6b60db` | `602a35aaf663` | `cf20868c36f4` | Record applied transformers pin in handoff |
| `6db655b050e6` | `2dec7e1344ea` | `6254604b0d27` | Scaffold Rosetta Neuron scaling replication project |
