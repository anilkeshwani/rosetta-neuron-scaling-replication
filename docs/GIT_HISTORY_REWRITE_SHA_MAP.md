# Commit SHA map for the 2026-10-24 author-email rewrite

The history was rewritten once, on 2026-10-24, to replace a work email in the two scaffold commits with the GitHub noreply address. Every commit SHA changed; file contents did not. Run manifests (`run_manifest.json`, field `repo_sha`) written before the rewrite carry the old SHAs. Map them with this table.

| Old SHA | New SHA | Subject |
|---|---|---|
| `9e2bea2cb266be71f8782c8517530170e22a3b3d` | `3ea6bb2bfea80048e10048ab02d4f532c08c39f3` | Record the GitHub remote in the handoff |
| `e608d635e9bf467d17a5545e916b810c24fa0b4c` | `8c1bb12989eda49ed69b02e86b3f52716d996472` | Summarize phase 0 and phase 1 results so far |
| `4236535cf265d2360bf560bf721b402d015a2a03` | `639f0d3c7a27e3631401106d0c787b01c00f550d` | Add the Pythia-1.4B/OPT-1.3B count and the first three-point fit |
| `7cc0e4fc10f4df4cdd5ac9dc147bc648fdb44464` | `88686461417b6d9153116eb39b888ce6d7eb3d26` | Add phase 0 and partial phase 1 results and a status note to the handoff |
| `3007e023007bdcdeb3397262bad4524d4ae73762` | `1c7af30d5e51253975a8cd7537ac2a959644b90a` | Record the paper's per-pair Pythia-OPT counts decoded from Fig. 25 |
| `170a86548895413213ac37f924ca394ac4c958ec` | `8862b13c27f0ea980d54835d83c82b999ee5b5f3` | Halve the batch size for the 1.4B pairs to fit in GPU memory |
| `0d26459ab070d1512044b6e7305ca1738d5cdfd3` | `54e9b7e472204c4a364e41646fe7a33081ddd4df` | Use expandable CUDA allocator segments on the local GPUs |
| `962679a434a1c2f71a0adb6f51c9c6cf53a5751e` | `1c512ea926aedcc447a647fac344cf02c9a6a993` | Tick off phase 0 checks and record the smoke-test result |
| `19b579210a7904a2a1e2290f89f004c7c209fe2b` | `1298dbd18dace6195b7420aa2bee51d306852cb9` | Record first measured timings |
| `9b2d8c3da3835d2b519aa7890996d6c6e0ff4b90` | `bf64724c05ed5d751395a14a9ea2b228465567e0` | Document the GPU-count dependence of upstream's token sample |
| `c2ea7abfc448ccaa01d9f5160f063429b58ec776` | `44f9ac830c7b97f77b3571a38d02d9c486bf636d` | Add the the local machine GPU schedule for phase 1 small pairs and the null |
| `c95beb3b08719fd49d2f506768bf1b7ea2f8f18f` | `b1f1c1678aba2a4b810aef0d53b26df469c5b2e8` | Skip superseded run dirs when collecting Rosetta counts |
| `45e30df7b5b3c3044934ed4ed31a621acd29de38` | `aa4b128544974739980caa8b1d2220ba491f069b` | Write a provenance sidecar next to each built token cache |
| `c94039cea9647d9f0011788c81979fc736c260a5` | `10f59e0cf9788179838a1b11bf8a1535b7845eff` | Start forced reruns and retries from an empty run directory |
| `d6160e24300ccecfde716949f1baa5169bd26d46` | `a231c29a6610c0673c67c83d8b03d56d6309424e` | Split reference fit keys on the first period only |
| `d46d02eef3d29e17d875df9f37b8ad829ec95a5b` | `b2405fbd9825da69845821bcd235192e499fa0c5` | Reject power-law fits with fewer than two distinct x values |
| `a83305d047519cd2caf8b2b64bd123174cf751de` | `57e05b38bcf1e1e386a702ac6ba37dc4d3e75d6f` | Count only the k values a run's manifest asked for |
| `3614eba88492f377ab989f99484218ffb99c4d2b` | `f3b3a5244b7193142e4e3c2b70e255125265eb65` | Use the authors' 8-rank token sample for phase 1 and the random-init null |
| `282f452e315b8f01894519fbeab8545e28fca7aa` | `6a5b5e42249c84963cc65a78abdf47b90032ef8b` | Let experiments draw their token sample as an N-rank run would |
| `81dec79426536127b8c39ecfb5aff57eb0783ab5` | `ea3bac87357ad0298104c2f0f002e3072cf75543` | Build token cache ranks in parallel and create Pile stats if missing |
| `9c183f3c37f9b6c6d9f947396c40d2f2565b2b79` | `b65ed2485b6b0c6a3566a0b55382618bf63b208e` | Record TOKEN_CACHE, GPU selection and port in run manifests |
| `095ef1fe3be392c7b369950d6308243ff878953c` | `deb6e189f72bde9d51a5822fa368912d46fd2eee` | Add CPU builder for upstream's token cache at a chosen world size |
| `6978b27ef896510ecc1f3894a0ddca03312fb067` | `d1121f93244574cc3a8711a331121c86766a00e1` | Add best-buddy pair-list comparison script |
| `0e32faa2767e9fa0cdb54f4e7056d8afcda772b8` | `0af841c5522d356eee008cf3594fe43d51c23cd4` | Ignore driver and download logs |
| `108d7d6b60db79ea6ac7398eb7968eb1911e6b8b` | `602a35aaf6639d5ca88e0f60274c7f5522ec14b0` | Record applied transformers pin in handoff |
| `6db655b050e685d400160a5cffd68e47f41f28d0` | `2dec7e1344ea96e600ac8b97373665a336adb257` | Scaffold Rosetta Neuron scaling replication project |
