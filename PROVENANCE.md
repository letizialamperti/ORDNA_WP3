# Origin and method choices

Based on ORDNA by Sanchez, Stalder, Lamperti et al. (2025),
Methods in Ecology and Evolution, DOI: 10.1111/2041-210X.70033.
Upstream: https://gitlab.renkulab.io/dnai/ordna
Local source revision: e36293f8c3f5150aad6a845614ea948a8c2a862b.
The upstream MIT license is retained in LICENSE. The paper itself is not bundled.

`model.py` ports `representation_module.py` and the triplet forward/loss from
`triplets.py`, preserving state-dict names. The encoder has trainable nucleotide
tokens, two shared convolutions for forward/reverse, a 2048-unit MLP, a sample
token, four attention layers, and a shared LayerNorm. A linear head produces
the final embedding. No protection, habitat or species labels enter the encoder.

Changes: integer token count17 (upstream `17.` is incompatible with Embedding),
torch reshape/permute instead of einops, standalone training instead of Lightning,
and a static copy of the upstream IUPAC token order instead of Biopython.
Original source files are not edited. The small CPU equivalence check against the
local upstream verified state dict, forward output, loss and all gradients before
this package was prepared. Portable tests cover the port and fold isolation.

Default representation256/output64 is a downstream-classification adaptation;
the paper's ordination output was2D. `--embedding-dim 2` allows a2D experiment.
The CSV name `preprojector` denotes the256D encoder representation; the triplet
head itself is linear, unlike the upstream BT projector MLP.

The loss follows **upstream implementation**, not a corrected interpretation of
paper equation2: triplet margin loss plus a dynamic-weight sum of Frobenius norms
over the three embedding batches. This has neither the per-sample mean nor the
division by3 printed in the paper. Batch size therefore affects the regularizer's
relative scale. CLI defaults match the upstream parser: margin0.3, max norm
weight0.1, lr1e-4. Upstream constructor defaults differ (margin2, weight0.003,
lr1e-5); they are not silently substituted.

Other explicit adaptations: random uniform training sample pairs and random
non-overlapping read subsets, rather than the upstream exhaustive ordered-pair
dataset with contiguous read windows. Fixed updates/epochs must be declared;
we do not call one reduced-budget epoch equivalent to an upstream full epoch.
AdamW weight decay0.01 and scheduler milestones2,8/gamma0.3 follow the upstream
optimizer. Export is frozen, deterministic per sample, equal subset means including
the final partial subset. Population subset std is provided for compatibility.
This aggregation is declared because other BT exporters can drop partial subsets.

The split names are user-provided `strict_5km_3fold`. These files contain only IDs
and roles, not coordinates: the code validates partition consistency, but cannot
independently certify a5km spatial buffer without coordinates and split-generation
rules. Each of432 samples is validation exactly once; train counts290/286/288,
validation142/146/144. The28 habitat-holdout samples are absent from the supplied
splits according to the existing project protocol; no files outside the split
are selected for training or export.

Training samples below2*subset_size valid pairs are explicitly excluded and audited;
the original split is preserved separately and validation roles remain unchanged.
Exports omit excluded training IDs. Model checkpoints include original/effective
training ID lists and exclusions.
