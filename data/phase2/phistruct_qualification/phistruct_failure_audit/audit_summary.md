# PHIStruct Failure Audit

- Dataset: 7627 RBPs; host classes: acinetobacter, enterobacter, enterococcus, escherichia, klebsiella, pseudomonas, staphylococcus
- Original/reconstructed SaProt test macro-F1: 0.454732
- Reconstructed BLASTp test macro-F1: 0.475180
- Delta (SaProt - BLASTp): -0.020448
- Bootstrap valid/invalid: 10000 / 5778
- Bootstrap 95% CI: [-0.11306635154120383, 0.07179410645613148]
- P(delta > 0): 0.3594
- P(delta < 0): 0.6406
- BLAST tiny-class-only driver: NO
- HMMER sanity: pass
- Final PHIStruct status: PHISTRUCT_FAILURE_NOT_STATISTICALLY_RESOLVED

## Per-Genus Delta

- acinetobacter: support=159, SaProt F1=0.676806, BLAST F1=0.624473, delta=0.052334
- enterobacter: support=1, SaProt F1=0.074074, BLAST F1=0.000000, delta=0.074074
- enterococcus: support=14, SaProt F1=0.903226, BLAST F1=0.933333, delta=-0.030108
- escherichia: support=12, SaProt F1=0.115385, BLAST F1=0.333333, delta=-0.217949
- klebsiella: support=6, SaProt F1=0.088889, BLAST F1=0.272727, delta=-0.183838
- pseudomonas: support=117, SaProt F1=0.709360, BLAST F1=0.623932, delta=0.085428
- staphylococcus: support=8, SaProt F1=0.615385, BLAST F1=0.538462, delta=0.076923
