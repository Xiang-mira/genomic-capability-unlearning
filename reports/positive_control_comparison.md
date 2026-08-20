# Positive-control comparison: classical baselines vs published gLM/BioJEPA numbers

`baseline_disjoint` = max(k-mer3-5, k-mer3-6, mean-over-seeds CNN) MCC on the split
matching the competitor table's test set (NTv3: official split, verified
chromosome-disjoint; GUE: official split, disjointness unverified). CNN uses the MEAN
across seeds, not the best -- max-of-N vs a single-point competitor number is an
optimistic-selection bias (this flipped the sign of two rows in an earlier version of
this table). `n_seeds`/`cnn_seed_std` are reported so any n=1 or high-variance cell is
visible rather than silent. `baseline_random` = same baseline architecture on a
pooled-and-reshuffled random split (no matching gLM number exists there -- robustness
check only). `gap` = best published model's MCC minus baseline_disjoint; positive and
large = clean positive control. Published-side numbers are themselves single point
estimates (their own reported +/- is not carried through here) -- treat small |gap|
values as ties, not wins, in either direction.

| section | task | baseline_disjoint | n_seeds | cnn_seed_std | baseline_random | best_glm | best_glm_mcc | gap | Hyena7M | Cad8M | DB2 | GROVER | NTv2 | GJ-T | GJ-B | S(test) | B(test) |
|:--|:--|--:|--:|--:|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| GUE | Promoter TATA | 0.6147 | 3 | 0.0611 | 0.7564 | GJ-B | 0.8210 | 0.2063 | 0.5270 | 0.6500 | 0.6590 | 0.6000 | 0.8040 | 0.8130 | 0.8210 | 0.6870 | 0.6100 |
| GUE | TF Human 4 | 0.4775 | 3 | 0.0320 | 0.4077 | DB2 | 0.6310 | 0.1535 | 0.4400 | 0.5210 | 0.6310 | 0.4740 | 0.4960 | 0.4990 | 0.5590 | 0.3570 | 0.4270 |
| GUE | TF Human 1 | 0.6575 | 3 | 0.0090 | 0.5637 | DB2 | 0.7190 | 0.0615 | 0.6300 | 0.6090 | 0.7190 | 0.5910 | 0.6440 | 0.6760 | 0.6800 | 0.6250 | 0.6360 |
| GUE | Promoter NoTATA | 0.9153 | 3 | 0.0029 | 0.9305 | GJ-B | 0.9600 | 0.0447 | 0.8860 | 0.8680 | 0.9200 | 0.8950 | 0.9130 | 0.9320 | 0.9600 | 0.8930 | 0.8970 |
| GUE | TF Human 5 | 0.7449 | 3 | 0.0076 | 0.7259 | DB2 | 0.7890 | 0.0441 | 0.7000 | 0.6650 | 0.7890 | 0.6830 | 0.7100 | 0.6130 | 0.7600 | 0.5270 | 0.5510 |
| GUE | TF Human 2 | 0.6908 | 3 | 0.0159 | 0.5226 | GJ-B | 0.7340 | 0.0432 | 0.6210 | 0.6790 | 0.7180 | 0.6400 | 0.6980 | 0.7240 | 0.7340 | 0.6510 | 0.6570 |
| GUE | Core Prom. NoTATA | 0.6895 | 3 | 0.0166 | 0.7041 | GJ-B | 0.7250 | 0.0355 | 0.6620 | 0.6640 | 0.7120 | 0.6530 | 0.7070 | 0.7040 | 0.7250 | 0.6750 | 0.6810 |
| GUE | Core Prom. All | 0.6871 | 3 | 0.0056 | 0.6932 | DB2 | 0.7170 | 0.0299 | 0.6410 | 0.6450 | 0.7170 | 0.6350 | 0.6940 | 0.7050 | 0.7030 | 0.6630 | 0.6700 |
| GUE | Splice All | 0.8714 | 3 | 0.0089 | 0.8790 | GJ-B | 0.8900 | 0.0186 | 0.8140 | 0.8580 | 0.8310 | 0.8150 | 0.8800 | 0.8360 | 0.8900 | 0.7840 | 0.7950 |
| GUE | Core Prom. TATA | 0.8104 | 3 | 0.0214 | 0.8333 | GJ-B | 0.8190 | 0.0086 | 0.7090 | 0.7410 | 0.7330 | 0.5840 | 0.7410 | 0.7630 | 0.8190 | 0.7980 | 0.7550 |
| GUE | Promoter All | 0.9134 | 3 | 0.0008 | 0.9131 | NTv2 | 0.9210 | 0.0076 | 0.8450 | 0.8250 | 0.8250 | 0.8470 | 0.9210 | 0.8970 | 0.9120 | 0.8620 | 0.8720 |
| GUE | TF Human 3 | 0.6307 | 3 | 0.0620 | 0.5627 | DB2 | 0.6340 | 0.0033 | 0.5970 | 0.5520 | 0.6340 | 0.5700 | 0.6210 | 0.5480 | 0.6110 | 0.4660 | 0.5060 |
| NT | Splice All | 0.3731 | 3 | 0.0102 | 0.3314 | NTv2 | 0.9710 | 0.5979 | 0.8430 | 0.9260 | 0.8280 | 0.8470 | 0.9710 | 0.9530 | 0.9690 | 0.9480 | 0.9560 |
| NT | Splice Acceptor | 0.6190 | 3 | 0.0086 | 0.6329 | GJ-B | 0.9710 | 0.3520 | 0.7830 | 0.8440 | 0.7940 | 0.8360 | 0.9620 | 0.9590 | 0.9710 | 0.9530 | 0.9500 |
| NT | Splice Donor | 0.6764 | 3 | 0.0028 | 0.6868 | GJ-B | 0.9840 | 0.3076 | 0.8840 | 0.9520 | 0.8080 | 0.8310 | 0.9720 | 0.9720 | 0.9840 | 0.9650 | 0.9570 |
| NT | H4K20me1 | 0.5693 | 3 | 0.0057 | 0.5819 | GJ-B | 0.6920 | 0.1227 | 0.6230 | 0.5900 | 0.6460 | 0.6560 | 0.6490 | 0.6590 | 0.6920 | 0.6100 | 0.6290 |
| NT | H3K9ac | 0.4982 | 3 | 0.0051 | 0.4882 | GJ-B | 0.6110 | 0.1128 | 0.4340 | 0.4900 | 0.5680 | 0.5130 | 0.5740 | 0.5380 | 0.6110 | 0.4480 | 0.5000 |
| NT | Enhancer Type | 0.4667 | 3 | 0.0153 | 0.4542 | NTv2 | 0.5760 | 0.1093 | 0.4980 | 0.4650 | 0.5280 | 0.5260 | 0.5760 | 0.5720 | 0.5530 | 0.4610 | 0.4680 |
| NT | H3K9me3 | 0.4184 | 3 | 0.0233 | 0.3855 | GJ-B | 0.5200 | 0.1016 | 0.4060 | 0.4210 | 0.4630 | 0.4280 | 0.4850 | 0.4850 | 0.5200 | 0.3870 | 0.4390 |
| NT | H3K27ac | 0.4336 | 3 | 0.0049 | 0.4458 | GJ-B | 0.5310 | 0.0974 | 0.4580 | 0.4360 | 0.4970 | 0.5030 | 0.4340 | 0.5010 | 0.5310 | 0.4360 | 0.4420 |
| NT | H3K4me1 | 0.4120 | 3 | 0.0305 | 0.4474 | GJ-B | 0.5090 | 0.0970 | 0.4140 | 0.3860 | 0.4790 | 0.3890 | 0.4830 | 0.4810 | 0.5090 | 0.4590 | 0.4660 |
| NT | H3K36me3 | 0.5757 | 3 | 0.0065 | 0.5880 | GJ-B | 0.6670 | 0.0913 | 0.5780 | 0.5430 | 0.6510 | 0.5980 | 0.5920 | 0.6180 | 0.6670 | 0.5760 | 0.5770 |
| NT | Promoter NoTATA | 0.7357 | 3 | 0.0030 | 0.7375 | NTv2 | 0.8230 | 0.0873 | 0.7260 | 0.7370 | 0.7380 | 0.7620 | 0.8230 | 0.7670 | 0.8070 | 0.7680 | 0.7500 |
| NT | Enhancer | 0.4933 | 3 | 0.0171 | 0.5051 | NTv2 | 0.5730 | 0.0797 | 0.5010 | 0.4480 | 0.5090 | 0.5000 | 0.5730 | 0.5370 | 0.5330 | 0.5030 | 0.5130 |
| NT | H3K27me3 | 0.5613 | 3 | 0.0071 | 0.5627 | GJ-B | 0.6410 | 0.0797 | 0.5800 | 0.5530 | 0.5970 | 0.6120 | 0.5990 | 0.5880 | 0.6410 | 0.5690 | 0.5740 |
| NT | H3K4me2 | 0.5102 | 3 | 0.0029 | 0.5353 | GJ-B | 0.5850 | 0.0748 | 0.5080 | 0.5270 | 0.5010 | 0.5290 | 0.5490 | 0.5530 | 0.5850 | 0.5050 | 0.5270 |
| NT | H2AFZ | 0.4696 | 3 | 0.0062 | 0.4731 | Hyena7M | 0.5350 | 0.0654 | 0.5350 | 0.5040 | 0.4820 | 0.5350 | 0.5340 | 0.5050 | 0.5160 | 0.4660 | 0.4730 |
| NT | H3K4me3 | 0.5942 | 3 | 0.0094 | 0.6275 | DB2 | 0.6590 | 0.0648 | 0.6570 | 0.6060 | 0.6590 | 0.6380 | 0.5950 | 0.6290 | 0.6540 | 0.6220 | 0.5730 |
| NT | Promoter TATA | 0.9135 | 3 | 0.0097 | 0.8757 | GJ-B | 0.9600 | 0.0465 | 0.8400 | 0.7980 | 0.8320 | 0.8300 | 0.8660 | 0.9050 | 0.9600 | 0.8890 | 0.8890 |
| NT | Promoter All | 0.7481 | 3 | 0.0126 | 0.7444 | NTv2 | 0.7880 | 0.0399 | 0.6930 | 0.7250 | 0.7240 | 0.7090 | 0.7880 | 0.7600 | 0.7870 | 0.7460 | 0.7500 |

## Summary
- Mean gap (published best - our baseline), GUE: 0.0547
- Mean gap (published best - our baseline), NT:  0.1404
- Tasks where baseline is within 0.05 MCC of the best published model (or beats it): 11/30
