# Alaala eval (dev set)

29 labeled queries (22 Filipino term, 7 English); 7 marked as no matches and skipped.

## Overall

| method | Hit@1 | Recall@10 | nDCG@10 | MRR |
|---|---|---|---|---|
| Baseline (plain CLIP) | 0.379 | 0.494 | 0.450 | 0.499 |
| Alaala v1 (descriptions only) | 0.586 | 0.760 | 0.694 | 0.720 |
| Alaala v2 (+ original word) | 0.586 | 0.751 | 0.681 | 0.722 |

## By query type

| type | method | Hit@1 | Recall@10 | nDCG@10 | MRR |
|---|---|---|---|---|---|
| English | Alaala v1 (descriptions only) | 0.714 | 0.810 | 0.739 | 0.786 |
| English | Alaala v2 (+ original word) | 0.714 | 0.810 | 0.739 | 0.786 |
| English | Baseline (plain CLIP) | 0.714 | 0.810 | 0.739 | 0.786 |
| Filipino term | Alaala v1 (descriptions only) | 0.545 | 0.744 | 0.680 | 0.698 |
| Filipino term | Alaala v2 (+ original word) | 0.545 | 0.733 | 0.663 | 0.701 |
| Filipino term | Baseline (plain CLIP) | 0.273 | 0.394 | 0.357 | 0.408 |

## Head to head (nDCG@10): Alaala v2 (+ original word) vs baseline
Wins 17, ties 7, loses 5.

## Biggest gains vs baseline

| query | nDCG gain |
|---|---|
| halo-halo | 0.976 |
| lolo | 0.857 |
| ensemada | 0.853 |
| lola sa beach | 0.832 |
| kanin | 0.652 |
| ate | 0.563 |
| lolo smiling | 0.500 |
| tito | 0.444 |

## Where it still loses to baseline

| query | nDCG change |
|---|---|
| lumpia | -0.370 |
| fruitas | -0.252 |
| red hotdog | -0.069 |
| puto | -0.046 |
| tita | -0.001 |