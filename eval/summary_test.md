# Alaala eval (test set)

14 labeled queries (9 Filipino term, 5 English); 0 marked as no matches and skipped.

## Overall

| method | Hit@1 | Recall@10 | nDCG@10 | MRR |
|---|---|---|---|---|
| Baseline (plain CLIP) | 0.429 | 0.434 | 0.452 | 0.545 |
| Alaala v1 (descriptions only) | 0.500 | 0.585 | 0.553 | 0.632 |
| Alaala v2 (+ original word) | 0.500 | 0.585 | 0.574 | 0.677 |

## By query type

| type | method | Hit@1 | Recall@10 | nDCG@10 | MRR |
|---|---|---|---|---|---|
| English | Alaala v1 (descriptions only) | 0.600 | 0.571 | 0.627 | 0.729 |
| English | Alaala v2 (+ original word) | 0.600 | 0.571 | 0.627 | 0.729 |
| English | Baseline (plain CLIP) | 0.600 | 0.571 | 0.627 | 0.729 |
| Filipino term | Alaala v1 (descriptions only) | 0.444 | 0.592 | 0.512 | 0.578 |
| Filipino term | Alaala v2 (+ original word) | 0.444 | 0.592 | 0.545 | 0.648 |
| Filipino term | Baseline (plain CLIP) | 0.333 | 0.357 | 0.354 | 0.443 |

## Head to head (nDCG@10): Alaala v2 (+ original word) vs baseline
Wins 5, ties 8, loses 1.

## Biggest gains vs baseline

| query | nDCG gain |
|---|---|
| pamilya dinner | 0.735 |
| barkada | 0.437 |
| lolo at the table | 0.405 |
| lolo and lola | 0.367 |
| tita at the beach | 0.142 |
| ate and kuya | 0.000 |
| dessert | 0.000 |
| group photo | 0.000 |

## Where it still loses to baseline

| query | nDCG change |
|---|---|
| pancit | -0.369 |
| ate and kuya | 0.000 |
| dessert | 0.000 |
| group photo | 0.000 |
| kids playing | 0.000 |