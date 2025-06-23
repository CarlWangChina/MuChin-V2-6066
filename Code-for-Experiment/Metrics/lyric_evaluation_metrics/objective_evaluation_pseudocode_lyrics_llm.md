# Objective Evaluation of LLM Fine-tuned Lyric Generation Effectiveness

## Content Overview
Using the original musical segment structure symbol sequence (denoted as `o_msstr`) from the prompt input as the primary basis, we conduct objective scoring across various dimensions for each generated lyric (denoted as `gen_lrc`) by LLMs, and provide a final total score.

**Note: Objective evaluation does not include any subjective assessment content, such as whether the lyric sentences are fluent or whether they conform to the theme specified in the prompt input.**

### Evaluation Dimensions
Evaluation is conducted according to these stages: (Dimension Overview & `Score Weight Name`)
 - [Overall Performance](#overall-performance-scoring) & `D1W`
 - [Musical Segment Structure](#musical-segment-structure-scoring) & `D2W`
 - [Character Count Alignment](#character-count-alignment-scoring) & `D3W`
 - [End-of-Line Rhyming](#end-of-line-rhyming-scoring) & `D4W`

(D1W + D2W + D3W + D4W = 1.0)

### Full Score Value, Weight Ratios, and Bonus Points
The full score value `FP` is preset to ***100***, with an additional rhyming bonus reward `ES` of ***10***;
`D1W`=***0.10*** `D2W`=***0.50*** `D3W`=***0.20*** `D4W`=***0.20***;

For other finer-grained weight divisions and rules in [Musical Segment Structure](#musical-segment-structure-scoring), please refer to the detailed descriptions in their respective corresponding sections.

## Overall Performance Scoring
This refers to the overall compliance of `gen_lrc` with the requirements of the original `o_msstr`, which is only a rough assessment process.
We can first simplify `gen_lrc` back to the sequence representation form of `o_msstr`, denoted as `g_msstr`, and it's easy to think of using *longest common subsequence matching* related algorithms to compare these two sequences.
Here we adopt the *Gestalt pattern matching* algorithm to calculate the overall similarity `p1_sr` between the two sequences, then the score for this part is:
```python
phase1score = FP * D1W * p1_sr
```

## Musical Segment Structure Scoring
**The most important step**, because the evaluation of the subsequent two stages (dimensions) highly depends on the indirect results from this part of the evaluation.
Musical segment structure includes two aspects:
 1. Musical segment names, order, and quantity (preset score proportion ***65%***)
 2. Number of sentence lines within each musical segment (preset score proportion ***35%***)

How to subdivide and compare two musical segment structure sequences of msstr according to specific requirements?

Assuming `o_msstr` and `gen_lrc` are split by lines as:
| o_msstr     | gen_lrc     |
| ----------- | ----------- |
| (verse)\n   | (verse)\n   |
| cccccR\n    | cccccR\n    |
| ...         | ...         |
| ...         | (verse)\n   |
| (verse)\n   | ...         |
| ...         | ...         |
| ...         | (chorus)\n  |

There are too many influencing factors to consider simultaneously, so the most reasonable approach should be to evaluate and calculate step by step according to primary and secondary priorities. We can first extract the musical segment name parts from the two msstr and substitute them with symbols. Assuming the extracted musical segment name symbol sequences of `o_msstr` and `gen_lrc` are respectively:
```
o_sncs = 'VVVVPCCVVBCC'; g_sncs = 'VVVCCCBCC'
```
Then their maximum matching part should be:
```
- VVVVPCCVVBCC
?    --  ^-
+ VVVCCCBCC
?      ^
```
Similarly, using sequence similarity algorithms similar to [Overall Performance](#overall-performance-scoring), we obtain `p2_1_sr`, and simultaneously acquire the mutually matching musical segment information for iterative calculation of the second block's score (number of sentences in each musical segment).
However, this step **cannot** use sequence similarity algorithms for calculation. Consider that whether '3343' is compared with '3323' or '3393', their similarity is the same, unable to reflect numerical differences. But we can borrow similar ideas to calculate `p2_2_cr`, such as:
'3343' and '3323', p2_2_cr = (3+3+2+3)*2/(3+3+4+3 + 3+3+2+3) = 0.9167;
'3343' and '3393', p2_2_cr = (3+3+4+3)*2/(3+3+4+3 + 3+3+9+3) = 0.8387;

$$
\displaystyle
\frac{2*\sum_{i=1}^n \min(a_i, b_i)}{\sum_{i=1}^n (a_i + b_i)}
$$

And in this dimension, we introduce cumulative impact multiplied similarity:
```python
# Initial value can be 1.0, or `p1_sr`, can be set by final function parameters
am_sr *= p2_1_sr
phase2_1score = FP * D2W * 0.65 * am_sr
am_sr *= p2_2_cr
phase2_2score = FP * D2W * 0.35 * am_sr

phase2score = phase2_1score + phase2_2score
```

## Character Count Alignment Scoring
Following the algorithm in the second block of [Musical Segment Structure](#musical-segment-structure-scoring) regarding sentence counts, we accumulate and multiply the similarity of character counts in matching lines to obtain the final `p3_cr`. The slightly different point is that it is calculated by accumulating and multiplying the similarity of effective character counts in each line.
So it's not difficult to calculate:
```python
phase3score = FP * D3W * p3_cr * am_sr
```
> Of course, if you want a more strict and aggressive scoring method, you can simply and roughly use the ratio of the total number of lines where the generated character count exactly equals the required character count for the corresponding line to the total number of lines in the original requirements.

## End-of-Line Rhyming Scoring
Chinese rhyming is quite complex. In Chinese classical literature, poetry, and modern Chinese, there are different understandings among the general public, and there is no unified official standard or specification corresponding to the modern Chinese pinyin system (this scheme was only promoted based on Mandarin starting from 1956). However, through our research, we found that adopting the eighteen rhymes and corresponding phonetic notations in the [Chinese New Rhyme Classification Table](https://baike.baidu.com/item/%E6%8A%BC%E9%9F%B5/192771#6) can achieve quite good practical performance, and we further refined and improved the "Wu Zhi" part on top of it.
Chinese also has a special point in that we need to consider the case of polyphonic characters, so combining existing pypinyin (pinyin annotation) and jieba (word segmentation) tool libraries, we can now easily achieve accurate judgment of whether two characters belong to the same type of rhyme.

Now let's think about several questions:
 - Assuming the required rhyme is 'cRcR', but the generated is 'RcRc', does such rhyme generation count?
 - What kind of rhyming is good rhyming?
 - How should the additional rhyming bonus mentioned earlier be given?

After comprehensive analysis and consideration, in this step we only calculate the ratio of quantities, without strictly requiring position alignment. The algorithm is similar to:
```python
# `rc_ing` is the rhymed line count in generated lrc;
# `rc_ino` is same in original msstr
p4_rr = 2 * min(rc_ino, rc_ing) / (rc_ino + rc_ing)
```

In the additional bonus stage, if the proportion of lines satisfying rhyming is between 60%~80%, it is considered good rhyming and given the full additional bonus, otherwise if it strictly (i.e., considering position) satisfies the input requirements, half of the additional bonus value is given.

```python
phase4score = FP * D4W * p4_rr * am_sr
# `valid_line_count` is matched line count in phase2
if 0.6 <= rc_ing/valid_line_count <= 0.8:
    extra_score = ES * acmp_sr
# `frmc` is full rhyme match count, pos is considered
elif frmc == rc_ino == rc_ing and frmc > 0:
    extra_score = ES * 0.5 * acmp_sr
```

## Total Score
Finally, we add up the scores from each stage to get the final total score:
```python
totalscore = sum((phase1score, phase2score, phase3score, 
    phase4score, extra_score))
```

This evaluation framework is designed to assess the intent fidelity of LLM-generated lyrics within the context of the Amateur-Professional Semantic Divide research, providing objective metrics for the MuChin project's targeted training experiments.
