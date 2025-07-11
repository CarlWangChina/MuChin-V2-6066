# 关于LLM微调后歌词生成效果的客观评估

## 内容概要
使用prompt输入中的原始乐段结构符号序列（记为`o_msstr`）作为主要依据，对LLM生成的每一首歌词
（记为`gen_lrc`）进行各个维度的客观评分，并给出最终的总分。

**注意：客观评估不包含任何主观评测的内容，如歌词语句是否通顺，是否符合prompt输入的主题等**

### 评估维度
按照这几个阶段分别进行评估：（维度概述 & `分值权重名`）
 - [整体表现](#整体表现的评分) & `D1W`
 - [乐段结构](#乐段结构的评分) & `D2W`
 - [字数对齐](#字数对齐的评分) & `D3W`
 - [句尾押韵](#句尾押韵的评分) & `D4W`

（D1W + D2W + D3W + D4W = 1.0）

### 满分值、各权重比及附加分
满分值`FP`预设为 ***100***，额外的押韵附加分奖励为`ES`为 ***10***;
`D1W`=***0.10*** `D2W`=***0.50*** `D3W`=***0.20*** `D4W`=***0.20***;

其他如[乐段结构](#乐段结构的评分)里的更细粒度权重划分及规则，详见各自对应部分的介绍。

## 整体表现的评分
即`gen_lrc`对原`o_msstr`要求的整体符合情况，只是一个粗略的评估过程。
我们可以先将`gen_lrc`简化回`o_msstr`的序列表示形式，记为`g_msstr`，并能很容易想到，使用
*最长连续子序列匹配*相关的算法对这两个序列比较即可。
这里我们采用*格式塔模式匹配*算法计算两个序列的整体相似度`p1_sr`，则该部分的得分：
```python
phase1score = FP * D1W * p1_sr
```

## 乐段结构的评分
**最重要的一步**，因为后面两个阶段（维度）的评估高度依赖这部分评估中的间接结果。
乐段结构包含两方面：
 1. 乐段名字、顺序和数量 （预设分值占比 ***65%***）
 2. 每个乐段内的句子行数 （预设分值占比 ***35%***）

如何将两个乐段结构序列的msstr，按具体的要求进行细分比较？

假设`o_msstr`和`gen_lrc`分别按行切割后为
| o_msstr     | gen_lrc     |
| ----------- | ----------- |
| (verse)\n   | (verse)\n   |
| cccccR\n    | cccccR\n    |
| ...         | ...         |
| ...         | (verse)\n   |
| (verse)\n   | ...         |
| ...         | ...         |
| ...         | (chorus)\n  |

有太多影响因素需要同时考虑，因此最合理的方式应该是按主次分步评估计算，我们可以先将两个msstr
中的乐段名部分提取并用符号替代，假设`o_msstr`和`gen_lrc`提取后的乐段名符号序列分别为
```
o_sncs = 'VVVVPCCVVBCC'; g_sncs = 'VVVCCCBCC'
```
则他们相互匹配的最大部分应该是：
```
- VVVVPCCVVBCC
?    --  ^-
+ VVVCCCBCC
?      ^
```
同样使用类似[整体表现](#整体表现的评分)中的序列相似度算法得出`p2_1_sr`，同时获取相互匹配的
乐段信息进行迭代计算第二小块（各乐段句子数）的得分。
而这一步就**不能**使用序列相似度的算法来计算了，考虑'3343'无论是和'3323'还是和'3393'比较，
他们的相似度都是一样的，无法体现数值上的差异，但我们可以借用类似的思路计算`p2_2_cr`，如  
'3343' 和 '3323'， p2_2_cr = (3+3+2+3)*2/(3+3+4+3 + 3+3+2+3) = 0.9167；  
'3343' 和 '3393'， p2_2_cr = (3+3+4+3)*2/(3+3+4+3 + 3+3+9+3) = 0.8387；  

$$
\displaystyle
\frac{2*\sum_{i=1}^n \min(a_i, b_i)}{\sum_{i=1}^n (a_i + b_i)}
$$

并且在这一维度中我们引入累计影响相乘的相似度:
```python
# 初始值可以为1.0，也可以为`p1_sr`, 可由最终函数的参数设置
am_sr *= p2_1_sr
phase2_1score = FP * D2W * 0.65 * am_sr
am_sr *= p2_2_cr
phase2_2score = FP * D2W * 0.35 * am_sr

phase2score = phase2_1score + phase2_2score
```

## 字数对齐的评分
沿用[乐段结构](#乐段结构的评分)中第二小块关于句子数的算法，累乘匹配行字数的相似度即可得到最终的`p3_cr`，稍微不同的点是，它由每行有效字数的相似度累乘求得。  
所以不难求出
```python
phase3score = FP * D3W * p3_cr * am_sr
```
> 当然如果想要更严格激进的评分方式，简单粗暴得使用 生成行字数完全等于对应要求行字数的行数总和比上原始要求的行数总和即可。

## 句尾押韵的评分
中文的押韵比较复杂，在中文古典文学、诗歌诗词以及现代汉语中大众普适理解上均有所不同，且对应于现代汉语拼音（该方案于1956年才开始基于普通话推广）系统也没有一个统一的官方标准或规范，但经我们调研发现，采用[中华新韵分韵表](https://baike.baidu.com/item/%E6%8A%BC%E9%9F%B5/192771#6)中的十八韵以及对应的注音，就能有比较不错的实际表现，并在其之上对"五支"部分做了进一步的细化完善。  
中文还有个特殊的点在于还需要考虑多音字的情况，所以结合现有的pypinyin（拼音注音）和jieba（分词）工具库，现在可以很容易地实现对两个字是否属于同一种韵律的准确判断。

现在我们思考几个问题：
 - 假设要求的押韵为'cRcR'，而生成的是'RcRc'，这样的押韵生成算不算？
 - 什么样的押韵是好的押韵？
 - 前述中的额外押韵分奖励该如何给出？

经过综合分析考虑，在这一步只计算数量的比率，不严格要求位置对齐，算法类似：
```python
# `rc_ing` is the rhymed line count in generated lrc;
# `rc_ino` is same in original msstr
p4_rr = 2 * min(rc_ino, rc_ing) / (rc_ino + rc_ing)
```

在额外奖励分阶段，满足押韵的行数占比在60%~80%之间的，则视为好押韵并给予全部的额外奖励分，否则如果是严格（即考虑位置）满足输入要求的，则给予额外分的一半奖励值。

```python
phase4score = FP * D4W * p4_rr * am_sr
# `valid_line_count` is matched line count in phase2
if 0.6 <= rc_ing/valid_line_count <= 0.8:
    extra_score = ES * acmp_sr
# `frmc` is full rhyme match count, pos is considered
elif frmc == rc_ino == rc_ing and frmc > 0:
    extra_score = ES * 0.5 * acmp_sr
```

## 总分
最后，我们将各个阶段的得分相加，得到最终的总分：
```python
totalscore = sum((phase1score, phase2score, phase3score, 
    phase4score, extra_score))
```