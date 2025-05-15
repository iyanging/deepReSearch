# DeepNote

> Ref: <https://arxiv.org/abs/2410.08821>

![Overview of DeepNote](https://arxiv.org/html/2410.08821v2/x2.png)

## Note Initialization

### 做法

* q(0) = 用户原始查询
* 文章集合 P(0) = retrieve( q(0) )，搜索知识库得到 top-k 文章
* N(0) = n(0) = LLM( Instruct`Init, q(0) + P(0) )

> retrieve 内部包含了 rerank

### 原因

按照预设分类 / 领域总结出的note，其中所蕴含的知识往往与解答用户问题所需难以匹配。

所以充分使能LLM的推理和决策能力，仅提供最高层次的目标，以促进其灵活且全面地收集可以支持回答 / 推理q0的知识。

> 改写了部分语句结构的原文:
>
> Since the system fails to foresee the characteristics and aspects of the retrieved knowledge, a fine-grained (strictly summarized from predefined aspects or domains) note construction approach often leads to misalignment between the collected knowledge and the actual relevant information.
>
> Therefore, we delegate reasoning and decision-making entirely to the LLM, providing only the highest-level objective to facilitate its flexible and comprehensive collection of knowledge that supports answering or reasoning about the q(0).

### 疑问

* 使用用户原始问题q0做 retrieve，通常效果是否很差？如果效果很差，是否会对整体流程/最终答案有恶化影响？

  使用q0做 retrieve，其本身召回的数据“相似性”能得到保证，但语义“相关性”会比较差。

  通常使用以下方式**缓解**：
  * 使用 ES analyzer + 加权 进行优化，例如 simple_bigram_analyzer
  * 较大的top-k，然后做rerank

* DeepNote的实现中，使用手写逻辑做rerank，效果是否很差？

  很差，rerank这一步应该使用语义相关性在已缩小范围的候选数据中进行重排。
  参考Cross-Encoder

## Note-Centric Adaptive Retrieval

### 做法

给定总迭代轮数k，当前迭代轮数t（t >= 1），重复以下步骤：

* Query Refinement
  * 新查询 q(t) = LLM( Instruct`QR, q(0) + N(t-1) + Q(t-1) )，要求LLM生成多个（DeepNote实现为2）不与Q(t-1)重复的新查询
  * 问题集合 Q(t) = { *Q(t-1) ... q(t) }

* Knowledge Accumulation
  * 文章集合 P(t) = retrieve( q(0) + q(t) )，文章需要重新召回进行覆盖，这同时也解决了与之前文章重复的问题
  * 当前Note n(t) = LLM( Instruct`KA, q(0) + P(t) + N(t-1) )，要求LLM根据更新后的文章集合，结合之前的Note，重新生成新Note

* Adaptive Retrieval Decision
  * 是否新Note为最佳 V(t) = LLM( Instruct`ARD, q(0) + N(t-1) + n(t) )，要求LLM比较新Note和上一轮迭代的最佳Note，V(t) = true if n(t) is best
  * N(t) = n(t) if V(t) is True else N(t-1)
  * 如果 { V(1) ... V(t) }.count(False) 大于预设阈值，则终止迭代
  > 论文把“终止迭代”这一步放在了【Note-Informed Answer Generation】

## Note-Informed Answer Generation

α = LLM( Instruct`ANS, q(0) + N(t) )
