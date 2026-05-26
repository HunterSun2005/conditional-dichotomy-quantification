# 错误分析

下表使用轻量 bag-of-words proxy 识别 DCF 失败样本，供人工复查使用。它不能替代最终的 DoGE/Opposite-Score 评测，但可以在 C 的 checkpoint 可用之前，为成员 A 提供一组稳定的困难样本。

| # | 数据集 | 错误类型 | 失败原因 | 可能的改进方向 |
| ---: | --- | --- | --- | --- |
| 1 | causal_reasoning | positive_negative_too_close（正负样本过近） | pos-neg 距离 0.203372 没有同时大于两条 neutral 距离 (0.230351, 0.322644). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 2 | causal_reasoning | positive_negative_too_close（正负样本过近） | pos-neg 距离 0.148165 没有同时大于两条 neutral 距离 (0.257219, 0.224909). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 3 | causal_reasoning | positive_negative_too_close（正负样本过近） | pos-neg 距离 0.118083 没有同时大于两条 neutral 距离 (0.183906, 0.224909). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 4 | causal_reasoning | positive_negative_too_close（正负样本过近） | pos-neg 距离 0.075527 没有同时大于两条 neutral 距离 (0.132773, 0.180187). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 5 | causal_reasoning | positive_negative_too_close（正负样本过近） | pos-neg 距离 0.081063 没有同时大于两条 neutral 距离 (0.180108, 0.161856). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 6 | causal_reasoning | positive_negative_too_close（正负样本过近） | pos-neg 距离 0.085591 没有同时大于两条 neutral 距离 (0.108504, 0.182943). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 7 | causal_reasoning | positive_negative_too_close（正负样本过近） | pos-neg 距离 0.092735 没有同时大于两条 neutral 距离 (0.132773, 0.186843). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 8 | causal_reasoning | positive_negative_too_close（正负样本过近） | pos-neg 距离 0.102607 没有同时大于两条 neutral 距离 (0.125252, 0.195383). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 9 | debate | neutral_lexically_close_to_positive（neutral 在词面上接近 positive） | pos-neg 距离 0.356321 没有同时大于两条 neutral 距离 (0.254566, 0.527134). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 10 | debate | positive_negative_too_close（正负样本过近） | pos-neg 距离 0.440741 没有同时大于两条 neutral 距离 (0.455534, 0.605945). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 11 | debate | neutral_lexically_close_to_positive（neutral 在词面上接近 positive） | pos-neg 距离 0.444063 没有同时大于两条 neutral 距离 (0.35746, 0.607768). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 12 | debate | positive_negative_too_close（正负样本过近） | pos-neg 距离 0.449518 没有同时大于两条 neutral 距离 (0.455534, 0.604372). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 13 | debate | neutral_lexically_close_to_positive（neutral 在词面上接近 positive） | pos-neg 距离 0.469068 没有同时大于两条 neutral 距离 (0.455534, 0.618422). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 14 | debate | positive_negative_too_close（正负样本过近） | pos-neg 距离 0.454545 没有同时大于两条 neutral 距离 (0.455534, 0.600725). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 15 | debate | neutral_lexically_close_to_positive（neutral 在词面上接近 positive） | pos-neg 距离 0.403417 没有同时大于两条 neutral 距离 (0.382787, 0.545141). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 16 | debate | neutral_lexically_close_to_positive（neutral 在词面上接近 positive） | pos-neg 距离 0.477767 没有同时大于两条 neutral 距离 (0.455534, 0.617724). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 17 | defeasible_nli | positive_negative_too_close（正负样本过近） | pos-neg 距离 0.087097 没有同时大于两条 neutral 距离 (0.153585, 0.207594). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 18 | defeasible_nli | positive_negative_too_close（正负样本过近） | pos-neg 距离 0.087097 没有同时大于两条 neutral 距离 (0.173984, 0.123583). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 19 | defeasible_nli | neutral_lexically_close_to_positive（neutral 在词面上接近 positive） | pos-neg 距离 0.205703 没有同时大于两条 neutral 距离 (0.0995, 0.292043). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 20 | defeasible_nli | positive_negative_too_close（正负样本过近） | pos-neg 距离 0.030254 没有同时大于两条 neutral 距离 (0.107548, 0.116518). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 21 | defeasible_nli | neutral_lexically_close_to_negative（neutral 在词面上接近 negative） | pos-neg 距离 0.215285 没有同时大于两条 neutral 距离 (0.297886, 0.057143). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 22 | defeasible_nli | neutral_lexically_close_to_positive（neutral 在词面上接近 positive） | pos-neg 距离 0.175137 没有同时大于两条 neutral 距离 (0.057143, 0.252468). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 23 | defeasible_nli | neutral_lexically_close_to_negative（neutral 在词面上接近 negative） | pos-neg 距离 0.155838 没有同时大于两条 neutral 距离 (0.229949, 0.04467). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |
| 24 | defeasible_nli | neutral_lexically_close_to_negative（neutral 在词面上接近 negative） | pos-neg 距离 0.151515 没有同时大于两条 neutral 距离 (0.223794, 0.04467). | Axis-aware loss 可能有帮助：它可以在同一 context 下显式拉开 positive-negative，同时约束 neutral 更接近中间区域。 |

## 详细样例

### 样例 1: causal_reasoning:dev:c6:c6_s1:c6_d1:c6_s1_n1_p3

- 上下文： Cause: A business finds that its revenue was lower than anticipated for the year. Effect: Hundreds of workers are laid off.
- Positive： A company that is not doing well financially will be forced to lay off workers.
- Negative： Leadership of the company vows that there will not be any layoffs.
- Neutral： A company should strive to innovate to reduce the need to lay off employees.
- 错误类型： positive_negative_too_close（正负样本过近）

### 样例 2: causal_reasoning:dev:c2:c2_s1:c2_d1:c2_d1_n5_p3

- 上下文： Cause: A CEO charges one of their managers to oversee business development. Effect: The manager replaces the CEO.
- Positive： The manager does an excellent job and makes the company millions of dollars.
- Negative： The manager fails and the development doesn’t happen.
- Neutral： Failure often happens when preparation is lacking.
- 错误类型： positive_negative_too_close（正负样本过近）

### 样例 3: causal_reasoning:dev:c2:c2_s0:c2_d1:c2_d1_n5_p3

- 上下文： Cause: A CEO charges one of their managers to oversee business development. Effect: The manager replaces the CEO.
- Positive： The manager is very hardworking and experienced
- Negative： The manager fails and the development doesn’t happen.
- Neutral： Failure often happens when preparation is lacking.
- 错误类型： positive_negative_too_close（正负样本过近）

### 样例 4: causal_reasoning:dev:c5:c5_s2:c5_d3:c5_s2_n2_p3

- 上下文： Cause: A business decides to offer new services that cater to younger people. Effect: Months later the business is doing better than ever.
- Positive： The new services attract a lot of younger people.
- Negative： Young people got bored of the new services.
- Neutral： Social media platforms attract younger people with engaging content.
- 错误类型： positive_negative_too_close（正负样本过近）

### 样例 5: causal_reasoning:dev:c9:c9_s1:c9_d1:c9_s1_n1_p3

- 上下文： Cause: A coffee shop hires way too many baristas. Effect: The coffee shop has profits that are too low to stay open.
- Positive： It keeps all the baristas on the employment roster.
- Negative： It dismisses all the extra baristas.
- Neutral： It keeps evolving, challenging our understanding of reality.
- 错误类型： positive_negative_too_close（正负样本过近）

### 样例 6: causal_reasoning:dev:c14:c14_s1:c14_d1:c14_s1_n1_p3

- 上下文： Cause: A conservation group starts to repair the habitat location of an endangered species. Effect: The endangered species is no longer endangered.
- Positive： It is possible to repair the habitat of the endangered species.
- Negative： Despite the group's efforts, the habitat of the endangered species has been completely destroyed.
- Neutral： Repairing habitats is crucial to protect endangered species from extinction.
- 错误类型： positive_negative_too_close（正负样本过近）

### 样例 7: causal_reasoning:dev:c5:c5_s2:c5_d2:c5_s2_n2_p3

- 上下文： Cause: A business decides to offer new services that cater to younger people. Effect: Months later the business is doing better than ever.
- Positive： The new services attract a lot of younger people.
- Negative： Younger people complain about the quality of the services received.
- Neutral： Social media platforms attract younger people with engaging content.
- 错误类型： positive_negative_too_close（正负样本过近）

### 样例 8: causal_reasoning:dev:c10:c10_s2:c10_d2:c10_d2_n6_p3

- 上下文： Cause: A company a company institute's new productivity protocols. Effect: All of their employees increase their productivity by 5%.
- Positive： The new productivity protocols are humane.
- Negative： The new productivity protocols are not humane and intuitive so employees are reluctant but forced to abide.
- Neutral： The new productivity protocols are essential for maximizing efficiency and encouraging innovation in today's competitive environment.
- 错误类型： positive_negative_too_close（正负样本过近）

### 样例 9: debate:dev:c3:c3_s1:c3_d1:c3_s1_n1_p3

- 上下文： Claim: It is crucial to explore the universe.
- Positive： It is a moral imperative to constantly seek to expand the boundaries of knowledge
- Negative： The cost of space exploration exceeds the positive benefits
- Neutral： Expanding knowledge to everyone is a moral imperative for fostering a more informed society.
- 错误类型： neutral_lexically_close_to_positive（neutral 在词面上接近 positive）

### 样例 10: debate:dev:c4:c4_s0:c4_d20:c4_s0_n0_p3

- 上下文： Claim: Gay couples should be allowed to marry.
- Positive： Marriage provides both physical and psychological health benefits, and banning gay marriage increases rates of psychological disorders.
- Negative： Those against gay marriage are called names if gay marriage is legalized.
- Neutral： Banning plastic bags increases environmental sustainability and provides a significant decrease in pollution rates worldwide.
- 错误类型： positive_negative_too_close（正负样本过近）

### 样例 11: debate:dev:c4:c4_s7:c4_d11:c4_s7_n7_p3

- 上下文： Claim: Gay couples should be allowed to marry.
- Positive： Denying some people the option to marry is discriminatory and creates a second class of citizens.
- Negative： The institution of marriage has traditionally been defined as being between a man and a woman.
- Neutral： Some people believe the option is to prioritize renewable energy over fossil fuels for sustainability.
- 错误类型： neutral_lexically_close_to_positive（neutral 在词面上接近 positive）

### 样例 12: debate:dev:c4:c4_s0:c4_d22:c4_s0_n0_p3

- 上下文： Claim: Gay couples should be allowed to marry.
- Positive： Marriage provides both physical and psychological health benefits, and banning gay marriage increases rates of psychological disorders.
- Negative： There are many types of marriage which are innaporpoate and dangerous which may be allowed if gay marriage is legalized.
- Neutral： Banning plastic bags increases environmental sustainability and provides a significant decrease in pollution rates worldwide.
- 错误类型： positive_negative_too_close（正负样本过近）

### 样例 13: debate:dev:c4:c4_s0:c4_d15:c4_s0_n0_p3

- 上下文： Claim: Gay couples should be allowed to marry.
- Positive： Marriage provides both physical and psychological health benefits, and banning gay marriage increases rates of psychological disorders.
- Negative： Gay marriage is contrary to the word of God and is incompatible with the beliefs, sacred texts, and traditions of many religious groups.
- Neutral： Banning plastic bags increases environmental sustainability and provides a significant decrease in pollution rates worldwide.
- 错误类型： neutral_lexically_close_to_positive（neutral 在词面上接近 positive）

### 样例 14: debate:dev:c4:c4_s0:c4_d8:c4_s0_n0_p3

- 上下文： Claim: Gay couples should be allowed to marry.
- Positive： Marriage provides both physical and psychological health benefits, and banning gay marriage increases rates of psychological disorders.
- Negative： If gay couples don't get married it means they lose benefits through the legalization of gay marriage.
- Neutral： Banning plastic bags increases environmental sustainability and provides a significant decrease in pollution rates worldwide.
- 错误类型： positive_negative_too_close（正负样本过近）

### 样例 15: debate:dev:c2:c2_s1:c2_d1:c2_s1_n1_p3

- 上下文： Claim: Goal line technology has to be adopted in football.
- Positive： Technology relies of facts and figures without the corrupting influence of opinions or morals.
- Negative： The cost would not match FIFA's aim of opening football to the world
- Neutral： Technology's corrupting influence often relies on facts being manipulated or distorted for personal gain.
- 错误类型： neutral_lexically_close_to_positive（neutral 在词面上接近 positive）

### 样例 16: debate:dev:c4:c4_s0:c4_d13:c4_s0_n0_p3

- 上下文： Claim: Gay couples should be allowed to marry.
- Positive： Marriage provides both physical and psychological health benefits, and banning gay marriage increases rates of psychological disorders.
- Negative： The institution of marriage has been weakened in recent years and allowing gay marriage will only further damage an already crumbling institution.
- Neutral： Banning plastic bags increases environmental sustainability and provides a significant decrease in pollution rates worldwide.
- 错误类型： neutral_lexically_close_to_positive（neutral 在词面上接近 positive）

### 样例 17: defeasible_nli:dev:c3:c3_s0:c3_d0:c3_s0_n0_p3

- 上下文： Premise: 4 children and 1 adult look at an armadillo on a grassy hill with 2 trees. Hypothesis: There are people outside.
- Positive： The people stand at the bottom of the grassy hill and look at the armadillo.
- Negative： The people look out the window at the armadilo.
- Neutral： Stand at bottom, look at armadillo, and appreciate the armadillo's unique adaptations to survive.
- 错误类型： positive_negative_too_close（正负样本过近）

### 样例 18: defeasible_nli:dev:c3:c3_s0:c3_d0:c3_d0_n4_p3

- 上下文： Premise: 4 children and 1 adult look at an armadillo on a grassy hill with 2 trees. Hypothesis: There are people outside.
- Positive： The people stand at the bottom of the grassy hill and look at the armadillo.
- Negative： The people look out the window at the armadilo.
- Neutral： The people gazed through the window, seeing endless possibilities.
- 错误类型： positive_negative_too_close（正负样本过近）

### 样例 19: defeasible_nli:dev:c3:c3_s1:c3_d0:c3_s1_n1_p3

- 上下文： Premise: 4 children and 1 adult look at an armadillo on a grassy hill with 2 trees. Hypothesis: There are people outside.
- Positive： The armadillo can wander freely for miles.
- Negative： The people look out the window at the armadilo.
- Neutral： One can wander for miles exploring nature.
- 错误类型： neutral_lexically_close_to_positive（neutral 在词面上接近 positive）

### 样例 20: defeasible_nli:dev:c12:c12_s1:c12_d1:c12_s1_n1_p3

- 上下文： Premise: A boy making an "oh!" face in front of a bay. Hypothesis: A child is surprised by the ships.
- Positive： A child is surprised by the oceanliners.
- Negative： A child is surprised by the swooping gulls.
- Neutral： The oceanliners' size always surprised first-timers.
- 错误类型： positive_negative_too_close（正负样本过近）

### 样例 21: defeasible_nli:dev:c3:c3_s0:c3_d2:c3_d2_n6_p3

- 上下文： Premise: 4 children and 1 adult look at an armadillo on a grassy hill with 2 trees. Hypothesis: There are people outside.
- Positive： The people stand at the bottom of the grassy hill and look at the armadillo.
- Negative： The people are standing on some carpet.
- Neutral： People standing on carpet are often comfortable.
- 错误类型： neutral_lexically_close_to_negative（neutral 在词面上接近 negative）

### 样例 22: defeasible_nli:dev:c3:c3_s2:c3_d0:c3_s2_n2_p3

- 上下文： Premise: 4 children and 1 adult look at an armadillo on a grassy hill with 2 trees. Hypothesis: There are people outside.
- Positive： The people are standing on some grass.
- Negative： The people look out the window at the armadilo.
- Neutral： People are standing on grass, enjoying nature.
- 错误类型： neutral_lexically_close_to_positive（neutral 在词面上接近 positive）

### 样例 23: defeasible_nli:dev:c3:c3_s1:c3_d1:c3_d1_n5_p3

- 上下文： Premise: 4 children and 1 adult look at an armadillo on a grassy hill with 2 trees. Hypothesis: There are people outside.
- Positive： The armadillo can wander freely for miles.
- Negative： The armadillo is behind a glass wall.
- Neutral： A glass wall is behind the main wall.
- 错误类型： neutral_lexically_close_to_negative（neutral 在词面上接近 negative）

### 样例 24: defeasible_nli:dev:c3:c3_s3:c3_d1:c3_d1_n5_p3

- 上下文： Premise: 4 children and 1 adult look at an armadillo on a grassy hill with 2 trees. Hypothesis: There are people outside.
- Positive： The armadillo begins to climb the tree.
- Negative： The armadillo is behind a glass wall.
- Neutral： A glass wall is behind the main wall.
- 错误类型： neutral_lexically_close_to_negative（neutral 在词面上接近 negative）

