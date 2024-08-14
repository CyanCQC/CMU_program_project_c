# README
This is the **project C** of *2024 Pittsburgh Summer Visit Program*, which is about transformer tasks like *captioning* and *classification*. To run this project, you can directly install pytorch and some other simple dependencies since there are not so many packages used.

Due to the small scale of training set, the model performs humorously bad on valid sets in captioning, and gains 0.66 accuracy in classification tasks. This is more a example than an appliable model (After all, it's no more than a final project of a summer camp).

We use `-inf` in `attn_mask` instead of `-1e10`, which LongVanTH used in the origin code. This is theoritically right, but perform not so well in lr of `1e-3`. `1e-3` is too large for the model and makes the 4-heads-6-layers transformer fail to converge.

The quality of datasets of task 1 isn't so high. There are plenty of unexpected <UNK> in the GT caption. We update the model to enable it the ability to learn <UNK> but not generate it. However, influenced by the exceedingly short token list (1004 words), the model is more like a baby who is recently start to learn to speak. That is funny, we hope you (probably summer camp fellows) enjoy it.
