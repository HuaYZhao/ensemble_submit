实验有两个分支：

该分支为实验adafactor优化器的分支。
该优化器可以在不降低模型性能的前提下，减少参数量。相较adam的参数量mxn减少至m+n。
xxlarge可以跑起batch_size 48 


all_ways分支：
1、强化学习
2、时间卷积的协同注意力
3、oom实验
由于xxlarge内存基本占满，只有强化学习能跑。

可运行的colab：
https://drive.google.com/open?id=120FySzVwbVmKGcBmIrTWn6FBN8hdYttS