"""交付包的场景级接口包。

pipeline 负责二维推理调度；inference/ 负责模型和滑窗，evidence/ 负责证据
导出、对照与复核排序，geometry/ 负责点云与跨照片核验，tbbr_review 提供
本机人工复核界面。
导入本包不会自动加载模型或运行任务；请经 __main__ 或各模块 main 调用。
"""
