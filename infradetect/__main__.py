"""命令行入口：`python -m infradetect` 等价于调用 pipeline.main。

这里只转发参数解析，不复制 pipeline 的校验或模型逻辑，避免两个入口行为分叉。
"""

from .pipeline import main


if __name__ == "__main__":
    # 直接执行包时才启动；其他模块 import infradetect 不会触发推理。
    main()
