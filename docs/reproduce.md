# 看成果、恢复材料、重新运行

## 不训练也能查看

```bash
git clone https://github.com/jkl-ddl/humanoid-robotics-portfolio.git
cd humanoid-robotics-portfolio
python -m pip install -r requirements-showcase.txt
python scripts/build_showcase.py
```

这会从仓库内 CSV 重画曲线，不下载模型、不启动模拟器、不更改原始指标。视频已在 `media/`，可以直接下载播放。GIF 是从相应 MP4 的时间段抽取的预览，完整视频未以好看片段替换连续记录。

## 本人：完整材料恢复

公开展示库与 [私有恢复库](https://github.com/jkl-ddl/humanoid-robotics-reproduction) 配合使用。先登录本人的 GitHub 账号，再按恢复库 README 下载固定 Release。恢复包包含选定 checkpoint、匹配 prior/归一化、参考动作、必要源码适配、运行参数和原始日志；不能只拿一个 `.pt` 配最新主分支就认为能复现。

私有归档不是为了隐藏贡献，而是避免公开重新分发许可未确认的同伴材料和受限参考数据。没有该权限的人可以查看全部公开演示、曲线与适配代码，也可使用官方框架和自己有权使用的数据做新的实验。

大型模型/动作包使用版本化 Release，不塞进 Git 历史。第三方模拟器、人体模型和运行库从官方渠道按指定版本安装；它们各自的授权条件仍然适用。[GitHub Release 说明](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)

## 实际用过的三条运行路线

| 工作 | 历史运行路线 | 恢复时保持什么 |
| --- | --- | --- |
| G1 视频动作跟踪 | Linux / Docker，BeyondMimic 与 Isaac Lab；最终 CPU PhysX 恢复验证 | 原管线版本、选定模型/参考动作、固定第40帧和验证条件 |
| T800 行走/转向 | 本机 Docker Desktop，SMP / MjLab，512环境 | T800化源码、匹配prior、25关节/EE映射、目标指令与50步切换 |
| T800 庆祝/出拳 | Windows Native Isaac Sim 5.1，1024环境，seed42 | EngineAI/IsaacLab源码快照、Torch/RSL版本、动作/模型与reset缓存刷新 |

不能将本机 512/1024 环境标作统一服务器 4096 环境训练；最初的行走 warm-start 是同伴的服务器交付。

## 源码入口

- 视频动作管线：[原公开仓库](https://github.com/jkl-ddl/gvhmr-gmr-beyondmimic-reproducible)，克隆后使用 `git lfs pull` 获取其已授权证据。
- 机器人与转向关键适配：[code](../code/README.md)。这些是项目特定文件，不是完整上游 fork；完整适配源码使用恢复包。
- 固定起点动作验证：[eval_t800_tracking.py](../code/eval_t800_tracking.py)，保留失败终止，明确随机扰动关闭与延长 time-out 的条件。
- 策略状态重渲染：[render_policy_states.py](../code/render_policy_states.py)。不需要训练，但需要官方 XML/mesh/texture 和匹配关节名称。不能把重渲染称作 MuJoCo 重新执行策略的 sim2sim。
- 曲线：[build_showcase.py](../scripts/build_showcase.py)。默认从公开 CSV 重画；`--collect` 仅用于把本机已有日志/视频收集成展示文件，不触发训练。

Native tracking须从匹配的EngineAI仓库根目录启动，因为T800 USD使用相对路径。恢复库的`run_action.ps1`已处理工作目录；从其他目录直接调用脚本会在模型执行前报`USD file not found`，不应因此改资产、训练配置或终止阈值。

## 复现承诺的范围

在线归档使选定模型和相配数据不依赖旧电脑的文件路径；恢复验证应报告使用了什么平台和哪些入口。重新训练是随机过程，不能保证逐字节重建同一个 checkpoint。没有声称从公开视频即可恢复受限人体模型，也没有声称这些策略可直接上实机。
