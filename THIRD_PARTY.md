# 来源、致谢与许可

本工作主要是适配与复现，不重新发明上游算法。第三方项目的贡献、名称和许可不因整合而改变，也不暗示其作者认可本人的结果。

| 资源 | 用途 | 来源 |
| --- | --- | --- |
| GVHMR | 单目视频人体运动恢复 | [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) |
| GMR | 人体→机器人重定向与落地处理 | [YanjieZe/GMR](https://github.com/YanjieZe/GMR) |
| BeyondMimic / tracking 实现 | 全身动作模仿 | [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)、[所用课程分支](https://github.com/HAOTianGa03/GAOTIANHAO-G1-BeyondMimic) |
| SUZ SMP | G1 SMP 课程复现及 locomotion/steering 基础 | [SUZ-tsinghua/smp](https://github.com/SUZ-tsinghua/smp) |
| 原始 SMP / MimicKit | 原算法来源 | [xbpeng/MimicKit](https://github.com/xbpeng/MimicKit) |
| MjLab / MuJoCo / RSL-RL | 环境、足部传感/奖励、仿真与 PPO | [mujocolab/mjlab](https://github.com/mujocolab/mjlab)、[MuJoCo](https://github.com/google-deepmind/mujoco)、[RSL-RL](https://github.com/leggedrobotics/rsl_rl) |
| EngineAI 官方框架和 T800 资产 | 动作训练、机器人模型与外观 | [engineai_rl_lab](https://github.com/engineai-robotics/engineai_rl_lab)、[Native SDK](https://github.com/engineai-robotics/engineai_robotics_native_sdk)、[模型描述](https://github.com/engineai-robotics/engineai_robotics_description) |
| Isaac Lab / Isaac Sim | tracking 的环境与运行平台 | [IsaacLab](https://github.com/isaac-sim/IsaacLab)、[Isaac Sim](https://developer.nvidia.com/isaac/sim) |
| EngineAI `dance_t800` | 庆祝参考动作（复用，不是原创） | [所用版本](https://github.com/engineai-robotics/engineai_rl_lab/tree/14ec57be718586bd0ac45375aa1115bd896fbdbc)；该仓库 BSD-3-Clause 许可按原条款保留 |
| G1 Moves：`M_Move7 / Rapid Punch` | 出拳参考动作 | [Experiential Technologies / exptech](https://huggingface.co/datasets/exptech/g1-moves)，CC-BY-4.0；本次改动为 G1→T800 关节映射、肘部方向/前臂中立位与参考 FK 转换 |

T800 行走的 12 段匹配复现包及初始模型由同伴提供。在其基础上开展机器人适配、策略回放与本机转向试验；对应的服务器训练不能归为本人独立训练。

原创文档、绘图/归档工具与项目特定的辅助代码可按本仓库 MIT 许可使用。上游代码、模型、动作和机器人资产保留原许可；没有授权的完整上游源码不因放进归档就获得新许可。受限动作与人体模型不公开重新分发，私有恢复入口也不授予第三方使用权。

文中的视频是各有限仿真实验的输出。人体原始录制不复制到本公开展示库；已有旧仓库中的已授权证据保持原有来源说明。Isaac Sim 等运行库从官方渠道安装并遵守其 EULA，不在本仓库分发安装缓存。
