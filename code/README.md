# 关键代码

这里保留项目特定的适配/评估工具，不复制整个上游算法库：

- `t800_asset.py`：25关节顺序、动作尺度、执行器与 Native SDK XML 接入。
- `steering_env_cfg.py`：T800 指令任务、官方足底 site/contact 传感与 MjLab 奖励项。
- `eval_t800_steering.py`：五段固定指令、seed42、平滑切换；每步显式刷新并核对actor指令，在首次终止前记录真实接触/速度/姿态与joint/root状态，拒绝将auto-reset后的姿态拼接进视频。GPU物理不承诺跨机器逐字节确定性。
- `steering_quality_task.py`：独立质量调参overlay，复用官方姿态/足底高度/肩部posture公式，不改动力学或失败条件。具体选中配置必须取自run参数，不能把这里的起始值当作最终结果。
- `retarget_g1_to_t800.py`：已有 G1 CSV→T800 关节映射；不是接触约束 IK。
- `eval_t800_tracking.py`：保留失败终止的完整固定起点动作验证，处理包装器 reset 后的参考缓存。
- `render_policy_states.py`：只渲染已记录的策略姿态与地面，不执行或替代策略。

这些文件对应的完整依赖与修改后的工作源码在本人恢复归档中。单独复制转向配置到未经适配的 SUZ 主分支不能保证运行：它依赖本地 T800 化的基础环境、特征与事件实现。不要把这一目录称作独立可训练框架。

`render_policy_states.py` 可在安装 MuJoCo、NumPy、ImageIO、Pillow 后独立使用；XML 从官方 Native SDK 获取。示例：

```bash
python code/render_policy_states.py --rollout policy_states.npz \
  --xml /path/to/serial_t800.xml --output replay.mp4 --label "T800 learned policy"
```

回放中的 root quaternion 为 `wxyz`，关节名称必须与 XML 一致。展示外观变化不改变训练动力学，也不证明 sim2sim。
