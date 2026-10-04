# 训练配置、命令与证据

这是四项仿真实践的训练索引，不是新算法或多种子基准测试。数据处理、机器人适配、
策略训练和实际验证分别留证；原始TensorBoard/run参数及checkpoint进入私有恢复归档。
公开CSV是直接提取的标量，不是手绘曲线。读者可查看完整公开记录，但受限输入及同伴
模型不会为了“一键运行”而未经许可公开。

## 实际训练覆盖

| 子任务 | 实际环境 / seed | 选定模型 | 公开曲线覆盖 | 策略验证 |
| --- | --- | --- | --- | --- |
| G1羽毛球风格跟踪 | 末期256环境 / 42 | 4493 | 3994–4493，最后500轮 | 第40帧起，236帧 / 4.72秒 |
| T800行走与自然转向 | 本机Docker，512环境 / 42 | 36991 | 所选续训链6900次更新，6892个唯一标签 | 固定直行500步；逆/顺时针各1000步 |
| T800庆祝 | Windows Native Isaac，1024环境 / 42 | 19999 | 0–19999，共20,000轮 | 第0帧起，完整1054帧 / 21.08秒 |
| T800挑衅 | Windows Native Isaac，1024环境 / 42 | 7996 | 完整状态链8000次更新，7997个唯一标签 | 第0帧起，原1190帧 / 23.8秒 |

转向是同伴行走基线上的本机warm-start，不能把初始服务器训练归为本人从零完成。
续训接缝会重复迭代标签；CSV保留后一个run的同标签记录，平滑线不跨配置边界。
失败或主动停止的分支不混入最终训练链。奖励权重变了，边界前后的奖励绝对值不能
直接当作性能提升；本次验收不是全部随机起点的收敛证明。

## 当前配置与修改

选定版本的任务ID、关节/EE契约、模型和动作哈希、奖励参数及训练链均在
[selected_models.json](../evidence/selected_models.json)。它只公开元数据，不包含模型或动作。

- **转向**：25关节、89维可部署actor；正确walk12的base/肘EE契约；upright=-0.30、
  肩部posture=0.003、task×SMP权重2.0、速度误差scale=2、SMP scale=6、速度/朝向75/25、
  slip=-0.10、foot-site目标0.16m。复用现有SUZ/MjLab公式；物理、PD和失败条件不变。
  site接触时本身约高0.064m，目标0.16m不是足底抬高16cm。
- **庆祝**：官方`dance_t800`，`Tracking-Flat-T800-Wo-State-Estimation-v0`；没有自己编动作。
  模型训练20k；整段验证保留失败项，修复的是reset后参考缓存的刷新顺序。
- **挑衅**：现有`M_Move7 / Rapid Punch`重映射到T800；最终使用官方完整状态任务
  `Tracking-Flat-T800-v0`，actor140/critic275，action-rate=-0.03，global anchor权重2.0、std0.45。
  原始23.8秒参考和失败阈值未放宽。
- **羽毛球**：复用GVHMR/GMR/BeyondMimic；处理参考落地并从既有run续训。
  末期日志和参数记录seed42、256环境、每500轮保存；这里不把缺失的早期曲线补画出来。

各项的完整指标、残差及静态关节限位检查在[结果说明](results.md)，实际困难与处理在
[心得](lessons.md)。固定直行通过原门槛不代表所有转向都通过严格足部门槛；
状态渲染不代表sim2sim或实机控制。

## 可查看的训练证据

| 子任务 | 实际CSV | 曲线 | 验收 |
| --- | --- | --- | --- |
| 羽毛球 | [CSV](../evidence/badminton_training.csv) | [图](../media/badminton_training.png) | [JSON](../evidence/badminton_acceptance.json) |
| 转向 | [CSV](../evidence/steering_training.csv) | [图](../media/steering_training.png) | [固定直行](../evidence/steering_fixed075_metrics.json) / [主转向](../evidence/steering_metrics.json) / [顺时针](../evidence/steering_clockwise_metrics.json) |
| 庆祝 | [CSV](../evidence/celebration_training.csv) | [图](../media/celebration_training.png) | [JSON](../evidence/celebration_acceptance.json) |
| 挑衅 | [CSV](../evidence/taunt_training.csv) | [图](../media/taunt_training.png) | [JSON](../evidence/taunt_acceptance.json) |

默认执行`python scripts/build_showcase.py`只重画这些CSV，不启动模拟器或训练。

## 训练 / 回放入口

入口代码已经公开：下面命令使用拥有权限的恢复材料和官方运行库。`$RestoreRoot`是
恢复脚本产生的**新目录的绝对路径**，不是旧电脑路径。完整源码包含本地T800适配，
单拿一个overlay配上游最新主分支不保证可用。公共代码可检查，许可受限材料仍需授权。

### T800庆祝与挑衅

实际使用Python3.11、Isaac Sim5.1、Torch2.7+cu128、RSL-RL5.0.1和固定源码快照。
依赖安装按[恢复说明](reproduce.md)及归档RUN.md执行；Native从零重装未在新机器上验证。
本人阅读并接受NVIDIA许可后，明确设置EULA；以下默认仅回放：

```powershell
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
pwsh -File scripts/run_action.ps1 -RestoreRoot $RestoreRoot -Python $NativePython -Action celebration
pwsh -File scripts/run_action.ps1 -RestoreRoot $RestoreRoot -Python $NativePython -Action taunt
```

明确加`-Mode train`才启动1024环境、seed42的新20k实验，例如：

```powershell
pwsh -File scripts/run_action.ps1 -RestoreRoot $RestoreRoot -Python $NativePython -Action celebration -Mode train
```

入口取同一版本清单的任务/奖励，不混用Wo与完整状态模型。其`train`模式是从零的新训练，
并非逐段重做历史挑衅2k+2k+2k+2k续训；精确历史切换由JSON训练链和归档run参数说明。
这份公开wrapper复用已验证入口，本次整理只做语法/参数检查，没有为发布而重新训练。

### T800 SMP

Linux运行配方为[Dockerfile.smp](../environment/Dockerfile.smp)及固定
[包版本](../environment/smp-runtime.txt)。从官方Torch2.7/CUDA12.8镜像重建，
训练源码为匹配的SMP/T800与MjLab快照，不混入旧Python挂载依赖：

```powershell
docker build -f environment/Dockerfile.smp -t t800-smp-runtime:restore environment
pwsh -File scripts/run_steering.ps1 -RestoreRoot $RestoreRoot -Image t800-smp-runtime:restore
```

第二行只回放。需要新续训时先显式smoke；同一个公开launcher复用既有runner，
默认奖励参数为选定36991版本，输出必须是新目录：

```powershell
$selected = Get-Content "$RestoreRoot/selected_models.json" -Raw | ConvertFrom-Json
$priorDir = Split-Path -Parent (Join-Path $RestoreRoot $selected.steering.prior)
pwsh -File scripts/launch_steering_quality.ps1 -Mode Smoke `
  -OutputRoot "$RestoreRoot/new-training/smoke1" -SourceRoot "$RestoreRoot/source" `
  -PriorRoot $priorDir -BaseCheckpoint (Join-Path $RestoreRoot $selected.steering.checkpoint) `
  -Image t800-smp-runtime:restore
```

Smoke是32环境/2轮，仅缩小初始化样本池；明确换为`-Mode Train`并使用另一个输出目录，
才执行512环境/500更新、seed42、每100轮保存。正式GSI池4096/批1024、compile开启，
不会用smoke的小池代替正式配置。历史分支使用自己的配置，不能靠当前默认值复写历史。
SMP新镜像实际策略复跑与线上恢复已通过；本次发布没有再执行新PPO。

### G1视频动作管线

原公开管线代码迁入[code/pipeline](../code/pipeline)，保留原固定上游版本和数据检查工具。
在对应BeyondMimic/Isaac环境、转换并回放参考后，上游训练入口示例是：

```bash
/isaac-sim/python.sh scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
  --motion_file="$MOTION_NPZ" --num_envs=256 --seed=42 --headless
```

这是用授权输入开展新训练的入口，不保证从零重建历史4493。最终历史记录使用固定起点
CPU PhysX恢复验证，与历史GPU训练不同；输入视频、人体模型及精确历史参数须按许可
和归档记录获取，不能把模板命令写成未经测试的新机器端到端成功。
