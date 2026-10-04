# 视频到G1策略：原管线代码

从本人旧的`gvhmr-gmr-beyondmimic-reproducible`仓库迁入，原版本为
`c8bf217557f5ef5a33c0ba2de4a582cf6fd9eef9`。这里只保留源码固定、环境检查和
数据/回放检查工具，不搬入全部中间文件；旧Git历史及LFS资料进入私有备份。

实际流程是：视频 → GVHMR人体恢复 → GMR重定向 → CSV/NPZ → BeyondMimic训练
→ 固定起点策略验证。原算法及机器人资产保持上游许可；不是从零实现这些算法。

```powershell
pwsh -File code/pipeline/bootstrap_repos.ps1 -Root ./vendor
python code/pipeline/check_environment.py
```

Linux/WSL使用`bash code/pipeline/bootstrap_repos.sh ./vendor`。源码版本见
[versions.env.example](versions.env.example)。安装和模型下载仍遵循上游README，
受限人体模型不得随仓库分发。

在固定源码及拥有授权的输入上，调用上游自己的入口：

```bash
cd vendor/GVHMR
python tools/demo/demo.py --video="$INPUT_VIDEO" -s --output_root="$DATA_ROOT/gvhmr/demo"
cd ../GMR
python scripts/gvhmr_to_robot.py --gvhmr_pred_file="$GVHMR_RESULT" \
  --robot=unitree_g1 --save_path="$GMR_OUTPUT" --record_video
cd ../BeyondMimic
/isaac-sim/python.sh scripts/csv_to_npz.py --input_file="$GMR_CSV" \
  --input_fps=30 --output_name=motion --headless
/isaac-sim/python.sh scripts/replay_npz.py --motion_file="$MOTION_NPZ" --headless
```

`$INPUT_VIDEO`等变量是自行提供的授权文件绝对路径，不是本人的旧机器路径。
转换脚本默认参数不能代替实际run记录；所选参考复用了原GMR逐帧落地处理。
训练命令、最后500轮实际覆盖及GPU训练/CPU恢复区别见
[训练记录](../../docs/training.md)，验收见[结果](../../docs/results.md)。

`validate_motion_npz.py`检查参考数组的有限性及帧间跳变；
`validate_rollout.py`检查已有回放NPZ。其默认阈值是旧管线的工具默认，
不是T800任务的门槛，也不意味着MuJoCo sim2sim或实机已经通过。
