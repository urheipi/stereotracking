# 管道中心线跟踪交接 · 2026-11-25

项目用首帧手工圈定的管道选区（ROI）跟踪图像序列，逐帧输出两侧管壁和中心线。后续修改沿用 `track_main_tube_sequence.py` 这一套序列实现。实时预览、全局录像和局部录像共用同一帧的测量结果。

## 接手后怎样运行

在项目根目录打开 PowerShell。本机验证环境为 Windows、Python 3.12.7。新环境用 `python -m pip install -r requirements-dev.txt` 安装运行和测试依赖。BMP 样本、首次手绘选区和录像都保存在本地。新克隆需要准备图片；没有选区文件时，从首帧手动画即可。

| 场景 | 操作 |
| --- | --- |
| 在本机查看样本 | 双击 `run_testbmp_live.cmd`，默认读取 `testbmp/` |
| 处理其他采集目录 | `.\run_testbmp_live.cmd -ImageDir "D:\采集数据\第二组"`，PNG 序列可追加 `-Pattern "*.png"` |
| 无窗口复跑本机 201 帧 | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_tracking.ps1 -ImageDir .\testbmp -InitialRois .\tracking_results\drawn_rois.json` |

首帧沿管道外围依次左击至少三个点，按 **N** 完成一根，按 **Enter** 开始。查看跟踪时，滚轮缩放，右键拖动平移，**P** 暂停，**N** 单步，**C** 继续。换机位或分辨率后应在新图上重新框选。无窗口命令可追加 `-MaxFrames 40` 做快速验证。

## 已实现的行为与代码位置

| 要修改的行为 | 从这里读起 |
| --- | --- |
| 两侧管壁与中心线 | `detect_main_tube.py` 的 `find_edge_pair()`、`detect_roi()`。在拉正的选区内，结合管宽、边缘连续性和同一截面两侧的亮度变化选取边界，再拟合线段。较细的横穿线条可以只影响部分截面。中心线来源记录在 `centerline_source`。 |
| 选区运动与速度 | `track_main_tube_sequence.py` 的 `_estimate_roi_transform()`。在局部窗口内计算光流；特征集中在管道一端时只估计平移，避免将局部运动放大成整框缩放。光流不足时尝试模板匹配，位置不唯一时保留搜索框。 |
| 丢失与恢复 | 同文件的 `assess_visibility()`、`reject_invalid_updates()`。当前帧须达到中等或高置信度，至少 20% 的截面同时支持两侧管壁。连续候选管宽比须在 2/3 到 1.5 之间，扣除搜索框位移后，横向偏差默认最多 7 像素。管壁证据不足、测量突跳或几何校正被拒时，当帧标记 `lost`，清空三条线。继续搜索并取得连续两帧相符测量后恢复显示。 |
| 画面与录像 | `detect_main_tube.py` 的 `TUBE_LINES` 定义配色，两侧管壁均为青色，中心线为黄色。`review_window.py` 管理缩放和步进，`tracking_videos.py` 写同步全局视频和按 ROI 编号排列的局部放大视频。红底 `LOST` 表示这一帧无法确认中心线。 |

## 已验证的效果

本机参考运行使用 `testbmp/` 中的 201 张 BMP 和首次手绘的 10 个选区。两份视频均为 201 帧、10 fps。以下帧号从 1 开始，与视频显示一致。

| 选区 | 可复查结果 |
| --- | --- |
| R6 | 201 帧都有当前帧测量。可在最后一帧局部视图检查中心线与两侧青线的位置。 |
| R9 | 第 1–54 帧有测量。第 55 帧管宽突变，第 56 帧等待第二次确认，第 57 帧恢复。第 58–201 帧持续丢失。共 146 帧显示 `LOST`，这些帧的 JSON `centerline` 均为 `null`。 |
| R1–R5、R7、R8、R10 | 各有 201 帧输出测量。 |

这些数字统计的是测量可用性。中心线的定位精度仍需结合图像复核。目前实拍验证覆盖这段测试序列；独立运动、反向、细线横穿及遮挡后恢复另有合成回归用例。

## 输出与后续验证

启动脚本默认将每次结果放入 `tracking_results/runs/<启动时间>/`。此前的六组结果集中在 `tracking_results/archive/`。上述参考运行的本机目录为 `tracking_results/runs/20261125_112342_702/`，其中 `roi_review_*.mp4` 用于逐根检查管壁，`tracking_review_*.mp4` 用于核对全图位置，`sequence_centerlines_*.json` 用于按帧查询 `roi_idx`、`status`、`failure_reason` 和 `wall_pair_coverage`。

修改前按项目约定先用 ponytail 审查。修改后在根目录执行以下命令。本次交接时 67 项自动化测试通过，Ruff 检查通过。

```powershell
python -m unittest -q
python -m ruff check .
```

涉及丢失判定时，先检查原图中两侧管壁是否可辨，再查对应帧的 `failure_reason`、双侧支持比例和选区运动记录。涉及绘制时，同时核对两个视频中同一帧的颜色、位置和 `LOST` 状态。扩展到新的采集序列时，应使用该序列自己的首帧选区验证，避免按 R6、R9 编号或固定帧号调整算法。
