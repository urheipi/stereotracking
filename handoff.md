# 管道中心线跟踪交接 · 2026-11-27

## 项目目标与当前进度

用户在首帧手工圈出要跟踪的管道。程序在后续每张图中寻找同一根管道的两侧管壁，估计中心线，输出逐帧 JSON、全局视频和按选区编号排列的局部放大视频。用户用青色双壁线核对黄色中心线的位置，在实时预览中可缩放。若当前帧无法确认双壁，应显示红底 `LOST` 并令 `centerline` 为 `null`；搜索框继续寻找目标。后续开发沿用一个序列入口，修改应适用于其他图像和选区。默认按机位写入 `tracking_results/runs/left_camera/` 或 `right_camera/` 下的时间戳目录。

## 双目配对与全局视频 · 2026-11-27

原图分别在 `F:\tracking_test\trackingbmp`（左，988 张）和 `F:\tracking_test\trackingright`（右，991 张）。按文件名前缀的毫秒时间戳，相差不超过 1,000 ms 的左右图共有 987 对。用文件名后半段的日期、时分秒和毫秒独立检查，匹配结果一致。所有候选均唯一。前缀时间差绝对值最大 75 ms，后半段时刻差最大 416 ms。

双目处理按[配对表](tracking_results/stereo_pairs_20250517.csv)的 `pair_id` 读取同一行的两张原图。表中只有已配对的图片。`left_frame` 和 `right_frame` 是各自跟踪视频及 JSON 的帧号（从 1 开始），两个时间差列均为右减左，单位毫秒。原图保留在原目录。以下 5 张在另一机位的 1 秒窗口内没有图片。

| 机位 | 单机帧号 | 未配对的文件 |
| --- | ---: | --- |
| 右 | 35 | `63883096137007_2025-05-17 16-28-47_307.bmp` |
| 右 | 147 | `63883097257018_2025-05-17 16-47-27_311.bmp` |
| 右 | 554 | `63883101427016_2025-05-17 17-56-57_343.bmp` |
| 左 | 623 | `63883102146998_2025-05-17 18-08-57_223.bmp` |
| 右 | 770 | `63883103647007_2025-05-17 18-33-57_352.bmp` |

全局画面的灰色选区框原先逐帧采用检测修正后的搜索框，管段可见长度变化时，标签随端点一起跳动。现在按 ROI 编号对齐相邻帧的多边形顶点，沿管道方向平滑框的位置和长度，横向滞后限制在当前框宽的 7% 以内。编号与状态的标签以框内同一相对位置为中心放置。青色双壁线、黄色中心线和 LOST 状态仍取自当前帧，JSON 保存实际搜索框。右 R1 全程框中心的相邻帧位移第 95 百分位由 36.0 降到 7.2 原图像素；右 R1 第 301–400 帧按绘制位置估算，标签中心位移第 95 百分位由 24.2 降到 9.1 原图像素。从两份实际编码视频中定位绿色标签文字，测得同一指标由 24.2 降到 10.0 原图像素。

两台相机均已用原选区完成全长导出。左相机通过 D 盘临时缓存读取同一批 988 张原图，JSON 继续记录 F 盘的原图路径。四份 10 fps 视频已逐帧读回，逐帧 JSON 与 11 月 26 日的完整运行一致。已核对配对表的全部 987 行，左右帧号都能定位到对应文件名。单机录像保留原始帧序，双目使用配对表中的左右帧号。

| 机位 | 总帧数 | 稳定显示后的完整结果 |
| --- | ---: | --- |
| 左相机 | 988 | [全局视频](tracking_results/runs/left_camera/20261127_140424_876/tracking_review_20261127_140424.mp4) · [局部视频](tracking_results/runs/left_camera/20261127_140424_876/roi_review_20261127_140424.mp4) · [逐帧结果](tracking_results/runs/left_camera/20261127_140424_876/sequence_centerlines_20261127_140424.json) |
| 右相机 | 991 | [全局视频](tracking_results/runs/right_camera/20261127_123923_274/tracking_review_20261127_123923.mp4) · [局部视频](tracking_results/runs/right_camera/20261127_123923_274/roi_review_20261127_123923.mp4) · [逐帧结果](tracking_results/runs/right_camera/20261127_123923_274/sequence_centerlines_20261127_123923.json) |

## 管道提取复查 · 左右相机

| 机位 | 输入、首帧选区 | 11 月 26 日完整运行 |
| --- | --- | --- |
| 右相机，991 帧 | `F:\tracking_test\trackingright`，使用[原始选区](tracking_results/runs/right_camera/20261126_154811_605/drawn_rois.json)。原问题视频也已移到该目录。 | [局部放大](tracking_results/runs/right_camera/20261126_222149_064/roi_review_20261126_222149.mp4) · [全局](tracking_results/runs/right_camera/20261126_222149_064/tracking_review_20261126_222149.mp4) · [逐帧 JSON](tracking_results/runs/right_camera/20261126_222149_064/sequence_centerlines_20261126_222149.json) |
| 左相机，988 帧 | `F:\tracking_test\trackingbmp`，使用[原始选区](tracking_results/runs/left_camera/20261125_114718_290/drawn_rois.json)。 | [局部放大](tracking_results/runs/left_camera/20261126_222148_893/roi_review_20261126_222148.mp4) · [全局](tracking_results/runs/left_camera/20261126_222148_893/tracking_review_20261126_222148.mp4) · [逐帧 JSON](tracking_results/runs/left_camera/20261126_222148_893/sequence_centerlines_20261126_222148.json) |

右相机 R7 在第 239 帧被细线上的少数光流点带走，估计跳动约 125 原图像素，此后搜索框偏离原管。现在遇到内点少且运动异常时，用选区连同周围纹理复核位移。第 239、735、991 帧的青线回到原管两侧，整段报丢失由 755 帧降至 1 帧（第 394 帧）。

右相机 R1 的部分管段被横梁遮住，改为在两壁实际共同可见的长度内评估缩短管段的支持率。第 89、330 帧能画出可见部分，报丢失由 56 帧降至 8 帧。R10 末段的栏杆与管壁之间有细亮缝，旧结果第 970 帧左边线落到栏杆外侧。现在用已有管宽识别持续的窄暗条，配合原图亮度验证候选，已确认双壁时保持两壁中点。第 930、970、991 帧的管壁经对照原图复查，整段报丢失由 43 帧降至 0 帧。以上规则共用于两台相机。

右相机共 9,910 次测量，报丢失由 855 次降至 10 次。仍需复核 R1 第 427–428、474–475、961–964 帧，R7 第 394 帧，R9 第 969 帧。左相机 R1、R3 全程有输出，已复核 R1 第 50、895 帧及 R3 第 830 帧。左相机合计报丢失由 43 次增至 49 次（R6 2、R9 12、R10 35）。下一步优先看左 R10 曝光变化后的第 336–365 帧，尤其是第 350 帧。以上统计按逐帧 JSON 计算，现场数据缺少逐帧人工标注，抽查以外的边缘误差仍待验证。

本机项目根目录为 `D:\1_stereoline\tracking\stereotracking`。当前阶段的代码和学习文档在 `main` 分支。在另一台已克隆项目的电脑上运行 `git pull --ff-only origin main` 获取该分支的更新。原图、现场选区和录像需单独取得；配对表可从两份逐帧 JSON 复现。[项目 README](README.md) · [GitHub 仓库](https://github.com/urheipi/stereotracking)

## 历史左相机文件

以下相对路径都从上述项目根目录开始。**现场 988 帧和历史 201 帧用了不同的手绘选区，R 编号只在各自的选区文件内有效。**

| 需要的文件 | 路径 |
| --- | --- |
| 现场 988 张原始 BMP | `F:\tracking_test\trackingbmp\` |
| 现场首帧选区，复现本次结果必须用这份 | [drawn_rois.json](tracking_results/runs/left_camera/20261125_114718_290/drawn_rois.json) |
| 上一版左相机局部放大视频，优先打开 | [roi_review_20261126_152039.mp4](tracking_results/runs/left_camera/20261126_152038_540/roi_review_20261126_152039.mp4) |
| 上一版左相机全局视频，核对是否跟到同一根管道 | [tracking_review_20261126_152039.mp4](tracking_results/runs/left_camera/20261126_152038_540/tracking_review_20261126_152039.mp4) |
| 上一版左相机逐帧结果，帧序与视频一致 | [sequence_centerlines_20261126_152039.json](tracking_results/runs/left_camera/20261126_152038_540/sequence_centerlines_20261126_152039.json) |
| 本次反馈中出现 R1 偏线、R3 被斜线带偏的录像与记录 | `tracking_results/runs/left_camera/20261126_094231_803/` |
| 用户最初指出问题的 988 帧录像与记录 | `tracking_results/runs/left_camera/20261125_114718_290/` |
| 本机快速验证图片 | `testbmp/`，文件名与现场序列的前 201 帧逐一对应。此前用**现场选区**跑这段的结果在 `tracking_results/runs/left_camera/20261125_212217_236/` |
| 早期独立手绘选区及历史结果 | `tracking_results/drawn_rois.json`、`tracking_results/runs/left_camera/20261125_112342_702/`。该运行曾用 R9 检查遮挡时能否及时报丢失。更早的六组实验已归入 `tracking_results/archive/` |

## 历史左相机问题、修复方法与实测效果

| 现场现象 | 实现与结果 |
| --- | --- |
| 曝光突然变亮，多根肉眼可见的管道同时报丢失；R6 有时选中管内明暗分界。 | 利用外侧亮背景与内侧亮度差识别真实轮廓。曝光保护可冻结搜索框的收缩，同时保留双壁可信、与冻结前管宽相符的当前测量。第 734 帧从十根全部丢失变为七根继续输出。R6 在第 735 帧恢复，整段丢失从 210 帧降至 1 帧。 |
| R10 管道紧挨平行栏杆，较强的栏杆边缘会把拟合线拉走。 | 按所选管壁的亮度变化方向拟合，限制相邻强边对搜索范围的影响，并用近期确认的管宽约束候选。R10 丢失从 609 帧降至 19 帧；前 201 帧逐帧均有测量。曝光突变后在第 741 帧恢复。 |
| 长时间运行后，光流的小幅缩放误差会积累，搜索框可能窄到裁掉管壁。 | 以近期确认的管宽保护搜索框宽度，以首次确认的管段长度和后续管宽变化保护搜索长度。R1 丢失从 69 帧降至 46 帧，仍是剩余丢失最多的选区。 |

上表三条记录的是本轮修改前的阶段结果（`20261126_094231_803`）。当时丢失帧数为 R1 46、R4 2、R6 1、R9 13、R10 19，其余选区全程有输出。丢失帧数只衡量当帧是否有测量；全程误差仍缺逐帧人工真值。

本轮 R1 的左壁经过天空、格栅时亮度变化会反向，另一壁的反光在深色背景上仍清晰。检测器检查窄亮边是否在近、远两处都比内外两侧亮，同时要求对侧有亮度台阶；候选管宽参考先前实测值。R3 的细线斜穿左壁时，沿两壁各自的候选方向统计整段边缘支持，清晰一壁可约束共同方向。光流只剩少数点却预测明显缩放时，改用整个选区核实位移。搜索框因可见长度缩短而保留时，有充分双壁证据的实测管段仍可输出。标签按本帧多边形放置，字号随可用空间缩小。

**实图复核**　在原图选可辨认的左管壁截面，把同一位置投影到两次录像，按原图尺度估计青线的横向偏差。R1 第 35、50、70 帧分别由约 4.0、5.8、2.9 像素降到 0.8、2.4、2.0 像素；第 895 帧由约 21.8 降到 1.0 像素。第 860–919 帧由 18 帧丢失降至 0。R3 第 830 帧左壁上部由约 5.7 降到 2.0 像素，下部约 0.3 像素。另用清晰右壁的亮度过渡独立估计方向，将方向差折算成每 700 原图像素的横向偏移，第 820–840 帧中位数由 2.40 降至 0.36 像素。录像线宽、压缩与截面选点均会影响读数。

## 下一位 agent 从哪里查

1. 在局部视频看 R1 第 30–80、860–919 帧及 R3 第 820–840 帧。时间换算为 `(帧号 - 1) / 10` 秒。原图路径在 JSON 的 `frames[帧号 - 1].image`，该帧的 `centerlines` 按 `roi_idx` 查找。
2. 在 R1 第 50、895 帧看左壁经过天空和格栅的位置，在 R3 第 830 帧看斜线与左壁分开的上下两段。画线是否贴合需要对照同一原图截面，单看 `status: ok` 无法判断。
3. 改动后同时回看 R1 第 561–575 帧与 R6、R10 在第 336、734、961 帧附近的曝光变化，以及历史选区 R9 的逐渐遮挡。使用同一份选区复现，跨数据集另选首帧 ROI。项目约定在写代码前先用 ponytail 审查。

| 要改的行为 | 文件与函数 |
| --- | --- |
| 边缘候选、管壁拟合和中心线 | `detect_main_tube.py` 的 `find_edge_pair()`、`_fit_boundary_lines()`、`detect_roi()` |
| 选区运动、尺寸与曝光保护 | `track_main_tube_sequence.py` 的 `_estimate_roi_transform()`、`_preserve_search_extent()`、`guard_exposure_shrunken_rois()`。局部光流跟随运动，特征集中在一端时只估计平移 |
| 丢失与重新捕获 | 同文件的 `assess_visibility()`、`reject_invalid_updates()`。默认要求中等或高置信度、至少 20% 的截面同时支持两壁，相邻候选管宽比在 2/3–1.5 内，补偿搜索框运动后的横向偏差上限为 7 像素。丢失后需连续两帧相符测量才能恢复 |
| 显示与录像 | `review_window.py` 管理缩放和步进，`tracking_videos.py` 写两份同步视频。全局和局部画面共用 `detect_main_tube.draw_roi_label()`，按本帧多边形确定标签位置、字号。`TUBE_LINES` 定义线条与颜色 |

## 怎样运行和验收

本机验证环境为 Windows、Python 3.12.7。新环境可在项目根目录执行 `python -m pip install -r requirements-dev.txt`。在 PowerShell 中复现这两次现场运行：

```powershell
cd D:\1_stereoline\tracking\stereotracking
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_tracking.ps1 -ImageDir "F:\tracking_test\trackingright"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_tracking.ps1 -ImageDir "F:\tracking_test\trackingbmp" -InitialRois ".\tracking_results\runs\left_camera\20261125_114718_290\drawn_rois.json"
```

快速复现前 201 帧时，将同一命令中的 `-ImageDir` 改为 `.\testbmp`，保留现场 `-InitialRois`。要自己选管道，双击 `run_testbmp_live.cmd`；换一组图片则运行 `.\run_testbmp_live.cmd -Camera right -ImageDir "D:\采集数据\第二组"`。首帧沿管道外围至少点三次，按 **N** 完成一根，按 **Enter** 开始。预览时滚轮缩放、右键拖动，**P** 暂停、**N** 单步、**C** 继续。

代码验证命令如下。本次复核为 **80 项测试通过，Ruff 通过**。用例覆盖全局显示框的平滑、原始测量与丢失提示，以及相机选区匹配与分目录输出、贴近管壁的细栏杆、浅角度斜线、反光与背景立杆、标签随选区移动、少量光流点造成的假缩放，以及可见管段变短时的双壁校验。另用历史选区按 `-TrackScale 1` 跑完 201 帧，R9 有 128 帧标记丢失，其他九个选区全程有测量。R9 第 100、150、201 帧均为 `LOST`，但第 75、80、85 帧仍可能把遮挡物边缘误作管壁。这组图来自同一段采集，尚未验证独立现场数据。

```powershell
python -m unittest -q
python -m ruff check .
```

现场数据太大且需要远程排查时，可在同一命令追加 `-Diagnostics`。运行目录会生成可手动分享的 `*_diagnostics.zip`，其中 `summary.json` 汇总丢失时段，`measurements.csv` 按帧和 ROI 记录判定依据，`samples/` 最多包含 32 张局部原图与画线结果。现场使用者标出帧号和 ROI 编号有助于查找误跟。本次 988 帧复核直接使用原图、两份视频和逐帧 JSON。
