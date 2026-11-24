# 小 ROI 自适应跟踪优化说明

## 1. 背景问题

原始跟踪逻辑主要依赖第一帧画好的小 ROI：

```text
上一帧 ROI
  -> ROI 内找角点
  -> LK 光流跟到下一帧
  -> RANSAC 估计仿射/相似变换
  -> 更新 ROI 顶点
  -> 在更新后的 ROI 内检测主管中心线
```

这个逻辑在目标运动较明显时会遇到两个问题：

1. ROI 内的角点可能来自背景，而不是柱体本身，导致 ROI 跟着背景不动或被带偏。
2. 光流/RANSAC 偶尔会估出异常变换，把小 ROI 放大或带飞，后续 `rectify_polygon()` 会生成过大的拉正图，导致 OpenCV `remap` 崩溃。

现在的优化目标是：

```text
仍然使用每根柱体一个小 ROI，
但让 ROI 每帧根据实际检测到的柱体位置自适应校正。
```

不做大 ROI 多目标自动检测，也不做跨目标 ID 重新分配。

---

## 2. 当前跟踪流程

当前默认流程是 `detect-corrected`：

```text
上一帧校正后的 ROI
  -> 光流/RANSAC 预测当前帧 ROI
  -> 检查预测 ROI 是否合理
  -> 在预测 ROI 内检测柱体左右边界和中心线
  -> 若检测成功，用检测结果重建当前帧 ROI
  -> 当前帧校正后的 ROI 作为下一帧输入
```

对应代码位置：

- `track_main_tube_sequence.py`
  - `track_rois_between_frames()`
  - `correct_roi_from_detection()`
  - `correct_rois_from_detections()`
- `detect_main_tube.py`
  - `roi_vertices_from_detected_boundaries()`

---

## 3. 光流预测 ROI

光流预测仍然保留，作用是给当前帧一个初始搜索位置。

步骤：

1. 在上一帧 ROI 内用 `cv2.goodFeaturesToTrack()` 找角点。
2. 用 `cv2.calcOpticalFlowPyrLK()` 做正向光流。
3. 再做反向光流，使用 forward-backward error 过滤坏点。
4. 用 `cv2.estimateAffinePartial2D(..., method=cv2.RANSAC)` 估计变换矩阵。

矩阵形式为：

```text
M =
[ s cos(theta)  -s sin(theta)  tx ]
[ s sin(theta)   s cos(theta)  ty ]
```

也就是：

```text
q ~= s R p + t
```

然后把该矩阵作用到 ROI 四个顶点上：

```text
v'_k = M [x_k, y_k, 1]^T
```

---

## 4. 光流预测 ROI 的合理性检查

为了防止 ROI 被光流/RANSAC 带飞，现在在 `track_rois_between_frames()` 内增加了预测 ROI 检查。

检查内容：

1. 顶点坐标必须是有限数值。
2. ROI 中心跳变不能超过 `--max-roi-jump`。
3. ROI 宽高跨度不能相对上一帧异常膨胀。

如果预测 ROI 不合理：

```text
status = failed
failure_reason = "unreasonable tracked roi"
```

并回退到上一帧 ROI，不再把异常 ROI 送进检测流程。

---

## 5. 检测结果反向校正 ROI

默认模式下，光流预测只是第一步。真正用于下一帧的 ROI 会由检测结果重新生成。

当当前帧检测成功时，程序会读取：

```text
left_boundary
right_boundary
centerline
```

然后重新构造一个包住当前柱体的小四边形 ROI。

重建逻辑：

1. 用中心线或左右边界中点估计柱体长轴方向。
2. 用左右边界估计柱体宽度方向。
3. 在长度方向增加 `--roi-length-padding-frac`。
4. 在宽度方向增加 `--roi-padding-frac`。
5. 得到新的四边形 ROI 顶点。

这样下一帧不再使用“首帧固定区域”，而是使用：

```text
上一帧实际检测到的柱体位置附近的 ROI
```

---

## 6. 检测失败时的处理

检测失败时不做大范围搜索，只保持小 ROI 策略。

处理规则：

1. 光流预测成功，但检测失败：
   - 使用光流预测 ROI 作为下一帧输入。
   - 标记 detection failed。

2. 光流预测失败：
   - 保留上一帧 ROI。
   - 标记 tracking failed。

3. 检测结果校正后 ROI 跳变过大：
   - 拒绝校正结果。
   - 保留预测 ROI。
   - 标记 `update_status = "rejected-roi-jump"`。

---

## 7. 检测层异常 ROI 兜底

除了跟踪阶段的防护，检测阶段也增加了防崩保护。

`rectify_polygon()` 在生成 `cv2.remap()` 所需网格前会检查拉正图尺寸：

```text
MAX_RECTIFIED_DIM = 32000
MAX_RECTIFIED_PIXELS = 40000000
```

如果 ROI 过大、尺寸异常或坐标非法，会直接返回失败结果：

```text
status = failed
confidence = failed
failure_reason = "invalid or oversized roi"
```

这样单个坏 ROI 不会中断整段序列处理。

同时 `make_roi_debug_panel()` 现在可以处理空 debug 数据。若某个 ROI 在检测前就失败，会生成一个失败占位 debug 面板，而不是访问不存在的 `debug["crop"]`。

---

## 8. 可视化状态

当前可视化保持原来的线条风格：

```text
左边界：红色
右边界：绿色
中心线：黄色
```

中心线可视化已经按要求回退到上一个 checkpoint，其他跟踪和异常防护优化仍然保留。

---

## 9. 新增命令行参数

### `--roi-update-mode`

控制 ROI 更新方式：

```text
detect-corrected  默认模式，光流预测后再用检测结果校正 ROI
flow              旧模式，只使用光流/RANSAC 更新 ROI
```

默认：

```text
--roi-update-mode detect-corrected
```

### `--roi-padding-frac`

检测成功后，重建 ROI 时在宽度方向额外扩展的比例。

默认：

```text
--roi-padding-frac 0.35
```

### `--roi-length-padding-frac`

检测成功后，重建 ROI 时在长度方向额外扩展的比例。

默认：

```text
--roi-length-padding-frac 0.10
```

### `--max-roi-jump`

限制单帧 ROI 中心最大跳变，单位是原图像素。

默认：

```text
--max-roi-jump 200
```

如果真实柱体运动速度很快，可以适当调大，例如：

```text
--max-roi-jump 500
```

### `--track-window-padding`

光流跟踪现在只在每个 ROI 周围的局部窗口内运行，而不是在整张原图上运行。该参数控制局部窗口向外扩展的原图像素数。

默认：

```text
--track-window-padding 200
```

建议优先保持：

```text
--track-scale 1
```

这样检测、ROI 更新和 LK 光流都能使用原图细节；速度主要靠局部窗口裁剪提升。如果目标单帧运动超过 200 px，或者跟踪经常报 `not enough valid tracked points`，可以把窗口放大到 `300` 或 `500`。

### `--track-retry-window-padding`

第一轮局部光流失败时，只对失败的 ROI 使用更大的局部窗口重试。重试阶段会先用模板匹配估计一个较大的平移初值，再交给 LK 光流细化，避免大位移时 LK 从零位移初值直接追丢。

默认：

```text
--track-retry-window-padding 800
```

大位移序列推荐：

```text
--track-window-padding 300
--track-retry-window-padding 1000
--max-roi-jump 300
```

不建议为了大位移直接关闭 `--max-anchor-drift` 或 `--max-corrected-roi-growth`。实测关闭这些防漂限制后，虽然状态可能仍显示 `ok`，但后段容易出现 `tracked` 回退线漂移；保留约束并只扩大失败重试窗口更稳。

---

## 10. 推荐运行命令

连续处理文件夹内 BMP 图像：

```powershell
python track_main_tube_sequence.py `
  --image-dir D:\1\测试专用\B_project\tracing_test\tarckingbmp `
  --pattern *.bmp `
  --sort-by timestamp `
  --draw-rois `
  --out-dir tracking_results_run `
  --review-mode continuous `
  --pause-on none
```

这个命令默认使用新的小 ROI 自适应校正模式。

如果需要恢复旧的纯光流跟踪模式：

```powershell
python track_main_tube_sequence.py `
  --image-dir D:\1\测试专用\B_project\tracing_test\tarckingbmp `
  --pattern *.bmp `
  --sort-by timestamp `
  --draw-rois `
  --out-dir tracking_results_run `
  --review-mode continuous `
  --pause-on none `
  --roi-update-mode flow
```

如果目标运动较快，建议增加跳变阈值：

```powershell
--max-roi-jump 500
```

如果检测校正后的 ROI 太紧，可以增大 padding：

```powershell
--roi-padding-frac 0.5
--roi-length-padding-frac 0.2
```

---

## 11. 当前策略的边界

当前优化仍然遵守以下假设：

1. 首帧需要给每根要跟踪的柱体单独画一个小 ROI。
2. 每个小 ROI 内主要只有一根目标柱体。
3. 不做大 ROI 内多柱体自动检测。
4. 不做目标 ID 重新分配。
5. 如果柱体单帧运动大到完全离开预测 ROI，仍可能检测失败；这时需要增大首帧 ROI、提高 `--max-roi-jump`，或提高 ROI padding。

当前方案优先解决的是：

```text
小 ROI 随柱体运动自适应更新，
并且坏 ROI 不再导致整个程序崩溃。
```

---

## 12. 2026-06-06 追加稳定化

本次优化针对当前 `tracking_results_run\drawn_rois.json` + `tarckingbmp` 序列增加了六层防护：

1. 窄柱体检测允许左右边缘最小间距降到 8 px。旧的 12 px 硬下限会让 ROI 1 在校正 ROI 连续变窄后失败。
2. `detect-corrected` 模式默认只允许 `medium` 及以上置信度的检测结果更新下一帧 ROI。`low` 置信度结果仍会写入输出，但不会再带偏跟踪状态。
3. 检测校正后的 ROI 默认保留至少 40 px 短轴宽度，并拒绝相对预测 ROI 跨度异常的校正结果，避免坏校正生成过大的拉正图。
4. 检测校正后的 ROI 默认不允许相对预测 ROI 继续放大（`--max-corrected-roi-growth 1.0`）。这解决了正确首帧 ROI 下 ROI 长度每帧约 1.2 倍膨胀、Jupyter 看起来卡在后续帧的问题。
5. 当当前帧检测偶发失败，但上一帧有真实检测结果可用时，输出会用 `confidence="tracked"` 的短期跟踪回退补中心线。该回退只用于输出连续性，不会更新下一帧 ROI 状态。
6. 检测校正后的 ROI 默认还会受首帧锚点约束（`--max-anchor-drift 30`）。当过曝导致检测在错误柱体或错误边缘上仍给出 `high` 置信度时，程序会拒绝这次更新，把 ROI 状态拉回首帧锚点附近，并用 tracked 回退输出，避免后续继续在偏移区域跟踪。

当前固定首帧 ROI 的 Jupyter 推荐命令：

```powershell
python track_main_tube_sequence.py `
  --image-dir tarckingbmp `
  --pattern *.bmp `
  --sort-by timestamp `
  --initial-rois tracking_results_run\drawn_rois.json `
  --out-dir tracking_results_run `
  --jupyter-friendly `
  --progress-every 20 `
  --track-scale 1 `
  --track-window-padding 300 `
  --track-retry-window-padding 1000 `
  --max-roi-jump 300
```

`--jupyter-friendly` 会关闭 OpenCV 窗口和视频写入，连续运行，不因失败暂停，并输出进度。当前版本未显式指定 `--track-scale` 时默认使用 `1.0`，因为光流已经改成局部窗口处理，不再需要靠整图降采样换速度。

如果现场目标确实存在明显平移，而不是曝光造成的误检漂移，可以适当放宽锚点：

```powershell
--max-anchor-drift 60
```

如果不希望使用首帧锚点约束，可显式关闭：

```powershell
--max-anchor-drift 0
```

如果某个真实目标必须允许低置信度检测更新状态，可显式降低门槛：

```powershell
--min-roi-update-confidence low
```
