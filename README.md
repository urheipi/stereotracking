# main_tube_detect 项目文档

## 项目实现了什么功能

本项目用于在图像中的多边形 ROI 内检测主导管的左右边界和中心线，并支持把首帧 ROI 跟踪到后续图像序列中，逐帧输出主导管中心线结果。

项目主要包含两个入口脚本：

- `detect_main_tube.py`：单张图像主导管检测。
- `track_main_tube_sequence.py`：图像序列 ROI 跟踪和逐帧中心线提取。

核心能力如下：

- 支持读取灰度图或彩色图，并兼容包含中文路径的图像读写。
- 支持从 JSON 读取 ROI，也支持交互式绘制多边形 ROI。
- 在每个 ROI 内检测主导管左右边界、中心线、置信度和调试指标。
- 输出完整检测结果 JSON、简化中心线 JSON、叠加可视化图和可选调试图。
- 对图像序列支持从首帧 ROI 自动传播到下一帧，再调用单帧检测器提取每帧中心线。
- 可生成序列复核视频，也可打开交互式窗口逐帧检查。

这个项目不是基于深度学习训练模型的检测器，而是基于传统计算机视觉和几何规则的 ROI 内目标检测与跟踪流程。

## 项目文件说明

```text
main_tube_detect/
├── detect_main_tube.py                 # 单张图像 ROI 内主导管检测
├── track_main_tube_sequence.py         # 图像序列 ROI 跟踪与中心线汇总
├── test_detect_main_tube.py            # 单图检测相关测试
├── test_track_main_tube_sequence.py    # 序列跟踪相关测试
├── pic/                                # 示例图像
├── output/                             # 单图检测输出示例
├── tracking_results*/                  # 序列跟踪输出示例
└── docs/                               # 额外算法说明文档
```

## 使用的主要算法

### 1. ROI 几何拉正

对应代码：`_basis_from_polygon()`、`rectify_polygon()`

程序先对用户提供的多边形 ROI 使用 `cv2.minAreaRect()` 求最小外接旋转矩形，取长边方向作为主导管的大致延伸方向，取其垂直方向作为宽度方向。然后基于这两个方向建立局部坐标系，并用 `cv2.remap()` 将原图 ROI 重采样为一个拉正后的矩形小图。

这一步的目的，是把原图中可能倾斜、旋转的导管区域转换成更规则的局部图像，使主导管左右边界在拉正图中尽量接近两条竖直长边。

### 2. Sobel 梯度边缘证据

对应代码：`_edge_evidence()`

拉正 ROI 后，程序先做高斯滤波降低噪声，再用 Sobel 算子计算水平和垂直方向梯度：

```text
gx = x 方向梯度
gy = y 方向梯度
```

主导管的左右边界在拉正图中通常表现为竖直边，因此灰度变化主要发生在 x 方向。程序用 `abs(gx)` 作为主要边缘强度，并通过方向权重过滤掉更像横向纹理或噪声的响应。

最终得到的 `evidence` 可以理解为“某个像素像主导管竖直边界的程度”。

### 3. 列投影和一维峰值搜索

对应代码：`_edge_evidence()`、`_smooth_1d()`、`_find_profile_peaks()`

程序只在 ROI 掩膜内部统计边缘证据，把每一列的边缘强度沿 y 方向求平均，形成一维宽度剖面 `profile`。真实的左右导管边界会在很多行上连续出现，因此会在 `profile` 中形成明显峰值。

程序对 `profile` 做一维高斯平滑，然后寻找局部峰值，并设置最小间隔，避免同一条边界附近产生多个重复候选。

### 4. 边界对评分

对应代码：`find_edge_pair()`、`_boundary_support()`

程序枚举峰值中的两两组合，把每一对候选看作“左边界 + 右边界”，并根据以下指标打分：

- 宽度约束：两条边之间的距离必须落在 `min_width_frac` 和 `max_width_frac` 之间。
- 连续性：候选边界附近有多少有效行存在足够强的边缘响应。
- 边缘对比度：候选峰值相对整体剖面背景是否突出。
- 宽度先验：过窄或过宽的候选会被降低权重。

得分最高的一对边界作为主导管左右边界候选。如果没有满足几何和边缘条件的候选，则该 ROI 标记为 `failed`。

### 5. 前景宽度修正

对应代码：`_foreground_column_profile()`、`_foreground_span_around()`、`_refine_pair_with_foreground_span()`

当 ROI 内前景和背景存在足够灰度差时，程序会额外统计较暗前景在每一列中的占比，估计主导管的整体前景宽度。如果初始边界对明显落在前景内部，程序会尝试用附近峰值把左右边界扩展到更合理的外侧边缘。

这一步用于降低“内部强边缘被误选为外边界”的概率。

### 6. 边界线拟合和中心线采样

对应代码：`_fit_one_boundary_line()`、`_fit_boundary_lines()`、`_sample_boundaries()`、`map_rectified_point()`

找到左右边界的大致 x 坐标后，程序会在每一行附近搜索局部最大边缘响应，收集边界采样点，并用加权一次直线拟合：

```text
x = slope * y + intercept
```

左右边界需要满足斜率接近、残差合理、支撑行数足够等条件。拟合通过后，程序使用共同斜率构造近似平行的左右边界线，再取两条边界中点作为中心线。

最后，所有边界点和中心线点都会通过 `map_rectified_point()` 从拉正图坐标映射回原始图像坐标。

### 7. 置信度评估

对应代码：`find_edge_pair()`

检测成功后，程序根据连续性、边缘对比度和宽度比例给出置信度：

- `high`：边界连续、对比明显、宽度合理。
- `medium`：边界基本成立，但连续性或对比度一般。
- `low`：找到了候选边界，但质量较弱。
- `failed`：没有找到有效边界对。

### 8. 序列 ROI 跟踪

对应代码：`track_rois_between_frames()`、`_estimate_roi_transform()`

序列处理不重新让用户逐帧画 ROI，而是从首帧 ROI 开始跟踪到下一帧。使用的算法链路是：

1. 在上一帧 ROI 掩膜内用 `cv2.goodFeaturesToTrack()` 提取角点特征。
2. 用金字塔 Lucas-Kanade 光流 `cv2.calcOpticalFlowPyrLK()` 跟踪特征点到下一帧。
3. 再反向跟踪回上一帧，用 forward-backward error 过滤不稳定点。
4. 用 `cv2.estimateAffinePartial2D()` 和 RANSAC 估计上一帧到下一帧的局部仿射变换。
5. 将该变换应用到 ROI 顶点，得到下一帧 ROI。
6. 对下一帧传播后的 ROI 再调用单帧检测器提取主导管中心线。

当仿射估计失败时，程序会使用跟踪点的中位平移量作为降级方案；如果跟踪点不足，则保留上一帧 ROI 位置并记录失败原因。

## 单张图像检测流程

```text
输入图像 + ROI JSON 或交互式绘制 ROI
        |
        v
标准化 ROI 数据
        |
        v
逐个 ROI 拉正为局部矩形图
        |
        v
计算 Sobel 边缘证据和列方向 profile
        |
        v
寻找候选峰值并选择最佳左右边界对
        |
        v
拟合左右边界线并计算中心线
        |
        v
映射回原图坐标
        |
        v
输出 results.json、centerline_results.json、overlay_detected.png
```

示例命令：

```bash
python detect_main_tube.py --image pic/1.bmp --draw-rois --out-dir output/1_draw --debug
```

使用已有 ROI JSON：

```bash
python detect_main_tube.py --image pic/1.bmp --rois output/1_draw/drawn_rois.json --out-dir output/verify --debug
```

ROI JSON 支持以下格式：

```json
{
  "image": "pic/1.bmp",
  "rois": [
    {
      "roi_idx": 1,
      "vertices": [[100, 120], [140, 300], [210, 290], [180, 110]]
    }
  ]
}
```

兼容旧字段 `roi_points`，程序会统一转换为 `vertices`。

## 图像序列处理流程

```text
输入有序图像序列
        |
        v
读取或绘制首帧 ROI
        |
        v
首帧直接运行单图检测
        |
        v
从上一帧 ROI 内提取角点
        |
        v
LK 光流跟踪到下一帧
        |
        v
forward-backward 误差过滤
        |
        v
RANSAC 估计局部仿射变换
        |
        v
传播 ROI 到当前帧
        |
        v
当前帧运行单图检测
        |
        v
循环直到所有帧处理完成
        |
        v
输出 sequence_centerlines.json 和可选 tracking_review.mp4
```

按目录读取图像：

```bash
python track_main_tube_sequence.py --image-dir pic --pattern "*.bmp" --initial-rois output/1_draw/drawn_rois.json --out-dir tracking_results
```

显式指定有序图像：

```bash
python track_main_tube_sequence.py --images pic/1.bmp pic/2.bmp pic/3.bmp pic/4.bmp --initial-rois output/1_draw/drawn_rois.json --out-dir tracking_results
```

如果不需要实时窗口：

```bash
python track_main_tube_sequence.py --image-dir pic --pattern "*.bmp" --initial-rois output/1_draw/drawn_rois.json --out-dir tracking_results --no-window
```

## 输出文件

单图检测默认输出到 `--out-dir`：

- `results.json`：完整检测结果，包括 ROI、左右边界采样点、中心线采样点、宽度、连续性、候选峰值和置信度。
- `centerline_results.json`：简化中心线结果，只保留每个 ROI 的中心线端点、二维 Plucker 表达和直线方程。
- `overlay_detected.png`：原图上的检测叠加图。
- `debug/roi_XX_rectified.png`：可选调试图，展示拉正 ROI、边界和 profile。
- `debug/all_rois_rectified_eval.png`：可选调试拼图。

序列处理默认输出到 `--out-dir`：

- `sequence_centerlines.json`：每一帧的中心线摘要。
- `tracking_review.mp4`：可选复核视频。
- `drawn_rois.json`：如果使用 `--draw-rois`，会保存首帧交互绘制的 ROI。

中心线摘要中的核心字段：

```json
{
  "roi_idx": 1,
  "status": "ok",
  "confidence": "high",
  "centerline": {
    "endpoints": [x1, y1, x2, y2],
    "plucker": [moment, dx, dy],
    "line_eq": {"A": 0.0, "B": 1.0, "C": -10.0}
  }
}
```

其中 `line_eq` 表示原图坐标系下的二维直线方程：

```text
A*x + B*y + C = 0
```

## 关键参数

### 单图检测参数

- `--min-width-frac`：主导管最小宽度占拉正 ROI 宽度的比例，默认 `0.28`。
- `--max-width-frac`：主导管最大宽度占拉正 ROI 宽度的比例，默认 `0.92`。
- `--boundary-samples`：每个 ROI 输出的边界/中心线采样点数量，默认 `60`。
- `--debug`：输出拉正 ROI 和 profile 调试图。

### 序列跟踪参数

- `--track-scale`：光流跟踪使用的缩放比例，默认 `0.5`。检测结果和 JSON 仍保持原图坐标。
- `--max-corners`：每个 ROI 内最多提取的角点数量，默认 `80`。
- `--min-track-points`：每个 ROI 至少需要的有效跟踪点数量，默认 `3`。
- `--max-forward-backward-error`：正反向光流误差阈值，默认 `1.5` 像素。
- `--review-mode`：复核窗口模式，支持 `step` 和 `continuous`。
- `--pause-on`：连续复核时遇到 `failed` 或 `low` 是否暂停。

## 依赖环境

项目依赖 Python、OpenCV 和 NumPy：

```bash
pip install opencv-python numpy
```

运行测试：

```bash
python -m unittest
```

开发环境可额外安装 pytest 和 Ruff，并运行统一质量检查：

```bash
pip install -r requirements-dev.txt
python -m pytest -q
ruff check .
```

## 适用场景和限制

适用场景：

- 用户已经知道主导管大致位置，并能给出覆盖主导管的多边形 ROI。
- 主导管左右边界在 ROI 内具有较稳定的灰度边缘。
- 图像序列相邻帧运动较小，ROI 内有足够纹理或角点可用于光流跟踪。

主要限制：

- 该方法依赖 ROI，不能在整张图中自动搜索所有导管。
- 如果 ROI 没有覆盖完整主导管，或包含过多强干扰边缘，检测可能偏移。
- 如果导管边缘对比度很低、严重遮挡或反光过强，置信度会下降。
- 序列跟踪依赖相邻帧局部仿射近似，过大形变或特征点不足会导致 ROI 传播失败。
#   s t e r e o t r a c k i n g  
 