# openMVS 稠密重建完整技术文档

**作者**: Claude
**日期**: 2025-11-20
**面向读者**: 希望深入理解多视图立体视觉稠密重建的研究者

---

## 目录

1. [概述](#1-概述)
2. [整体架构](#2-整体架构)
3. [数学基础](#3-数学基础)
4. [视图选择算法](#4-视图选择算法)
5. [PatchMatch深度图估计](#5-patchmatch深度图估计)
6. [深度图后处理](#6-深度图后处理)
7. [深度图融合](#7-深度图融合)
8. [代码流程详解](#8-代码流程详解)
9. [关键参数说明](#9-关键参数说明)
10. [完整示例](#10-完整示例)

---

## 1. 概述

### 1.1 什么是稠密重建？

稠密重建(Dense Reconstruction)是指从多张不同视角的图像中，恢复出场景的密集三维点云。相比稀疏重建(Sparse Reconstruction，如SfM)，稠密重建能产生每个像素级别的深度信息，从而得到更完整、更细致的三维模型。

### 1.2 openMVS的核心思想

openMVS使用**PatchMatch立体匹配算法**来估计每张图像的深度图(Depth Map)，然后将多个深度图融合成最终的稠密点云。

**核心流程**：
```
稀疏点云(SfM) → 视图选择 → 深度图估计(PatchMatch) → 深度图过滤 → 深度图融合 → 稠密点云
```

### 1.3 核心文件结构

```
openMVS/
├── apps/DensifyPointCloud/
│   └── DensifyPointCloud.cpp          # 主程序入口
├── libs/MVS/
│   ├── SceneDensify.cpp/.h            # 稠密重建主流程
│   ├── DepthMap.cpp/.h                # 深度图估计核心算法
│   ├── PatchMatchCUDA.cu/.h           # CUDA加速版本
│   └── Scene.cpp/.h                   # 场景数据结构
```

---

## 2. 整体架构

### 2.1 数据流图

```
输入: scene.mvs (包含相机参数、稀疏点云)
  ↓
[步骤1] 准备阶段
  - 加载场景数据
  - 调整图像分辨率
  - 估计ROI区域
  ↓
[步骤2] 视图选择 (SelectViews)
  - 为每个参考图像选择最佳邻居视图
  - 使用MRF优化全局视图配对
  ↓
[步骤3] 深度图估计 (EstimateDepthMap) - 核心算法
  - 初始化深度和法向
  - PatchMatch迭代优化
    * 空间传播 (Spatial Propagation)
    * 随机搜索 (Random Search)
    * 视图一致性检查 (View Consistency)
  - 多分辨率处理
  ↓
[步骤4] 深度图后处理 (FilterDepthMap)
  - 去除小斑点 (Remove Speckles)
  - 填补空洞 (Gap Interpolation)
  ↓
[步骤5] 深度图融合 (FuseDepthMaps)
  - 多视图深度一致性检查
  - 法向一致性检查
  - 加权平均融合
  ↓
输出: 稠密点云 (dense_pointcloud.ply)
```

### 2.2 核心数据结构

#### 2.2.1 Scene（场景）
```cpp
class Scene {
    ImageArr images;              // 所有图像
    PointCloud pointcloud;         // 稀疏/稠密点云
    PlatformArr platforms;         // 相机平台
    OBB3f obb;                    // 场景边界框(ROI)
    unsigned nMaxThreads;          // 最大线程数
};
```

#### 2.2.2 DepthData（深度数据）
```cpp
struct DepthData {
    ViewDataArr images;            // 参考图像+邻居图像
    ViewScoreArr neighbors;        // 邻居视图及其得分
    IndexArr points;               // 可见的稀疏点索引
    DepthMap depthMap;             // 深度图 (float)
    NormalMap normalMap;           // 法向图 (float3)
    ConfidenceMap confMap;         // 置信度图 (float)
    ViewsMap viewsMap;             // 视图ID图 (uint8_t x 4)
    float dMin, dMax;              // 深度范围
};
```

#### 2.2.3 DepthEstimator（深度估计器）
```cpp
struct DepthEstimator {
    // 配置
    const unsigned nIteration;     // 当前迭代次数
    const DepthData::ViewDataArr images; // 邻居图像
    const DepthData::ViewData& image0;   // 参考图像

    // Patch参数
    static const int nSizeHalfWindow = 4;  // patch半径
    static const int nSizeWindow = 9;       // patch大小 9x9
    static const int nTexels = 25;          // patch像素数

    // 当前处理的像素
    Point3 X0;                     // 3D光线方向
    ImageRef x0;                   // 2D像素坐标
    float normSq0;                 // patch归一化平方和

    // 输出
    DepthMap& depthMap0;           // 参考图像深度图
    NormalMap& normalMap0;         // 参考图像法向图
    ConfidenceMap& confMap0;       // 参考图像置信度图
};
```

---

## 3. 数学基础

### 3.1 相机模型

#### 3.1.1 针孔相机模型

世界坐标系中的3D点 **X** 投影到图像平面的过程：

```
x = K [R | t] X
```

其中：
- **K**: 内参矩阵 (3x3)
  ```
  K = [fx  0  cx]
      [0  fy  cy]
      [0   0   1]
  ```
  - fx, fy: 焦距（像素单位）
  - cx, cy: 主点坐标

- **R**: 旋转矩阵 (3x3)，表示相机朝向
- **t**: 平移向量 (3x1)，表示相机位置
- **C**: 相机中心 = -R^T * t

#### 3.1.2 坐标转换

```cpp
// 1. 图像坐标 → 相机坐标
Point3 TransformPointI2C(Point3(x, y, depth)) {
    return Point3(
        (x - K(0,2)) * depth / K(0,0),  // X
        (y - K(1,2)) * depth / K(1,1),  // Y
        depth                            // Z
    );
}

// 2. 相机坐标 → 世界坐标
Point3 TransformPointC2W(Point3 Xc) {
    return R.t() * Xc + C;
}

// 3. 世界坐标 → 相机坐标
Point3 TransformPointW2C(Point3 Xw) {
    return R * (Xw - C);
}

// 4. 相机坐标 → 图像坐标
Point2 TransformPointC2I(Point3 Xc) {
    return Point2(
        K(0,0) * Xc.x / Xc.z + K(0,2),
        K(1,1) * Xc.y / Xc.z + K(1,2)
    );
}
```

### 3.2 单应性矩阵（Homography）

#### 3.2.1 理论

对于场景中的平面，参考图像和目标图像之间的对应关系可以用单应性矩阵表示：

```
x' = H * x
```

其中 **H** 是 3x3 矩阵。

对于一个深度为 **d**，法向为 **n** 的平面片：

```
H = K' * R' * (I - (t' * n^T) / (n^T * X * d)) * R^T * K^-1
```

简化后（代码中的实现）：
```cpp
H = (H_l + H_m * (n^T / (n·X0 * depth))) * H_r
```

其中：
- H_l = K' * R' * R^T
- H_m = K' * R' * (C - C')
- H_r = K^-1

这些在代码中预计算并存储在 `ViewData::Init()` 中。

#### 3.2.2 代码实现

```cpp
// libs/MVS/DepthMap.cpp:414
Matrix3x3f ComputeHomographyMatrix(
    const DepthData::ViewData& img,
    Depth depth,
    const Normal& normal) const
{
    const Vec3 n(normal);
    return (img.Hl + img.Hm * (n.t() * INVERT(n.dot(X0) * depth))) * img.Hr;
}
```

### 3.3 归一化互相关 (NCC - Normalized Cross Correlation)

#### 3.3.1 原理

NCC用于衡量两个图像块的相似度，范围 [-1, 1]，值越大越相似。

对于参考图像的patch **I**，目标图像的patch **I'**：

```
NCC = (Σ(I - μ_I)(I' - μ_{I'})) / sqrt(Σ(I - μ_I)² * Σ(I' - μ_{I'})²)
```

其中：
- μ_I: patch I 的均值
- μ_{I'}: patch I' 的均值

openMVS使用 **1 - NCC** 作为cost，因此值越小越好。

#### 3.3.2 加权NCC (Weighted NCC)

openMVS默认使用加权版本(DENSE_NCC_WEIGHTED)，对patch中的每个像素赋予不同权重：

```cpp
// 权重函数
w(x) = exp(-color_weight - spatial_weight)

color_weight = (I(x) - I(center))² / (2 * sigma_color²)
spatial_weight = ||x||² / (2 * sigma_spatial²)
```

这样中心像素和颜色相近的像素权重更大。

#### 3.3.3 代码实现

```cpp
// libs/MVS/DepthMap.cpp:465
float ScorePixelImage(const DepthData::ViewData& image1,
                      Depth depth,
                      const Normal& normal)
{
    // 1. 计算单应性矩阵
    Matrix3x3f H = ComputeHomographyMatrix(image1, depth, normal);

    // 2. 采样目标图像patch
    int n = 0;
    float sum = 0, sumSq = 0, num = 0;
    for (int i = -nSizeHalfWindow; i <= nSizeHalfWindow; i += nSizeStep) {
        for (int j = -nSizeHalfWindow; j <= nSizeHalfWindow; j += nSizeStep) {
            Point3f X = H * Point3f(x0.x+j, x0.y+i, 1);
            Point2f pt(X.x/X.z, X.y/X.z);

            if (!image1.image.isInside(pt))
                return thRobust; // 超出边界，返回惩罚

            float v = image1.image.sample(pt); // 双线性插值
            const Weight::Pixel& pw = w.weights[n++];
            sum += v * pw.weight;
            sumSq += v * v * pw.weight;
            num += v * pw.tempWeight;
        }
    }

    // 3. 计算NCC
    float normSq1 = sumSq - sum*sum / w.sumWeights;
    float ncc = CLAMP(num / sqrt(normSq0 * normSq1), -1.f, 1.f);
    float score = 1.f - ncc;

    return score;
}
```

### 3.4 平滑性约束

#### 3.4.1 理论

为了使深度图局部平滑，openMVS在NCC得分中加入平滑项：

```
score_final = score_NCC * smooth_depth_factor * smooth_normal_factor
```

其中：
```
smooth_depth_factor = 1 - bonus * exp(-(depth_diff / depth)² / (2 * sigma²))
smooth_normal_factor = 1 - bonus * exp(-angle² / (2 * sigma²))
```

这鼓励相邻像素的深度和法向相似。

#### 3.4.2 平面平滑性 (Plane Smoothness)

openMVS使用更强的约束：不仅要求深度相似，还要求满足同一平面方程：

```
plane: n·X + d = 0
```

对于邻居像素的3D点 X_neighbor：
```
dist = |plane.distance(X_neighbor)|
smooth_factor = exp(-dist² / (2 * sigma²))
```

代码位置: `libs/MVS/DepthMap.cpp:522-533`

---

## 4. 视图选择算法

### 4.1 为什么需要视图选择？

在稠密重建中，每个参考图像需要选择若干邻居图像来进行立体匹配。选择不当会导致：
- **基线太小**：缺乏视差，深度不准确
- **基线太大**：遮挡增多，纹理变化大
- **视角太大**：外观变化大，匹配困难

### 4.2 局部视图选择 (SelectViews - 单图像)

对于每个参考图像，选择最佳邻居视图的标准：

#### 4.2.1 评分计算

对于候选邻居视图，计算综合得分：

```cpp
score = num_shared_points * area_overlap / baseline_quality
```

其中：
- **num_shared_points**: 共同可见的稀疏点数量
- **area_overlap**: 图像重叠区域面积
- **baseline_quality**: 基线质量（与最优角度的偏差）

#### 4.2.2 过滤条件

```cpp
// libs/MVS/Scene.cpp - FilterNeighborViews
bool FilterNeighborViews(
    ViewScoreArr& neighbors,
    float fMinArea,      // 最小重叠面积 (default: 0.05)
    float fMinScale,     // 最小尺度比例 (default: 0.2)
    float fMaxScale,     // 最大尺度比例 (default: 3.2)
    float fMinAngle,     // 最小视角 (default: 3°)
    float fMaxAngle,     // 最大视角 (default: 65°)
    unsigned nMaxViews   // 最大邻居数 (default: 8)
)
```

**过滤逻辑**：
1. 移除重叠面积 < fMinArea 的视图
2. 移除尺度比例不在 [fMinScale, fMaxScale] 的视图
3. 移除视角不在 [fMinAngle, fMaxAngle] 的视图
4. 按得分排序，保留前 nMaxViews 个

代码位置: `libs/MVS/SceneDensify.cpp:273-293`

### 4.3 全局视图选择 (SelectViews - MRF优化)

当 `nNumViews=1` 时，使用MRF(Markov Random Field)优化全局视图配对，确保：
1. 每个图像都有合适的配对
2. 整体覆盖场景完整
3. 避免同一对视图被多次使用

#### 4.3.1 MRF建模

**节点(Node)**：每个待处理的参考图像

**标签(Label)**：每个节点可选的邻居视图（包括"空"标签）

**一元能量(Unary Energy)**：
```cpp
E_unary(node, label) = avg_score / neighbor_score[label]
```
- 得分越高的邻居，能量越低

**二元能量(Pairwise Energy)**：
```cpp
if (node_i选择view_A && node_j也选择view_A) {
    E_pairwise = fSamePairwise;  // 高惩罚
} else {
    E_pairwise = fPairwiseMul * (area_ratio_i + area_ratio_j);
}
```
- 惩罚两个节点选择相同的视图
- 鼓励视图对覆盖更大区域

#### 4.3.2 求解算法

使用 **TRW-S (Tree-reweighted message passing)** 算法求解：

```cpp
// libs/MVS/SceneDensify.cpp:223-237
MRFEnergyType::Options options;
options.m_eps = OPTDENSE::fOptimizerEps;  // 收敛阈值
options.m_iterMax = OPTDENSE::nOptimizerMaxIters;  // 最大迭代次数

TypeGeneral::REAL energyVal, lowerBound;
energy->Minimize_TRW_S(options, lowerBound, energyVal);
```

代码位置: `libs/MVS/SceneDensify.cpp:150-267`

---

## 5. PatchMatch深度图估计

### 5.1 PatchMatch算法概述

PatchMatch是一种高效的近似最近邻搜索算法，由Barnes et al. 2009提出。openMVS将其应用于立体匹配：

**核心思想**：
1. **随机初始化**：为每个像素随机分配深度和法向
2. **空间传播**：尝试使用邻居像素的估计值
3. **随机搜索**：在当前估计附近随机采样
4. **迭代优化**：重复2-3步，逐渐收敛到最优解

### 5.2 算法详细流程

#### 5.2.1 初始化阶段

```cpp
// libs/MVS/SceneDensify.cpp:464-485
bool InitDepthMap(DepthData& depthData) {
    // 方案1: 如果有稀疏点，从稀疏点云初始化
    if (nMinViewsTrustPoint >= 2 && !depthData.points.empty()) {
        TriangulatePoints2DepthMap(
            image, pointcloud, depthData.points,
            depthData.depthMap, depthData.normalMap,
            depthData.dMin, depthData.dMax, bAddCorners
        );
    }
    // 方案2: 否则随机初始化
    else {
        depthData.depthMap.create(size);
        depthData.depthMap.memset(0);
        depthData.normalMap.create(size);
        depthData.dMin = 0.1f;
        depthData.dMax = 100.0f;
    }
}
```

**稀疏点初始化**：
1. 对可见的稀疏点进行Delaunay三角化
2. 对每个三角形，插值得到内部像素的深度和法向
3. 在稀疏点附近设置小窗口，直接赋值深度

代码位置: `libs/MVS/DepthMap.cpp:1100-1300` (TriangulatePoints2DepthMap)

#### 5.2.2 多分辨率策略

openMVS使用金字塔策略加速收敛：

```cpp
// 从低分辨率到高分辨率
for (scale = 1/4, 1/2, 1) {
    // 1. 缩放图像和深度图
    DepthData scaledData = ScaleDepthData(fullResData, scale);

    // 2. 上采样上一层的结果作为初始化
    if (scale > 1/4) {
        cv::resize(lowResDepthMap, scaledData.depthMap, size, INTER_LINEAR);
        cv::resize(lowResNormalMap, scaledData.normalMap, size, INTER_NEAREST);
    }

    // 3. 在当前分辨率运行PatchMatch
    for (iter = 0; iter < nEstimationIters; ++iter) {
        EstimateDepthMapIteration(scaledData, iter);
    }

    // 4. 保存结果用于下一层
    lowResDepthMap = scaledData.depthMap;
    lowResNormalMap = scaledData.normalMap;
}
```

参数控制：
- `nSubResolutionLevels`: 子分辨率层数 (default: 2)
- `nEstimationIters`: 每层的PatchMatch迭代次数 (default: 3)

代码位置: `libs/MVS/SceneDensify.cpp:651-769`

#### 5.2.3 PatchMatch迭代

每次迭代处理所有像素：

```cpp
// libs/MVS/DepthMap.cpp:630-912
void ProcessPixel(IDX idx) {
    // 1. 准备patch
    if (!PreparePixelPatch(x0) || !FillPixelPatch())
        return;

    // 2. 空间传播 (Spatial Propagation)
    for (neighbor in neighbors) {
        Depth d_neighbor = depthMap(neighbor.x);
        Normal n_neighbor = normalMap(neighbor.x);

        // 插值到当前像素位置
        d_new = InterpolatePixel(neighbor.x, d_neighbor, n_neighbor);
        CorrectNormal(n_neighbor);

        // 计算得分
        score_new = ScorePixel(d_new, n_neighbor);

        // 如果更好，更新
        if (score_new < score_current) {
            depth = d_new;
            normal = n_neighbor;
            score_current = score_new;
        }
    }

    // 3. 随机搜索 (Random Refinement)
    for (iter = 0; iter < nRandomIters; ++iter) {
        // 在当前估计附近随机采样
        d_rand = RandomDepth(depth - range, depth + range);
        n_rand = RandomNormal(normal, angle_range);

        score_rand = ScorePixel(d_rand, n_rand);

        if (score_rand < score_current) {
            depth = d_rand;
            normal = n_rand;
            score_current = score_rand;
            range *= 0.5;  // 缩小搜索范围
        }
    }
}
```

**关键点**：
- 迭代方向交替：奇数次从左上到右下，偶数次从右下到左上
- 邻居：上下左右4个方向
- 随机搜索范围逐渐缩小（自适应）

代码位置: `libs/MVS/DepthMap.cpp:630-912`

#### 5.2.4 视图一致性检查（几何一致性）

如果邻居视图也有深度图，检查几何一致性：

```cpp
// libs/MVS/DepthMap.cpp:535-551
// 当前像素在参考图像中的估计
Point3 X = camera0.TransformPointI2W(Point3(x0, depth0));

// 投影到邻居图像
Point3f X1 = camera1.ProjectPointP3(X);
Point2f x1(X1.x/X1.z, X1.y/X1.z);

// 读取邻居图像在x1位置的深度
Depth depth1 = depthMap1.sample(x1);

// 反投影回参考图像
Point2f x0_back = camera0.ProjectPoint(camera1.TransformPointI2W(Point3(x1, depth1)));

// 计算重投影误差
float dist = norm(x0 - x0_back);
float consistency = min(sqrt(dist*(dist+2)), 4.0);

// 加入代价
score += fEstimationGeometricWeight * consistency;
```

这个一致性项鼓励：
- 从参考图像估计的深度
- 投影到邻居图像后
- 邻居图像的深度再投影回参考图像
- 应该回到原来的位置

参数：
- `nEstimationGeometricIters`: 几何一致性迭代次数 (default: 2)
- `fEstimationGeometricWeight`: 几何一致性权重 (default: 0.1)

### 5.3 得分聚合策略

对于多个邻居视图，如何聚合它们的NCC得分？

openMVS支持4种策略 (DENSE_AGGNCC):

#### 5.3.1 DENSE_AGGNCC_NTH (第N个)
```cpp
sort(scores);
final_score = scores[N];  // N = num_views / 3
```
取排序后的第N个，容忍一些外点。

#### 5.3.2 DENSE_AGGNCC_MEAN (平均)
```cpp
final_score = mean(scores);
```
所有视图平等对待。

#### 5.3.3 DENSE_AGGNCC_MIN (最小)
```cpp
final_score = min(scores);
```
最保守，要求所有视图都匹配好。

#### 5.3.4 DENSE_AGGNCC_MINMEAN (最小+平均) **[默认]**
```cpp
sort(scores);
if (num_views <= 2)
    final_score = min(scores);
else
    final_score = mean(scores[0:1]);  // 最好的2个视图的平均
```
平衡鲁棒性和准确性。

代码位置: `libs/MVS/DepthMap.cpp:566-626`

---

## 6. 深度图后处理

### 6.1 去除小斑点 (Remove Speckles)

使用连通域分析，移除孤立的小区域：

```cpp
// libs/MVS/SceneDensify.cpp:810-900
bool RemoveSmallSegments(DepthData& depthData) {
    const float fDepthDiffThreshold = 0.007;  // 深度相似阈值
    const unsigned speckle_size = 100;        // 最小连通域大小

    // 对每个像素做BFS/DFS
    for (each pixel (u,v)) {
        if (visited(u,v)) continue;

        // 初始化连通域
        segment = {(u,v)};
        queue = {(u,v)};

        while (!queue.empty()) {
            (x,y) = queue.pop();
            depth_curr = depthMap(x,y);

            // 检查4邻域
            for (neighbor in {left, right, top, bottom}) {
                depth_neighbor = depthMap(neighbor);

                // 如果深度相似，加入连通域
                if (IsDepthSimilar(depth_curr, depth_neighbor, fDepthDiffThreshold)) {
                    segment.add(neighbor);
                    queue.push(neighbor);
                    visited(neighbor) = true;
                }
            }
        }

        // 如果连通域太小，移除
        if (segment.size() < speckle_size) {
            for (pixel in segment) {
                depthMap(pixel) = 0;
                normalMap(pixel) = (0,0,0);
                confMap(pixel) = 0;
            }
        }
    }
}
```

**深度相似性判断**：
```cpp
bool IsDepthSimilar(Depth d1, Depth d2, float threshold) {
    return abs(d1 - d2) / max(d1, d2) < threshold;
}
```

参数：
- `nSpeckleSize`: 最小连通域大小 (default: 100)

### 6.2 填补空洞 (Gap Interpolation)

对于小的空洞区域，使用线性插值填补：

```cpp
// libs/MVS/SceneDensify.cpp:904-1050
bool GapInterpolation(DepthData& depthData) {
    const unsigned nIpolGapSize = 7;  // 最大gap大小

    // 1. 行方向插值
    for (int v = 0; v < height; ++v) {
        gap_start = -1;
        for (int u = 0; u < width; ++u) {
            if (depthMap(v,u) == 0) {
                if (gap_start < 0) gap_start = u;
            } else {
                if (gap_start >= 0) {
                    gap_size = u - gap_start;

                    // 如果gap足够小且两端深度相似
                    if (gap_size <= nIpolGapSize &&
                        IsDepthSimilar(depthMap(v,gap_start-1), depthMap(v,u), threshold)) {

                        // 线性插值
                        depth_left = depthMap(v, gap_start-1);
                        depth_right = depthMap(v, u);
                        for (int k = gap_start; k < u; ++k) {
                            float t = (k - gap_start + 1) / (gap_size + 1);
                            depthMap(v,k) = depth_left * (1-t) + depth_right * t;

                            // 法向也插值
                            normal_left = normalMap(v, gap_start-1);
                            normal_right = normalMap(v, u);
                            normalMap(v,k) = normalize(normal_left * (1-t) + normal_right * t);
                        }
                    }
                    gap_start = -1;
                }
            }
        }
    }

    // 2. 列方向插值（同理）
    for (int u = 0; u < width; ++u) {
        // ... 类似逻辑
    }
}
```

参数：
- `nIpolGapSize`: 最大可填补gap大小 (default: 7)

### 6.3 深度图过滤 (FilterDepthMap)

基于多视图一致性过滤不可靠的深度：

```cpp
// libs/MVS/SceneDensify.cpp:1052-1200
bool FilterDepthMap(DepthData& depthData, const IIndexArr& idxNeighbors) {
    for (each pixel (x,y)) {
        depth = depthMap(x,y);
        if (depth == 0) continue;

        // 投影到邻居视图
        Point3 X = camera.TransformPointI2W(Point3(x,y,depth));

        int num_consistent = 0;
        for (neighbor_idx in idxNeighbors) {
            DepthData& neighborData = arrDepthData[neighbor_idx];

            // 投影到邻居图像
            Point3f pt = neighborCamera.ProjectPointP3(X);
            Point2f x_neighbor(pt.x/pt.z, pt.y/pt.z);

            if (!neighborData.depthMap.isInside(x_neighbor))
                continue;

            Depth depth_neighbor = neighborData.depthMap(x_neighbor);
            if (depth_neighbor == 0) continue;

            // 检查深度一致性
            if (IsDepthSimilar(pt.z, depth_neighbor, fDepthDiffThreshold)) {
                // 检查法向一致性
                Normal normal = normalMap(x,y);
                Normal normal_neighbor = neighborData.normalMap(x_neighbor);
                if (normal.dot(normal_neighbor) > cos(fNormalDiffThreshold)) {
                    num_consistent++;
                }
            }
        }

        // 如果一致性不足，移除
        if (num_consistent < nMinViewsFilter) {
            depthMap(x,y) = 0;
            normalMap(x,y) = (0,0,0);
            confMap(x,y) = 0;
        }
    }
}
```

参数：
- `nMinViewsFilter`: 最小一致视图数 (default: 2)
- `fDepthDiffThreshold`: 深度差异阈值 (default: 0.01)
- `fNormalDiffThreshold`: 法向差异阈值，度数 (default: 25)

---

## 7. 深度图融合

### 7.1 融合策略

openMVS支持两种融合模式：

#### 7.1.1 合并模式 (Merge) - nMinViewsFuse < 2
```cpp
// 直接合并所有深度图，不检查一致性
for (each depth_map) {
    for (each pixel with depth > 0) {
        Point3 X = camera.TransformPointI2W(Point3(x,y,depth));
        pointcloud.add(X);
    }
}
```
速度快，但可能包含噪声。

#### 7.1.2 融合模式 (Fuse) - nMinViewsFuse >= 2 **[推荐]**
```cpp
// 只保留多视图一致的点
for (reference_depth_map) {
    for (each pixel (x,y)) {
        depth = depthMap(x,y);
        if (depth == 0) continue;

        Point3 X = camera.TransformPointI2W(Point3(x,y,depth));

        // 检查所有邻居深度图
        views = [reference_view];
        weights = [conf];
        X_sum = X * conf;

        for (neighbor_depth_map) {
            Point3f pt = neighborCamera.ProjectPointP3(X);
            Point2f x_n(pt.x/pt.z, pt.y/pt.z);

            Depth depth_n = neighborDepthMap(x_n);
            if (depth_n == 0) continue;

            // 深度一致性
            if (!IsDepthSimilar(pt.z, depth_n, threshold))
                continue;

            // 法向一致性
            Normal normal = camera.R.t() * normalMap(x,y);
            Normal normal_n = neighborCamera.R.t() * neighborNormalMap(x_n);
            if (normal.dot(normal_n) < cos(fNormalDiffThreshold))
                continue;

            // 通过检查，加入
            views.add(neighbor_view);
            conf_n = Conf2Weight(neighborConfMap(x_n), depth_n);
            weights.add(conf_n);
            X_sum += neighborCamera.TransformPointI2W(Point3(x_n, depth_n)) * conf_n;
        }

        // 如果一致视图数足够，保留点
        if (views.size() >= nMinViewsFuse) {
            X_final = X_sum / sum(weights);  // 加权平均
            pointcloud.points.add(X_final);
            pointcloud.pointViews.add(views);
            pointcloud.pointWeights.add(weights);
        }
    }
}
```

代码位置: `libs/MVS/SceneDensify.cpp:1450-1646`

### 7.2 置信度加权

深度图的每个像素有一个置信度（从NCC得分转换）：

```cpp
// libs/MVS/SceneDensify.cpp:120-122
float Conf2Weight(float conf, Depth depth) {
    return 1.0f / (max(1.0f - conf, 0.03f) * depth * depth);
}
```

- conf: 置信度，范围 [0, 1]，越大越好
- depth: 深度值
- weight: 权重，用于加权平均

**解释**：
- 置信度高的点权重大
- 深度小的点权重大（近距离点更可靠）

### 7.3 颜色和法向估计

#### 7.3.1 颜色估计 (nEstimateColors)

- **0**: 不估计颜色
- **1**: 最后估计（从原始图像采样）
- **2**: 融合时估计（加权平均）**[默认]**

```cpp
if (nEstimateColors == 2) {
    Pixel32F color_sum(0,0,0);
    for (view in views) {
        ImageRef x_view = projectToView(X, view);
        Pixel color = images[view](x_view);
        color_sum += color * weights[view];
    }
    pointcloud.colors.add(color_sum / sum(weights));
}
```

#### 7.3.2 法向估计 (nEstimateNormals)

- **0**: 不估计法向
- **1**: 最后估计（K近邻拟合平面）
- **2**: 融合时估计（加权平均）**[默认]**

```cpp
if (nEstimateNormals == 2) {
    Normal normal_sum(0,0,0);
    for (view in views) {
        ImageRef x_view = projectToView(X, view);
        Normal normal_camera = normalMaps[view](x_view);
        Normal normal_world = cameras[view].R.t() * normal_camera;
        normal_sum += normal_world * weights[view];
    }
    pointcloud.normals.add(normalize(normal_sum));
}
```

---

## 8. 代码流程详解

### 8.1 主程序入口

```cpp
// apps/DensifyPointCloud/DensifyPointCloud.cpp:273-432
int main(int argc, char* argv[]) {
    // 1. 初始化
    Application app;
    app.Initialize(argc, argv);

    // 2. 加载场景
    Scene scene(OPT::nMaxThreads);
    scene.Load(OPT::strInputFileName);

    // 3. 估计ROI
    if (!scene.IsBounded())
        scene.EstimateROI(OPT::nEstimateROI, 1.1f);

    // 4. 稠密重建（核心）
    scene.DenseReconstruction(
        OPT::nFusionMode,      // 融合模式
        OPT::bCrop2ROI,        // 是否裁剪到ROI
        OPT::fBorderROI        // ROI边界扩展
    );

    // 5. 保存结果
    scene.pointcloud.Save(baseFileName + ".ply");
    scene.Save(baseFileName + ".mvs");

    return 0;
}
```

### 8.2 稠密重建主流程

```cpp
// libs/MVS/Scene.cpp:1683-1749
bool Scene::DenseReconstruction(int nFusionMode, bool bCrop2ROI, float fBorderROI) {
    // 创建数据结构
    DenseDepthMapData data(*this, nFusionMode);

    // === 阶段1: 计算深度图 ===
    if (!ComputeDepthMaps(data))
        return false;

    if (ABS(nFusionMode) == 1)
        return true;  // 只导出深度图，不融合

    // === 阶段2: 融合深度图 ===
    pointcloud.Release();

    if (OPTDENSE::nMinViewsFuse < 2) {
        // 合并模式
        data.depthMaps.MergeDepthMaps(pointcloud,
            OPTDENSE::nEstimateColors == 2,
            OPTDENSE::nEstimateNormals == 2);
    } else {
        // 融合模式（推荐）
        data.depthMaps.FuseDepthMaps(pointcloud,
            OPTDENSE::nEstimateColors == 2,
            OPTDENSE::nEstimateNormals == 2);
    }

    // === 阶段3: 后处理 ===
    // 裁剪到ROI
    if (bCrop2ROI && IsBounded()) {
        OBB3f ROI = (fBorderROI == 0) ? obb :
                    (fBorderROI > 0) ? obb.EnlargePercent(fBorderROI) :
                                       obb.Enlarge(-fBorderROI);
        pointcloud.RemovePointsOutside(ROI);
    }

    // 估计颜色（如果需要）
    if (pointcloud.colors.empty() && OPTDENSE::nEstimateColors == 1)
        EstimatePointColors(images, pointcloud);

    // 估计法向（如果需要）
    if (pointcloud.normals.empty() && OPTDENSE::nEstimateNormals == 1)
        EstimatePointNormals(images, pointcloud);

    // 删除深度图文件（如果指定）
    if (OPTDENSE::bRemoveDmaps) {
        for (depthData in data.depthMaps.arrDepthData) {
            File::deleteFile(ComposeDepthFilePath(depthData.GetID(), "dmap"));
        }
    }

    return true;
}
```

### 8.3 计算深度图流程

```cpp
// libs/MVS/Scene.cpp:1754-1920
bool Scene::ComputeDepthMaps(DenseDepthMapData& data) {
    // === 步骤1: 准备图像 ===
    for (image in images) {
        // 重载图像到合适分辨率
        image.ReloadImage(nMaxResolution);
        image.UpdateCamera(platforms);
        data.images.add(image);
    }

    // === 步骤2: 选择邻居视图 ===
    for (image in data.images) {
        DepthData& depthData = data.depthMaps.arrDepthData[image];
        data.depthMaps.SelectViews(depthData);
    }

    // 全局优化（如果nNumViews==1）
    if (OPTDENSE::nNumViews == 1) {
        data.depthMaps.SelectViews(data.images, imagesMap, data.neighborsMap);
    }

    // === 步骤3: 初始化CUDA（如果可用）===
    #ifdef _USE_CUDA
    if (CUDA::desiredDeviceID >= -1) {
        data.depthMaps.pmCUDA = new PatchMatchCUDA(CUDA::desiredDeviceID);
    }
    #endif

    // === 步骤4: 多线程估计深度图 ===
    data.idxImage = 0;
    data.events.AddEvent(new EVTProcessImage(0));

    // 启动工作线程
    Thread threads[nMaxThreads];
    for (thread in threads) {
        thread.start(DenseReconstructionEstimateTmp, &data);
    }

    // 等待完成
    for (thread in threads) {
        thread.join();
    }

    return true;
}
```

### 8.4 深度图估计线程

```cpp
// libs/MVS/SceneDensify.cpp:2100-2300
void* DenseReconstructionEstimateTmp(void* arg) {
    DenseDepthMapData& data = *(DenseDepthMapData*)arg;

    while (true) {
        // 从事件队列获取任务
        Event* event = data.events.GetEvent();

        switch (event->GetID()) {
            case EVT_PROCESSIMAGE: {
                // 处理新图像
                IIndex idxImage = ((EVTProcessImage*)event)->idxImage;

                // === 初始化 ===
                DepthData& depthData = data.depthMaps.arrDepthData[idxImage];
                data.depthMaps.InitViews(depthData, NO_ID, 0, true, 0);

                // === 估计深度图 ===
                data.events.AddEvent(new EVTEstimateDepthMap(idxImage));

                // 发送下一个图像任务
                IIndex idxImageNext = Thread::safeInc(data.idxImage);
                if (idxImageNext < data.images.size()) {
                    data.events.AddEvent(new EVTProcessImage(idxImageNext));
                }
                break;
            }

            case EVT_ESTIMATEDEPTHMAP: {
                IIndex idxImage = ((EVTEstimateDepthMap*)event)->idxImage;

                // PatchMatch估计
                data.depthMaps.EstimateDepthMap(idxImage, -1);

                // 几何一致性迭代
                for (int iter = 0; iter < OPTDENSE::nEstimationGeometricIters; ++iter) {
                    data.depthMaps.EstimateDepthMap(idxImage, iter);
                }

                // 后处理
                if (OPTDENSE::nOptimize & OPTDENSE::OPTIMIZE) {
                    data.events.AddEvent(new EVTOptimizeDepthMap(idxImage));
                } else {
                    data.events.AddEvent(new EVTSaveDepthMap(idxImage));
                }
                break;
            }

            case EVT_OPTIMIZEDEPTHMAP: {
                IIndex idxImage = ((EVTOptimizeDepthMap*)event)->idxImage;
                DepthData& depthData = data.depthMaps.arrDepthData[idxImage];

                // 去除小斑点
                if (OPTDENSE::nOptimize & OPTDENSE::REMOVE_SPECKLES) {
                    data.depthMaps.RemoveSmallSegments(depthData);
                }

                // 填补空洞
                if (OPTDENSE::nOptimize & OPTDENSE::FILL_GAPS) {
                    data.depthMaps.GapInterpolation(depthData);
                }

                // 过滤深度图
                if (OPTDENSE::nOptimize & OPTDENSE::ADJUST_FILTER) {
                    data.events.AddEvent(new EVTFilterDepthMap(idxImage));
                } else {
                    data.events.AddEvent(new EVTSaveDepthMap(idxImage));
                }
                break;
            }

            case EVT_FILTERDEPTHMAP: {
                IIndex idxImage = ((EVTFilterDepthMap*)event)->idxImage;
                DepthData& depthData = data.depthMaps.arrDepthData[idxImage];

                // 基于邻居深度图过滤
                data.depthMaps.FilterDepthMap(depthData, depthData.neighbors);

                data.events.AddEvent(new EVTSaveDepthMap(idxImage));
                break;
            }

            case EVT_SAVEDEPTHMAP: {
                IIndex idxImage = ((EVTSaveDepthMap*)event)->idxImage;
                DepthData& depthData = data.depthMaps.arrDepthData[idxImage];

                // 保存深度图到文件
                depthData.Save(ComposeDepthFilePath(idxImage, "dmap"));
                depthData.ReleaseImages();

                data.progress->operator++();
                break;
            }

            case EVT_CLOSE: {
                return NULL;  // 退出线程
            }
        }

        delete event;
    }
}
```

### 8.5 关键函数调用链

```
main()
  └─ Scene::DenseReconstruction()
      ├─ Scene::ComputeDepthMaps()
      │   ├─ 准备图像
      │   ├─ DepthMapsData::SelectViews()  [视图选择]
      │   └─ 启动线程 → DenseReconstructionEstimateTmp()
      │       └─ DepthMapsData::EstimateDepthMap()  [PatchMatch]
      │           ├─ InitViews() [初始化]
      │           ├─ ScoreDepthMapTmp() [初始得分]
      │           ├─ EstimateDepthMapTmp() [PatchMatch迭代]
      │           │   └─ DepthEstimator::ProcessPixel()
      │           │       ├─ 空间传播
      │           │       ├─ 随机搜索
      │           │       └─ ScorePixel()
      │           │           └─ ScorePixelImage()
      │           │               └─ ComputeHomographyMatrix()
      │           └─ EndDepthMapTmp() [结束处理]
      │
      └─ DepthMapsData::FuseDepthMaps()  [融合]
          ├─ 多视图一致性检查
          ├─ 加权平均
          └─ 生成点云
```

---

## 9. 关键参数说明

### 9.1 图像分辨率控制

| 参数 | 说明 | 默认值 | 影响 |
|------|------|--------|------|
| `--resolution-level` | 图像缩放倍数 | 1 | 越大越快，但精度下降 |
| `--max-resolution` | 最大分辨率 | 2560 | 限制图像最大尺寸 |
| `--min-resolution` | 最小分辨率 | 640 | 限制图像最小尺寸 |
| `--sub-resolution-levels` | 子分辨率层数 | 2 | 金字塔层数，加速收敛 |

**建议**：
- 高分辨率图像（>4K）：设置 `--resolution-level 2`
- 中等分辨率（1080p）：使用默认值
- 低分辨率（<720p）：设置 `--resolution-level 0`

### 9.2 视图选择参数

| 参数 | 说明 | 默认值 | 影响 |
|------|------|--------|------|
| `--number-views` | 邻居视图数 | 5 (CPU) / 8 (CUDA) | 越多越准确，但越慢 |
| `--number-views-fuse` | 融合最小视图数 | 3 | 过滤阈值，越大越严格 |

**建议**：
- 高质量重建：`--number-views 8 --number-views-fuse 3`
- 快速重建：`--number-views 3 --number-views-fuse 2`
- 数据集图像少：`--number-views 0`（使用所有可用视图）

### 9.3 PatchMatch迭代参数

| 参数 | 说明 | 默认值 | 影响 |
|------|------|--------|------|
| `--iters` | PatchMatch迭代次数 | 3 (CPU) / 4 (CUDA) | 越多越准确，越慢 |
| `--geometric-iters` | 几何一致性迭代 | 2 | 提高精度，去除外点 |

**建议**：
- 纹理丰富场景：`--iters 3`
- 纹理稀疏场景：`--iters 5 --geometric-iters 3`
- 快速预览：`--iters 2 --geometric-iters 0`

### 9.4 深度图后处理

| 参数 | 说明 | 默认值 | 影响 |
|------|------|--------|------|
| `--postprocess-dmaps` | 后处理标志 | 7 | 二进制标志（见下） |
| `--filter-point-cloud` | 点云过滤阈值 | 0 | >0启用可见性过滤 |

`--postprocess-dmaps` 标志位：
- **1** (REMOVE_SPECKLES): 去除小斑点
- **2** (FILL_GAPS): 填补空洞
- **4** (ADJUST_FILTER): 基于邻居深度图过滤
- **7** (默认): 1+2+4 全部启用

**建议**：
- 标准场景：`--postprocess-dmaps 7`（全部启用）
- 快速模式：`--postprocess-dmaps 1`（仅去除斑点）
- 无后处理：`--postprocess-dmaps 0`

### 9.5 融合参数

| 参数 | 说明 | 默认值 | 影响 |
|------|------|--------|------|
| `--fusion-mode` | 融合模式 | 0 | -2:disparity, -1:export only, 0:fuse, 1:export depthmaps |
| `--estimate-colors` | 估计颜色 | 2 | 0:no, 1:final, 2:during fusion |
| `--estimate-normals` | 估计法向 | 2 | 0:no, 1:final, 2:during fusion |

**建议**：
- 标准重建：`--fusion-mode 0 --estimate-colors 2 --estimate-normals 2`
- 仅导出深度图：`--fusion-mode 1`
- 不需要颜色：`--estimate-colors 0`

### 9.6 高级参数

| 参数 | 说明 | 默认值 | 范围 |
|------|------|--------|------|
| `fNCCThresholdKeep` | NCC阈值 | 0.9 | [0.5, 1.0] |
| `fDepthDiffThreshold` | 深度差异阈值 | 0.01 | [0.005, 0.05] |
| `fNormalDiffThreshold` | 法向差异阈值（度） | 25 | [10, 45] |
| `fPairwiseMul` | 平滑权重 | 0.3 | [0.1, 1.0] |

可通过配置文件调整：
```ini
# dense_config.ini
[Dense]
fNCCThresholdKeep = 0.85
fDepthDiffThreshold = 0.015
fNormalDiffThreshold = 30
```

使用：
```bash
DensifyPointCloud --dense-config-file dense_config.ini scene.mvs
```

---

## 10. 完整示例

### 10.1 标准重建流程

```bash
# 1. 从COLMAP导入场景
InterfaceCOLMAP -i path/to/colmap -o scene.mvs

# 2. 稠密重建
DensifyPointCloud scene.mvs \
    --resolution-level 1 \
    --number-views 5 \
    --number-views-fuse 3 \
    --iters 3 \
    --geometric-iters 2 \
    --postprocess-dmaps 7 \
    --estimate-colors 2 \
    --estimate-normals 2 \
    -o dense.mvs

# 输出：dense.ply（稠密点云）

# 3. 网格重建
ReconstructMesh dense.mvs -o mesh.mvs

# 4. 纹理映射
TextureMesh mesh.mvs -o textured_mesh.mvs

# 输出：textured_mesh.ply 和 textured_mesh.png
```

### 10.2 高质量重建

```bash
DensifyPointCloud scene.mvs \
    --resolution-level 0 \          # 全分辨率
    --max-resolution 4096 \          # 允许更大图像
    --number-views 8 \               # 更多视图
    --number-views-fuse 4 \          # 更严格融合
    --iters 5 \                      # 更多迭代
    --geometric-iters 3 \            # 更多几何一致性检查
    --sub-resolution-levels 3 \      # 更多金字塔层
    --postprocess-dmaps 7 \
    --filter-point-cloud 2 \         # 启用可见性过滤
    -o dense_highquality.mvs
```

### 10.3 快速预览

```bash
DensifyPointCloud scene.mvs \
    --resolution-level 2 \           # 1/4分辨率
    --number-views 3 \               # 少量视图
    --number-views-fuse 2 \
    --iters 2 \                      # 少量迭代
    --geometric-iters 0 \            # 跳过几何一致性
    --sub-resolution-levels 1 \
    --postprocess-dmaps 1 \          # 仅去除斑点
    -o dense_preview.mvs
```

### 10.4 处理大场景

```bash
# 1. 分割场景
DensifyPointCloud scene.mvs \
    --sub-scene-area 100 \           # 子场景最大面积
    -o chunks/

# 输出：chunks/scene_0000.mvs, chunks/scene_0001.mvs, ...

# 2. 并行处理每个子场景
for chunk in chunks/scene_*.mvs; do
    DensifyPointCloud $chunk -o ${chunk%.mvs}_dense.mvs &
done
wait

# 3. 合并结果（手动或使用工具）
```

### 10.5 使用CUDA加速

```bash
# 自动选择最佳GPU
DensifyPointCloud scene.mvs \
    --cuda-device -1 \               # -1:自动, -2:CPU, >=0:GPU ID
    --number-views 8 \               # CUDA支持更多视图
    --iters 4 \
    -o dense.mvs
```

### 10.6 导出深度图

```bash
# 仅导出深度图，不融合
DensifyPointCloud scene.mvs \
    --fusion-mode 1 \                # 1:仅导出深度图
    -o depthmap_export.mvs

# 深度图保存在：depth0000.dmap, depth0001.dmap, ...

# 可视化深度图（生成PNG）
# 代码会自动生成 depth0000.png 等文件（如果verbosity足够高）
```

### 10.7 使用Mask（掩码）

```bash
# 准备mask图像（16位PNG，0表示忽略）
# 文件命名：image_name.mask.png

DensifyPointCloud scene.mvs \
    --mask-path path/to/masks/ \     # mask文件夹
    --ignore-mask-label 0 \          # 标签0表示忽略
    -o dense.mvs
```

### 10.8 指定ROI（感兴趣区域）

```bash
# 方法1: 自动估计ROI
DensifyPointCloud scene.mvs \
    --estimate-roi 2 \               # 0:禁用, 1:启用, 2:自适应
    --crop-to-roi true \             # 裁剪到ROI
    --roi-border 0.1 \               # ROI边界扩展10%
    -o dense.mvs

# 方法2: 手动指定ROI（需要先导出）
DensifyPointCloud scene.mvs \
    --export-roi-file roi.txt

# 编辑roi.txt，然后导入
DensifyPointCloud scene.mvs \
    --import-roi-file roi.txt \
    -o dense.mvs
```

---

## 总结

openMVS的稠密重建算法是一个高度优化、功能完整的系统，核心特点：

1. **PatchMatch算法**：高效的立体匹配，适合大规模场景
2. **多分辨率策略**：金字塔加速收敛
3. **几何一致性**：多视图交叉验证，提高鲁棒性
4. **完善的后处理**：去噪、填洞、过滤
5. **灵活的参数**：适应不同场景需求
6. **CUDA加速**：支持GPU加速

**核心公式回顾**：

1. 单应性矩阵：`H = (H_l + H_m * (n^T / (n·X0 * depth))) * H_r`
2. NCC得分：`score = 1 - NCC`
3. 平滑项：`score *= (1 - bonus * exp(-diff² / (2*σ²)))`
4. 融合权重：`w = 1 / (max(1-conf, 0.03) * depth²)`

**关键代码位置**：
- 主流程：`apps/DensifyPointCloud/DensifyPointCloud.cpp:273-432`
- PatchMatch：`libs/MVS/DepthMap.cpp:630-912`
- 视图选择：`libs/MVS/SceneDensify.cpp:150-293`
- 融合：`libs/MVS/SceneDensify.cpp:1450-1646`

---

**祝你研究顺利！如有疑问，请参考代码注释或联系openMVS社区。**
