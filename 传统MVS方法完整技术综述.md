# 传统MVS方法完整技术综述

本文档全面梳理传统多视图立体匹配(MVS)中的各个模块和技术方案，涵盖COLMAP、PMVS、Gipuma、SGM、openMVS、ACMMP等主流方法。

---

## 目录

**第一部分：核心表示与数据结构**
1. [深度假设表示](#1-深度假设表示)
2. [代价体积构建](#2-代价体积构建)

**第二部分：初始化与采样**
3. [深度初始化策略](#3-深度初始化策略)
4. [采样与假设生成](#4-采样与假设生成)

**第三部分：匹配与代价计算**
5. [匹配代价度量](#5-匹配代价度量)
6. [代价聚合方法](#6-代价聚合方法)

**第四部分：优化与传播**
7. [传播策略](#7-传播策略)
8. [全局优化方法](#8-全局优化方法)

**第五部分：多视图处理**
9. [视图选择](#9-视图选择)
10. [遮挡处理](#10-遮挡处理)
11. [几何一致性](#11-几何一致性)

**第六部分：后处理与增强**
12. [置信度估计](#12-置信度估计)
13. [深度图滤波](#13-深度图滤波)
14. [纹理分析与处理](#14-纹理分析与处理)

**第七部分：多尺度与融合**
15. [多分辨率策略](#15-多分辨率策略)
16. [深度图融合](#16-深度图融合)

**第八部分：主流系统对比**
17. [各系统技术选型](#17-各系统技术选型)

---

## 第一部分：核心表示与数据结构

### 1. 深度假设表示

深度假设的表示方式直接影响算法的搜索空间和优化难度。

#### 1.1 逐像素深度值 (Per-pixel Depth)

**代表方法**: SGM, 基础立体匹配

**表示**: 每个像素存储一个深度值 d
```
D(u,v) = d ∈ [d_min, d_max]
```

**优点**: 简单，内存效率高
**缺点**: 无法表示表面朝向，倾斜表面误差大

#### 1.2 视差表示 (Disparity)

**代表方法**: 双目立体匹配, SGM

**表示**: 视差 = 基线 × 焦距 / 深度
```
disparity = b × f / d
```

**优点**:
- 搜索空间均匀（远处物体视差变化小）
- 整数视差便于优化

**缺点**: 仅适用于校正后的立体对

#### 1.3 逆深度表示 (Inverse Depth)

**代表方法**: LSD-SLAM, DSO

**表示**:
```
ρ = 1/d
```

**优点**:
- 远距离物体的不确定性自然建模
- 无穷远点可表示(ρ=0)
- 高斯分布假设更合理

**数学原理**:
```
深度不确定性: σ_d ∝ d²
逆深度不确定性: σ_ρ ≈ 常数
```

#### 1.4 平面假设 (Plane Hypothesis)

**代表方法**: PatchMatch Stereo, openMVS, ACMMP, COLMAP

**表示**: 每个像素存储深度d和法向量n
```
Plane(u,v) = (d, n) 或 (n, d) 其中 n·X + d = 0

或者使用齐次表示:
π = (a, b, c, d) 其中 ax + by + cz + d = 0
```

**openMVS表示**:
```cpp
struct PixelEstimate {
    Depth depth;    // 深度值
    Normal normal;  // 3D法向量 (nx, ny, nz)
};
```

**ACMMP表示**:
```cuda
float4 plane_hypothesis;  // (nx, ny, nz, d)
// 其中 d = -n·X 是到原点的距离
```

**优点**:
- 可表示倾斜表面
- 支持单应性变换
- 邻域传播更准确

**缺点**:
- 搜索空间增大(3DOF → 4DOF)
- 需要更多迭代

#### 1.5 3D点+法向量 (Surfels)

**代表方法**: PMVS, Point-based MVS

**表示**:
```cpp
struct Surfel {
    Point3 position;  // 3D位置
    Normal normal;    // 表面法向量
    float radius;     // 点的尺寸
    Color color;      // 颜色
};
```

**PMVS的Patch表示**:
```cpp
struct Patch {
    Vec3 center;      // 3D中心点
    Vec3 normal;      // 法向量
    vector<int> visible_images;  // 可见图像列表
    vector<int> consistent_images;  // 一致性图像
    float score;      // NCC得分
};
```

---

### 2. 代价体积构建

代价体积是立体匹配的核心数据结构。

#### 2.1 3D代价体积 (Cost Volume)

**代表方法**: SGM, 传统立体匹配

**结构**: C(u, v, d) - 高×宽×深度采样数
```
维度: H × W × D
每个位置存储该像素在深度d处的匹配代价
```

**构建**:
```cpp
for (int d = 0; d < num_depths; d++) {
    float depth = d_min + d * depth_step;
    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            Point2f src_pt = Warp(u, v, depth, H);
            cost_volume[v][u][d] = MatchingCost(ref[v][u], src[src_pt]);
        }
    }
}
```

#### 2.2 4D代价体积 (考虑法向量)

**代表方法**: Slanted plane stereo

**结构**: C(u, v, d, θ, φ) 或 离散化的法向量
```
维度: H × W × D × N_normals
```

**优点**: 可搜索最优法向量
**缺点**: 内存和计算量巨大

#### 2.3 代价体积正则化

**目的**: 增强代价体积的平滑性和鲁棒性

**方法1: 3D卷积滤波**
```cpp
// 3D高斯滤波
for (int d = 0; d < D; d++) {
    GaussianBlur3D(cost_volume, sigma_spatial, sigma_depth);
}
```

**方法2: Soft Argmin**
```
d* = Σ_d d × softmax(-C(u,v,d) / τ)
```

**方法3: Winner-Take-All (WTA)**
```
d* = argmin_d C(u,v,d)
```

---

## 第二部分：初始化与采样

### 3. 深度初始化策略

#### 3.1 完全随机初始化

**代表方法**: 原始PatchMatch

```cpp
void RandomInit(DepthMap& depth, NormalMap& normal) {
    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            depth(v,u) = RandomUniform(d_min, d_max);
            normal(v,u) = RandomUnitVector();
            // 确保法向量朝向相机
            if (normal(v,u).dot(viewDir(u,v)) > 0)
                normal(v,u) = -normal(v,u);
        }
    }
}
```

**ACMMP随机初始化** (`ACMMP.cu`):
```cuda
float4 GenerateRandomPlaneHypothesis(Camera camera, int2 p,
    curandState *rand_state, float depth_min, float depth_max)
{
    // 随机深度
    float depth = curand_uniform(rand_state) * (depth_max - depth_min) + depth_min;

    // 随机法向量（Marsaglia方法生成均匀球面分布）
    float q1 = 1.0f, q2 = 1.0f, s = 2.0f;
    while (s >= 1.0f) {
        q1 = 2.0f * curand_uniform(rand_state) - 1.0f;
        q2 = 2.0f * curand_uniform(rand_state) - 1.0f;
        s = q1 * q1 + q2 * q2;
    }
    float sq = sqrt(1.0f - s);
    normal.x = 2.0f * q1 * sq;
    normal.y = 2.0f * q2 * sq;
    normal.z = 1.0f - 2.0f * s;

    return plane_hypothesis;
}
```

#### 3.2 稀疏点初始化

**代表方法**: openMVS, COLMAP

**流程**:
1. 使用SfM稀疏点
2. Delaunay三角化
3. 三角形内插值深度

**openMVS实现** (`DepthMap.cpp`):
```cpp
// 使用CGAL进行Delaunay三角化
std::pair<float,float> TriangulatePointsDelaunay(
    const ViewData& image,
    const PointCloud& pointcloud,
    const IndexArr& points,
    Mesh& mesh)
{
    typedef CGAL::Delaunay_triangulation_2<...> Delaunay;
    Delaunay delaunay;

    for (uint32_t idx: points) {
        // 投影3D点到图像
        Point3f pt = image.camera.ProjectPointP3(pointcloud.points[idx]);
        Point3f x(pt.x/pt.z, pt.y/pt.z, pt.z);

        // 插入Delaunay三角化
        delaunay.insert(CPoint(x.x, x.y))->info() = mesh.vertices.size();
        mesh.vertices.emplace_back(image.camera.TransformPointI2C(x));
    }

    // 对每个像素，找到所在三角形，插值深度
    for (像素 in 图像) {
        Face face = delaunay.locate(像素);
        depth = BarycentricInterpolation(face, 像素);
    }
}
```

#### 3.3 SGM初始化

**代表方法**: 某些混合方法

**流程**:
1. 运行SGM获得初始深度
2. 作为PatchMatch的初始化
3. 继续优化

```cpp
void SGMInit(DepthMap& depth) {
    // 1. 构建代价体积
    CostVolume cost = BuildCostVolume(ref, src);

    // 2. SGM路径聚合
    for (int r = 0; r < 8; r++) {  // 8个方向
        AggregatePathCost(cost, r);
    }

    // 3. WTA选择最优深度
    for (int v = 0; v < H; v++) {
        for (int u = 0; u < W; u++) {
            depth(v,u) = argmin(cost[v][u]);
        }
    }
}
```

#### 3.4 分层初始化 (Hierarchical)

**代表方法**: ACMMP, Gipuma

**流程**:
```
层级3 (1/8分辨率): 随机初始化 + PatchMatch
    ↓ 上采样
层级2 (1/4分辨率): 继承上层 + PatchMatch
    ↓ 上采样
层级1 (1/2分辨率): 继承上层 + PatchMatch
    ↓ 上采样
层级0 (全分辨率): 继承上层 + PatchMatch
```

#### 3.5 时序初始化 (视频MVS)

**代表方法**: DTAM, DynamicFusion

```cpp
void TemporalInit(DepthMap& depth_t, const DepthMap& depth_t1,
                  const Pose& T_t, const Pose& T_t1) {
    // 将上一帧深度图warp到当前帧
    for (int v = 0; v < H; v++) {
        for (int u = 0; u < W; u++) {
            Point3 X = Unproject(u, v, depth_t1(v,u), K, T_t1);
            Point2 pt = Project(X, K, T_t);
            if (IsValid(pt))
                depth_t(v,u) = (T_t.R * X + T_t.t).z;
        }
    }
}
```

---

### 4. 采样与假设生成

#### 4.1 均匀采样

**深度均匀采样**:
```cpp
vector<float> depths;
for (int i = 0; i < num_samples; i++) {
    depths.push_back(d_min + i * (d_max - d_min) / (num_samples - 1));
}
```

**逆深度均匀采样**（推荐）:
```cpp
vector<float> depths;
float inv_d_min = 1.0f / d_max;
float inv_d_max = 1.0f / d_min;
for (int i = 0; i < num_samples; i++) {
    float inv_d = inv_d_min + i * (inv_d_max - inv_d_min) / (num_samples - 1);
    depths.push_back(1.0f / inv_d);
}
```

#### 4.2 自适应采样

**基于梯度的采样**:
```cpp
// 在深度变化剧烈的区域采样更密
vector<float> AdaptiveSample(const CostCurve& cost) {
    vector<float> samples;
    for (int i = 1; i < cost.size(); i++) {
        float gradient = abs(cost[i] - cost[i-1]);
        if (gradient > threshold)
            samples.push_back((i-0.5) * depth_step);
    }
    return samples;
}
```

#### 4.3 重要性采样

**ACMMP的概率采样**:
```cuda
// 根据代价计算采样概率
TransformPDFToCDF(sampling_probs, num_samples);

// 按概率采样
float rand_prob = curand_uniform(rand_state);
for (int i = 0; i < num_samples; i++) {
    if (sampling_probs[i] > rand_prob) {
        selected_sample = i;
        break;
    }
}
```

---

## 第三部分：匹配与代价计算

### 5. 匹配代价度量

#### 5.1 绝对差之和 (SAD)

```cpp
float SAD(const Patch& ref, const Patch& src) {
    float sum = 0;
    for (int i = 0; i < patch_size * patch_size; i++) {
        sum += abs(ref[i] - src[i]);
    }
    return sum;
}
```
**特点**: 计算快，对光照变化敏感

#### 5.2 平方差之和 (SSD)

```cpp
float SSD(const Patch& ref, const Patch& src) {
    float sum = 0;
    for (int i = 0; i < patch_size * patch_size; i++) {
        float diff = ref[i] - src[i];
        sum += diff * diff;
    }
    return sum;
}
```
**特点**: 对大误差敏感

#### 5.3 归一化互相关 (NCC)

```cpp
float NCC(const Patch& ref, const Patch& src) {
    float mean_ref = Mean(ref);
    float mean_src = Mean(src);
    float var_ref = 0, var_src = 0, covar = 0;

    for (int i = 0; i < n; i++) {
        float dr = ref[i] - mean_ref;
        float ds = src[i] - mean_src;
        var_ref += dr * dr;
        var_src += ds * ds;
        covar += dr * ds;
    }

    return covar / sqrt(var_ref * var_src);  // 范围[-1, 1]
}
// 代价 = 1 - NCC, 范围[0, 2]
```
**特点**: 光照不变，计算量中等

#### 5.4 零均值归一化互相关 (ZNCC)

```cpp
float ZNCC(const Patch& ref, const Patch& src) {
    // 与NCC相同，但先减去均值
    // 更好地处理光照偏移
}
```

#### 5.5 Census变换

**原理**: 比较中心像素与邻域的大小关系
```cpp
uint64_t CensusTransform(const Image& img, int cx, int cy, int radius) {
    uint64_t census = 0;
    float center = img(cy, cx);
    int bit = 0;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            if (dx == 0 && dy == 0) continue;
            if (img(cy+dy, cx+dx) < center)
                census |= (1ULL << bit);
            bit++;
        }
    }
    return census;
}

// 匹配代价 = Hamming距离
float CensusCost(uint64_t c1, uint64_t c2) {
    return __builtin_popcountll(c1 ^ c2);
}
```
**特点**: 光照不变，边缘保持好

#### 5.6 加权NCC (openMVS)

```cpp
float WeightedNCC(const Patch& ref, const Patch& src, const Image& ref_img) {
    float center_color = ref_img(cy, cx);
    float sum_weights = 0;
    float sum_ref = 0, sum_src = 0;

    for (int i = 0; i < n; i++) {
        // 双边权重 = 空间高斯 × 颜色高斯
        float spatial_w = exp(-dist[i]² / (2 * σ_s²));
        float color_w = exp(-|ref[i] - center_color|² / (2 * σ_c²));
        float w = spatial_w * color_w;

        sum_weights += w;
        sum_ref += w * ref[i];
        sum_src += w * src[i];
        // ... 继续计算加权NCC
    }
}
```

#### 5.7 双边NCC (ACMMP)

与加权NCC类似，但权重计算略有不同：
```cuda
float weight = exp(-spatial_dist / (2 * σ_s²) - color_dist / (2 * σ_c²));
```

#### 5.8 多通道匹配代价

**RGB空间**:
```cpp
float ColorNCC(const ColorPatch& ref, const ColorPatch& src) {
    float ncc_r = NCC(ref.r, src.r);
    float ncc_g = NCC(ref.g, src.g);
    float ncc_b = NCC(ref.b, src.b);
    return (ncc_r + ncc_g + ncc_b) / 3;
}
```

**Lab空间**:
```cpp
// 对光照变化更鲁棒
ColorPatch ref_lab = RGB2Lab(ref);
ColorPatch src_lab = RGB2Lab(src);
```

#### 5.9 梯度匹配代价

```cpp
float GradientCost(const Patch& ref, const Patch& src) {
    // 计算梯度图像
    Patch grad_ref_x = Sobel_X(ref);
    Patch grad_ref_y = Sobel_Y(ref);
    Patch grad_src_x = Sobel_X(src);
    Patch grad_src_y = Sobel_Y(src);

    // 梯度方向直方图匹配
    return HistogramDiff(grad_ref, grad_src);
}
```

---

### 6. 代价聚合方法

#### 6.1 固定窗口聚合

```cpp
float BoxAggregation(const CostVolume& cost, int u, int v, int d, int win_size) {
    float sum = 0;
    int half = win_size / 2;
    for (int dy = -half; dy <= half; dy++) {
        for (int dx = -half; dx <= half; dx++) {
            sum += cost(v+dy, u+dx, d);
        }
    }
    return sum / (win_size * win_size);
}
```
**问题**: 边缘模糊

#### 6.2 自适应权重聚合

**代表方法**: Adaptive Support Weight (ASW)

```cpp
float AdaptiveAggregation(const CostVolume& cost, const Image& img,
                          int u, int v, int d, int win_size) {
    float sum = 0, weight_sum = 0;
    Color center = img(v, u);

    for (int dy = -half; dy <= half; dy++) {
        for (int dx = -half; dx <= half; dx++) {
            Color neighbor = img(v+dy, u+dx);

            // 颜色相似性权重
            float color_diff = ColorDistance(center, neighbor);
            float weight = exp(-color_diff / σ_c);

            // 空间权重
            weight *= exp(-(dx*dx + dy*dy) / σ_s);

            sum += weight * cost(v+dy, u+dx, d);
            weight_sum += weight;
        }
    }
    return sum / weight_sum;
}
```

#### 6.3 引导滤波聚合

**代表方法**: Guided Filter

```cpp
CostVolume GuidedFilterAggregation(const CostVolume& cost, const Image& guide) {
    CostVolume result;
    for (int d = 0; d < num_depths; d++) {
        result[d] = GuidedFilter(cost[d], guide, radius, epsilon);
    }
    return result;
}
```

#### 6.4 半全局匹配 (SGM)

**代表方法**: SGM, COLMAP的Photometric Consistency

**原理**: 沿多个方向传播，聚合路径代价

```cpp
void SGMAggregation(CostVolume& cost) {
    // 8个或16个方向
    int dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
    int dy[] = {0, 1, 1, 1, 0, -1, -1, -1};

    CostVolume aggregated = 0;

    for (int r = 0; r < 8; r++) {
        CostVolume path_cost = 0;

        // 沿方向r遍历
        for (沿方向r的每个像素 p) {
            for (int d = 0; d < D; d++) {
                float prev_min = min(path_cost(p-r));

                // SGM递推公式
                path_cost(p, d) = cost(p, d) + min(
                    path_cost(p-r, d),           // 深度不变
                    path_cost(p-r, d-1) + P1,    // 深度变化1
                    path_cost(p-r, d+1) + P1,    // 深度变化1
                    prev_min + P2                 // 深度变化>1
                ) - prev_min;
            }
        }

        aggregated += path_cost;
    }

    cost = aggregated;
}
```

**参数**:
- P1: 小惩罚，用于深度变化=1的情况
- P2: 大惩罚，用于深度变化>1的情况
- 通常 P2 = P2_init / (1 + |I(p) - I(p-r)|)

#### 6.5 多尺度聚合

```cpp
CostVolume MultiScaleAggregation(const CostVolume& cost) {
    vector<CostVolume> pyramid;

    // 构建金字塔
    pyramid.push_back(cost);
    for (int level = 1; level < num_levels; level++) {
        pyramid.push_back(Downsample(pyramid[level-1]));
    }

    // 从粗到细聚合
    CostVolume result = pyramid[num_levels-1];
    for (int level = num_levels-2; level >= 0; level--) {
        result = Upsample(result) + pyramid[level];
    }

    return result;
}
```

---

## 第四部分：优化与传播

### 7. 传播策略

#### 7.1 顺序传播 (Sequential)

**openMVS锯齿形扫描**:
```
扫描模式:
1 2 3
4 5 6  →  对角线扫描: 1,2,4,3,5,7,6,8,9
7 8 9

传播方向交替:
奇数迭代: 左上→右下
偶数迭代: 右下→左上
```

#### 7.2 棋盘格传播 (Checkerboard)

**ACMMP, Gipuma**:
```
红色格子(偶数迭代):     黑色格子(奇数迭代):
■ □ ■ □ ■              □ ■ □ ■ □
□ ■ □ ■ □              ■ □ ■ □ ■
■ □ ■ □ ■              □ ■ □ ■ □
```

#### 7.3 多方向传播

**4方向传播 (openMVS)**:
```
    ↑
  ← P →  (只使用扫描方向的2个邻居)
    ↓
```

**8方向传播 (ACMMP)**:
```
  ↖ ↑ ↗
  ← P →
  ↙ ↓ ↘
```

**自适应邻居选择 (ACMMP)**:
```
近邻域: 1-4像素，搜索对角线方向
远邻域: 3-23像素，搜索最小代价点
```

#### 7.4 分层传播

**流程**:
```
粗分辨率:
  [P] → 传播范围大，找到大致位置
    ↓ 上采样
细分辨率:
  [P] → 传播范围小，精细优化
```

---

### 8. 全局优化方法

#### 8.1 MRF/CRF优化

**能量函数**:
```
E(D) = Σᵢ ψᵢ(dᵢ) + λ Σᵢⱼ ψᵢⱼ(dᵢ, dⱼ)

ψᵢ(dᵢ) = 数据项 = 匹配代价
ψᵢⱼ(dᵢ, dⱼ) = 平滑项 = 邻域深度一致性
```

**平滑项选择**:

Potts模型:
```cpp
float Potts(float d1, float d2) {
    return (d1 != d2) ? lambda : 0;
}
```

截断线性:
```cpp
float TruncatedLinear(float d1, float d2, float tau) {
    return min(abs(d1 - d2), tau);
}
```

截断二次:
```cpp
float TruncatedQuadratic(float d1, float d2, float tau) {
    float diff = d1 - d2;
    return min(diff * diff, tau * tau);
}
```

#### 8.2 Belief Propagation

```cpp
void BeliefPropagation(CostVolume& cost, int iterations) {
    // 消息数组
    Message m_left, m_right, m_up, m_down;

    for (int iter = 0; iter < iterations; iter++) {
        for (每个像素 p) {
            // 更新向左的消息
            for (int d = 0; d < D; d++) {
                m_left(p, d) = min_d'(
                    cost(p, d') +
                    pairwise(d, d') +
                    m_right(p+1, d') +
                    m_up(p-W, d') +
                    m_down(p+W, d')
                );
            }
            // 类似更新其他方向
        }
    }

    // 计算belief
    for (每个像素 p) {
        for (int d = 0; d < D; d++) {
            belief(p, d) = cost(p, d) +
                           m_left(p-1, d) + m_right(p+1, d) +
                           m_up(p-W, d) + m_down(p+W, d);
        }
        depth(p) = argmin(belief(p));
    }
}
```

#### 8.3 Graph Cut

```cpp
void GraphCut(const CostVolume& cost, DepthMap& depth) {
    // 构建图
    Graph g;

    for (每个像素 p) {
        // 添加数据项边
        for (int d = 0; d < D; d++) {
            g.add_edge(p_d, p_{d+1}, cost(p, d), cost(p, d));
        }

        // 添加平滑项边
        for (每个邻居 q) {
            g.add_edge(p_d, q_d, pairwise(d, d), 0);
        }
    }

    // 求最小割
    g.maxflow();

    // 提取深度
    for (每个像素 p) {
        depth(p) = g.what_segment(p);
    }
}
```

#### 8.4 PatchMatch优化

**迭代过程**:
```cpp
void PatchMatchOptimization(DepthMap& depth, NormalMap& normal) {
    for (int iter = 0; iter < num_iters; iter++) {
        // 1. 空间传播
        for (每个像素 p，按扫描顺序) {
            for (每个邻居 n) {
                // 尝试邻居的假设
                float cost_n = ScorePixel(depth(n), normal(n));
                if (cost_n < cost(p)) {
                    depth(p) = InterpolateDepth(n, p);
                    normal(p) = normal(n);
                }
            }
        }

        // 2. 随机搜索
        for (每个像素 p) {
            float range = initial_range;
            while (range > min_range) {
                // 随机扰动
                float d_new = depth(p) + RandomUniform(-range, range);
                Normal n_new = PerturbNormal(normal(p), range);

                float cost_new = ScorePixel(d_new, n_new);
                if (cost_new < cost(p)) {
                    depth(p) = d_new;
                    normal(p) = n_new;
                }

                range *= 0.5;  // 缩小搜索范围
            }
        }
    }
}
```

---

## 第五部分：多视图处理

### 9. 视图选择

#### 9.1 基于共视性的选择

```cpp
vector<int> SelectViewsByCovisibility(int ref_id, const Scene& scene) {
    vector<pair<int, float>> scores;

    for (int src_id = 0; src_id < scene.num_images; src_id++) {
        if (src_id == ref_id) continue;

        // 计算共视点数
        int shared_points = CountSharedPoints(ref_id, src_id, scene);

        // 计算基线角度
        float angle = ComputeBaselineAngle(ref_id, src_id, scene);

        // 评分
        float score = 0;
        if (angle >= min_angle && angle <= max_angle) {
            score = shared_points * (1 - abs(angle - optimal_angle) / max_angle);
        }

        scores.push_back({src_id, score});
    }

    // 排序选择top-k
    sort(scores.begin(), scores.end(), [](auto& a, auto& b) {
        return a.second > b.second;
    });

    vector<int> selected;
    for (int i = 0; i < min(k, scores.size()); i++) {
        selected.push_back(scores[i].first);
    }

    return selected;
}
```

#### 9.2 基于三角化质量

**COLMAP方法**:
```cpp
float TriangulationQuality(const Camera& ref, const Camera& src, const Point3& X) {
    // 计算三角化角度
    Vec3 ray_ref = normalize(X - ref.center);
    Vec3 ray_src = normalize(X - src.center);
    float angle = acos(ray_ref.dot(ray_src));

    // 理想角度 5-15度
    if (angle < deg2rad(5) || angle > deg2rad(30))
        return 0;

    return sin(angle);  // 角度越接近90度越好
}
```

#### 9.3 动态视图选择

**ACMMP方法** - 在迭代中动态调整:
```cuda
// 计算每个视图的匹配代价
for (int i = 0; i < num_views; i++) {
    cost_vector[i] = ComputeNCC(ref, views[i], p, hypothesis);
}

// 选择top-k个最佳视图
sort(cost_vector);
for (int i = 0; i < top_k; i++) {
    selected_views |= (1 << cost_vector[i].index);
}
```

#### 9.4 概率视图选择

```cpp
// 基于代价的概率分布
vector<float> probs;
for (int i = 0; i < num_views; i++) {
    probs.push_back(exp(-cost[i] / temperature));
}
Normalize(probs);

// 概率采样
int selected = SampleFromDistribution(probs);
```

---

### 10. 遮挡处理

#### 10.1 前向-后向一致性检查

```cpp
bool ForwardBackwardConsistency(const DepthMap& depth_ref, const DepthMap& depth_src,
                                 const Camera& cam_ref, const Camera& cam_src,
                                 int u, int v, float threshold) {
    // 前向投影: ref → src
    float d_ref = depth_ref(v, u);
    Point3 X = cam_ref.Unproject(u, v, d_ref);
    Point2 p_src = cam_src.Project(X);

    // 后向投影: src → ref
    float d_src = depth_src(p_src);
    Point3 X_src = cam_src.Unproject(p_src, d_src);
    Point2 p_ref_back = cam_ref.Project(X_src);

    // 检查重投影误差
    float reproj_error = Distance(Point2(u, v), p_ref_back);
    return reproj_error < threshold;
}
```

#### 10.2 深度序一致性

```cpp
bool DepthOrderConsistency(float d_ref, float d_src_expected, float d_src_actual) {
    // 如果源图像的实际深度比期望深度近，说明被遮挡
    return d_src_actual >= d_src_expected * (1 - tolerance);
}
```

#### 10.3 软可见性 (Soft Visibility)

**代表方法**: 一些全局MVS方法

```cpp
float SoftVisibility(float d_ref, float d_src_expected, float d_src_actual, float sigma) {
    if (d_src_actual == 0) return 0;  // 无效深度

    float diff = d_src_expected - d_src_actual;
    if (diff > 0)  // 被遮挡
        return exp(-diff * diff / (2 * sigma * sigma));
    else
        return 1.0;  // 可见
}
```

#### 10.4 多视图一致性投票

```cpp
float OcclusionAwareAggregation(const vector<float>& costs, const vector<bool>& visible) {
    vector<float> valid_costs;
    for (int i = 0; i < costs.size(); i++) {
        if (visible[i])
            valid_costs.push_back(costs[i]);
    }

    if (valid_costs.empty())
        return MAX_COST;

    // 使用中位数或截断均值
    sort(valid_costs.begin(), valid_costs.end());
    return valid_costs[valid_costs.size() / 2];  // 中位数
}
```

---

### 11. 几何一致性

#### 11.1 重投影误差

```cpp
float ReprojectionError(const Camera& ref, const Camera& src,
                        const DepthMap& depth_ref, const DepthMap& depth_src,
                        int u, int v) {
    float d_ref = depth_ref(v, u);
    Point3 X = ref.Unproject(u, v, d_ref);

    Point2 p_src = src.Project(X);
    float d_src = depth_src(p_src);

    Point3 X_src = src.Unproject(p_src, d_src);
    Point2 p_ref_back = ref.Project(X_src);

    return Distance(Point2(u, v), p_ref_back);
}
```

#### 11.2 深度一致性

```cpp
float DepthConsistency(const Camera& ref, const Camera& src,
                       const DepthMap& depth_ref, const DepthMap& depth_src,
                       int u, int v) {
    float d_ref = depth_ref(v, u);
    Point3 X = ref.Unproject(u, v, d_ref);

    Point2 p_src = src.Project(X);
    float d_src_expected = src.Depth(X);
    float d_src_actual = depth_src(p_src);

    return abs(d_src_expected - d_src_actual) / d_src_expected;
}
```

#### 11.3 法向量一致性

```cpp
float NormalConsistency(const NormalMap& normal_ref, const NormalMap& normal_src,
                        const Camera& ref, const Camera& src,
                        int u, int v, float depth) {
    Normal n_ref = normal_ref(v, u);
    Point3 X = ref.Unproject(u, v, depth);
    Point2 p_src = src.Project(X);
    Normal n_src = normal_src(p_src);

    // 将法向量转换到世界坐标系
    Normal n_ref_world = ref.R.transpose() * n_ref;
    Normal n_src_world = src.R.transpose() * n_src;

    // 计算夹角
    return acos(n_ref_world.dot(n_src_world));
}
```

#### 11.4 ACMMP几何一致性代价

```cuda
__device__ float ComputeGeomConsistencyCost(...) {
    // 参考视图 → 源视图 → 参考视图
    float3 X_ref = Get3DPoint(ref_cam, p, depth);
    float2 p_src = Project(X_ref, src_cam);
    float d_src = tex2D(depth_src, p_src);
    float3 X_src = Get3DPoint(src_cam, p_src, d_src);
    float2 p_ref_back = Project(X_src, ref_cam);

    return sqrt((p.x - p_ref_back.x)² + (p.y - p_ref_back.y)²);
}
```

---

## 第六部分：后处理与增强

### 12. 置信度估计

#### 12.1 基于匹配代价

```cpp
float CostBasedConfidence(float cost, float cost_threshold) {
    // 代价越低，置信度越高
    return 1.0 - min(cost / cost_threshold, 1.0);
}
```

#### 12.2 左右一致性置信度

```cpp
float LRConsistencyConfidence(const DepthMap& depth_left, const DepthMap& depth_right,
                               int u, int v, float threshold) {
    float d_left = depth_left(v, u);

    // 在右图中找对应点
    float disparity = baseline * focal / d_left;
    int u_right = u - disparity;

    float d_right = depth_right(v, u_right);
    float disparity_right = baseline * focal / d_right;

    // 检查一致性
    float diff = abs(disparity - disparity_right);
    return exp(-diff / threshold);
}
```

#### 12.3 唯一性置信度

```cpp
float UniquenessConfidence(const CostCurve& cost, int best_d) {
    // 找到次优深度
    float best_cost = cost[best_d];
    float second_best = MAX_FLOAT;

    for (int d = 0; d < cost.size(); d++) {
        if (d == best_d) continue;
        if (cost[d] < second_best)
            second_best = cost[d];
    }

    // 唯一性比率
    return (second_best - best_cost) / second_best;
}
```

#### 12.4 多视图一致性置信度

```cpp
float MultiViewConfidence(int consistent_views, int total_views) {
    return (float)consistent_views / total_views;
}
```

---

### 13. 深度图滤波

#### 13.1 中值滤波

```cpp
void MedianFilter(DepthMap& depth, int kernel_size) {
    for (int v = 0; v < H; v++) {
        for (int u = 0; u < W; u++) {
            vector<float> values;
            for (int dy = -k/2; dy <= k/2; dy++) {
                for (int dx = -k/2; dx <= k/2; dx++) {
                    if (depth(v+dy, u+dx) > 0)
                        values.push_back(depth(v+dy, u+dx));
                }
            }
            sort(values.begin(), values.end());
            depth_out(v, u) = values[values.size() / 2];
        }
    }
}
```

#### 13.2 双边滤波

```cpp
void BilateralFilter(DepthMap& depth, const Image& guide,
                     float sigma_spatial, float sigma_depth) {
    for (int v = 0; v < H; v++) {
        for (int u = 0; u < W; u++) {
            float sum = 0, weight_sum = 0;

            for (int dy = -k/2; dy <= k/2; dy++) {
                for (int dx = -k/2; dx <= k/2; dx++) {
                    float d = depth(v+dy, u+dx);
                    if (d <= 0) continue;

                    // 空间权重
                    float w_s = exp(-(dx*dx + dy*dy) / (2 * sigma_spatial²));

                    // 深度权重
                    float w_d = exp(-|depth(v,u) - d|² / (2 * sigma_depth²));

                    // 颜色权重（引导图像）
                    float w_c = exp(-|guide(v,u) - guide(v+dy,u+dx)|² / (2 * sigma_color²));

                    float w = w_s * w_d * w_c;
                    sum += w * d;
                    weight_sum += w;
                }
            }

            depth_out(v, u) = sum / weight_sum;
        }
    }
}
```

#### 13.3 形态学滤波

```cpp
void MorphologicalFilter(DepthMap& depth) {
    // 膨胀 - 填充小孔
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(5, 5));
    cv::dilate(depth, depth_dilated, kernel);

    // 腐蚀 - 去除噪点
    cv::erode(depth, depth_eroded, kernel);

    // 开运算 - 去除小噪点
    cv::morphologyEx(depth, depth_opened, cv::MORPH_OPEN, kernel);

    // 闭运算 - 填充小孔
    cv::morphologyEx(depth, depth_closed, cv::MORPH_CLOSE, kernel);
}
```

#### 13.4 边缘保持滤波

**引导滤波**:
```cpp
cv::Mat GuidedFilter(const cv::Mat& depth, const cv::Mat& guide, int r, float eps) {
    cv::Mat result;
    cv::ximgproc::guidedFilter(guide, depth, result, r, eps);
    return result;
}
```

#### 13.5 斑点去除

```cpp
void SpeckleFilter(DepthMap& depth, int min_size, float max_diff) {
    // 连通组件分析
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(
        depth > 0, labels, stats, centroids
    );

    // 去除小斑点
    for (int i = 1; i < num_labels; i++) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < min_size) {
            // 将该区域设为无效
            depth.setTo(0, labels == i);
        }
    }
}
```

---

### 14. 纹理分析与处理

#### 14.1 纹理度量

**方差方法**:
```cpp
float TextureVariance(const Image& img, int u, int v, int win_size) {
    float sum = 0, sum_sq = 0;
    int n = 0;

    for (int dy = -win_size/2; dy <= win_size/2; dy++) {
        for (int dx = -win_size/2; dx <= win_size/2; dx++) {
            float val = img(v+dy, u+dx);
            sum += val;
            sum_sq += val * val;
            n++;
        }
    }

    float mean = sum / n;
    return sum_sq / n - mean * mean;
}
```

**梯度方法**:
```cpp
float TextureGradient(const Image& img, int u, int v) {
    float gx = img(v, u+1) - img(v, u-1);
    float gy = img(v+1, u) - img(v-1, u);
    return sqrt(gx*gx + gy*gy);
}
```

#### 14.2 弱纹理区域处理

**方法1: 增大窗口**
```cpp
int GetAdaptiveWindowSize(float texture) {
    if (texture < low_threshold)
        return large_window;
    else if (texture < high_threshold)
        return medium_window;
    else
        return small_window;
}
```

**方法2: 使用先验**
```cpp
if (texture < threshold) {
    // 使用平面先验
    depth = prior_depth;
    normal = prior_normal;
}
```

**方法3: 插值填充**
```cpp
void InterpolateLowTextureRegions(DepthMap& depth, const TextureMap& texture) {
    // 找到高纹理区域的深度
    // 插值填充低纹理区域
    for (每个低纹理像素) {
        depth(p) = BilinearInterpolation(nearby_high_texture_depths);
    }
}
```

#### 14.3 重复纹理处理

```cpp
float AmbiguityAwareMatching(const CostCurve& cost) {
    // 找到所有局部最小值
    vector<int> minima = FindLocalMinima(cost);

    // 如果有多个相似的最小值，降低置信度
    if (minima.size() > 1) {
        float diff = cost[minima[1]] - cost[minima[0]];
        if (diff < ambiguity_threshold)
            return LOW_CONFIDENCE;
    }

    return cost[minima[0]];
}
```

---

## 第七部分：多尺度与融合

### 15. 多分辨率策略

#### 15.1 图像金字塔

```cpp
void BuildImagePyramid(const Image& img, vector<Image>& pyramid, int levels) {
    pyramid.push_back(img);
    for (int l = 1; l < levels; l++) {
        Image down;
        cv::pyrDown(pyramid[l-1], down);
        pyramid.push_back(down);
    }
}
```

#### 15.2 粗到细优化

```cpp
void CoarseToFineOptimization(DepthMap& depth, const ImagePyramid& pyramid) {
    // 从最粗层开始
    for (int l = num_levels - 1; l >= 0; l--) {
        if (l == num_levels - 1) {
            // 最粗层：随机初始化
            RandomInit(depth[l]);
        } else {
            // 其他层：从上层上采样
            depth[l] = Upsample(depth[l+1]);
        }

        // 在当前层优化
        PatchMatchOptimize(depth[l], pyramid[l]);
    }
}
```

#### 15.3 上采样方法比较

| 方法 | 公式 | 特点 |
|-----|-----|------|
| 最近邻 | d(p) = d(round(p/s)) | 块状，快速 |
| 双线性 | d(p) = BilinearInterp() | 平滑，模糊边缘 |
| 双三次 | d(p) = BicubicInterp() | 更平滑 |
| JBU | d(p) = Σ w(p,q)×d(q) / Σw | 边缘保持 |

---

### 16. 深度图融合

#### 16.1 简单融合

```cpp
PointCloud SimpleFusion(const vector<DepthMap>& depths,
                        const vector<Camera>& cameras) {
    PointCloud cloud;

    for (int i = 0; i < depths.size(); i++) {
        for (int v = 0; v < H; v++) {
            for (int u = 0; u < W; u++) {
                float d = depths[i](v, u);
                if (d > 0) {
                    Point3 X = cameras[i].Unproject(u, v, d);
                    cloud.AddPoint(X);
                }
            }
        }
    }

    // 去重
    cloud.RemoveDuplicates(distance_threshold);

    return cloud;
}
```

#### 16.2 一致性融合

```cpp
PointCloud ConsistencyFusion(const vector<DepthMap>& depths,
                             const vector<Camera>& cameras,
                             int min_consistent_views) {
    PointCloud cloud;

    for (int ref = 0; ref < depths.size(); ref++) {
        for (int v = 0; v < H; v++) {
            for (int u = 0; u < W; u++) {
                float d = depths[ref](v, u);
                if (d <= 0) continue;

                Point3 X = cameras[ref].Unproject(u, v, d);
                int consistent_count = 0;
                float depth_sum = d;

                // 检查与其他视图的一致性
                for (int src = 0; src < depths.size(); src++) {
                    if (src == ref) continue;

                    Point2 p_src = cameras[src].Project(X);
                    float d_src = depths[src](p_src);
                    float d_expected = cameras[src].Depth(X);

                    if (abs(d_src - d_expected) / d_expected < consistency_threshold) {
                        consistent_count++;
                        depth_sum += d_src;
                    }
                }

                if (consistent_count >= min_consistent_views) {
                    // 使用平均深度
                    float avg_d = depth_sum / (consistent_count + 1);
                    X = cameras[ref].Unproject(u, v, avg_d);
                    cloud.AddPoint(X);
                }
            }
        }
    }

    return cloud;
}
```

#### 16.3 TSDF融合

```cpp
void TSDFFusion(const vector<DepthMap>& depths,
                const vector<Camera>& cameras,
                VoxelGrid& tsdf) {
    for (int i = 0; i < depths.size(); i++) {
        for (每个体素 voxel) {
            Point3 X = voxel.center;
            Point2 p = cameras[i].Project(X);

            float d_observed = depths[i](p);
            float d_voxel = cameras[i].Depth(X);

            // 计算TSDF值
            float sdf = d_observed - d_voxel;
            float tsdf_value = clamp(sdf / truncation, -1, 1);

            // 加权更新
            float w_new = 1.0;
            tsdf.value(voxel) = (tsdf.weight(voxel) * tsdf.value(voxel) + w_new * tsdf_value)
                              / (tsdf.weight(voxel) + w_new);
            tsdf.weight(voxel) += w_new;
        }
    }

    // 提取等值面
    MarchingCubes(tsdf, mesh);
}
```

---

## 第八部分：主流系统技术选型

### 17. 各系统技术选型

| 模块 | COLMAP | openMVS | ACMMP | Gipuma | PMVS | SGM |
|-----|--------|---------|-------|--------|------|-----|
| **深度表示** | 平面假设 | 平面假设 | 平面假设 | 平面假设 | Patch | 视差 |
| **初始化** | 随机+稀疏 | Delaunay | 分层 | 分层 | 稀疏扩展 | 全搜索 |
| **匹配代价** | 双边NCC | 加权NCC | 双边NCC | NCC | NCC | Census |
| **代价聚合** | PatchMatch | PatchMatch | PatchMatch | PatchMatch | 局部 | SGM |
| **传播方式** | 棋盘格 | 锯齿形 | 棋盘格 | 棋盘格 | 扩展 | 8路径 |
| **视图选择** | 几何+光度 | MRF | 动态 | 预选 | 可见性 | 双目 |
| **几何一致性** | 多阶段 | 后处理 | 融入代价 | 后处理 | 内置 | 左右 |
| **平面先验** | 无 | 无 | Delaunay | 无 | 无 | 无 |
| **多分辨率** | 金字塔 | 金字塔 | JBU | 金字塔 | 无 | 无 |
| **滤波** | 一致性 | 双边+形态 | 一致性 | 一致性 | - | 左右+唯一 |
| **融合** | 一致性 | 一致性 | 一致性 | 一致性 | 直接 | - |
| **实现平台** | CPU+CUDA | CPU(+CUDA) | CUDA | CUDA | CPU | CPU/GPU |

---

## 总结：技术发展脉络

```
早期方法 (2000s)
├── 全局方法: Graph Cut, Belief Propagation
│   └── 精度高，速度慢
└── 局部方法: Block Matching, SGM
    └── 速度快，边缘差

PatchMatch革命 (2010s)
├── PatchMatch Stereo (2011)
│   ├── 随机搜索 + 空间传播
│   └── 平面假设处理倾斜表面
├── Gipuma (2015)
│   └── GPU棋盘格并行
├── COLMAP (2016)
│   └── 完整pipeline + 多视图一致性
└── ACMMP (2019)
    ├── 动态视图选择
    ├── 几何一致性融入代价
    └── Delaunay平面先验

深度学习时代 (2020s)
├── MVSNet系列
│   └── 学习代价体积正则化
├── PatchmatchNet
│   └── 学习传播和代价
└── 传统方法仍有优势
    ├── 泛化性好
    ├── 无需训练
    └── 可解释性强
```

---

## 附录：参数调优指南

### 常见参数及其影响

| 参数 | 典型范围 | 增大效果 | 减小效果 |
|-----|---------|---------|---------|
| patch_size | 5-35 | 更鲁棒，但模糊 | 更精细，但噪声多 |
| num_iterations | 3-12 | 收敛更好 | 速度更快 |
| num_views | 4-20 | 更鲁棒 | 速度更快 |
| depth_samples | 32-256 | 精度更高 | 速度更快 |
| consistency_threshold | 0.01-0.05 | 更多点，噪声多 | 更少点，更干净 |
| P1 (SGM) | 5-20 | 更平滑 | 更多细节 |
| P2 (SGM) | 50-200 | 更平滑 | 更多细节 |
