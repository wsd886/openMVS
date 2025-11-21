# ACMMP vs openMVS 稠密重建算法深度对比

## 目录
1. [算法概述](#1-算法概述)
2. [核心差异总结](#2-核心差异总结)
3. [传播策略对比](#3-传播策略对比)
4. [视图选择机制](#4-视图选择机制)
5. [几何一致性](#5-几何一致性)
6. [平面先验](#6-平面先验acmmp独有)
7. [多分辨率策略](#7-多分辨率策略)
8. [代价聚合方式](#8-代价聚合方式)
9. [融合策略](#9-融合策略)
10. [性能与适用场景](#10-性能与适用场景)

---

## 1. 算法概述

### 1.1 openMVS
- **论文基础**: "Accurate Multiple View 3D Reconstruction Using Patch-Based Stereo for Large-Scale Scenes" (Shen, 2013)
- **核心特点**: 经典PatchMatch + 多分辨率 + 加权NCC
- **实现语言**: C++ (CPU为主，可选CUDA加速)

### 1.2 ACMMP
- **论文**: "Multi-Scale Geometric Consistency Guided and Planar Prior Assisted Multi-View Stereo" (Xu et al., TPAMI 2022)
- **核心特点**: 棋盘传播 + 自适应视图选择 + 平面先验 + 多尺度几何一致性
- **实现语言**: CUDA (纯GPU实现)

---

## 2. 核心差异总结

| 特性 | openMVS | ACMMP |
|------|---------|-------|
| **传播模式** | 锯齿形(Zigzag)顺序传播 | 红黑棋盘(Checkerboard)并行传播 |
| **视图选择** | 静态预选择 + MRF全局优化 | 动态自适应视图选择 |
| **几何一致性** | 可选的后处理 | 多尺度几何一致性融入代价 |
| **平面先验** | 无 | Delaunay三角化平面先验 |
| **NCC计算** | 加权NCC (Weighted NCC) | 双边加权NCC (Bilateral NCC) |
| **邻居传播** | 4邻域 (上下左右) | 8邻域 (近+远邻居) |
| **搜索策略** | 固定邻域 + 随机搜索 | 自适应邻域搜索 + 随机精炼 |
| **多分辨率** | 简单上采样 | 联合双边上采样(JBU) |
| **实现平台** | CPU + 可选CUDA | 纯CUDA |

---

## 3. 传播策略对比

### 3.1 openMVS: 锯齿形传播 (Zigzag)

```
传播方向交替:
迭代0: 左上→右下 (LT2RB)
迭代1: 右下→左上 (RB2LT)
迭代2: 左上→右下
...

邻居定义 (4邻域):
LT2RB方向: 左邻居、上邻居
RB2LT方向: 右邻居、下邻居
```

**代码位置**: `libs/MVS/DepthMap.cpp:641-766`

```cpp
// openMVS的邻居传播
if (dir == LT2RB) {
    // 从左上到右下
    if (x0.x > nSizeHalfWindow) {
        const ImageRef nx(x0.x-1, x0.y);  // 左邻居
        // 传播左邻居的估计
    }
    if (x0.y > nSizeHalfWindow) {
        const ImageRef nx(x0.x, x0.y-1);  // 上邻居
        // 传播上邻居的估计
    }
}
```

**优点**: 实现简单，内存访问模式好
**缺点**: 顺序依赖，难以完全并行化

### 3.2 ACMMP: 棋盘传播 (Checkerboard)

```
棋盘模式:
  B R B R B R
  R B R B R B
  B R B R B R
  R B R B R B

更新顺序:
1. 先更新所有黑色(B)像素 - 完全并行
2. 再更新所有红色(R)像素 - 完全并行

每个像素使用8个邻居:
- 4个近邻居: up_near, down_near, left_near, right_near
- 4个远邻居: up_far, down_far, left_far, right_far
```

**代码位置**: `ACMMP.cu:1126-1148`

```cuda
// ACMMP的红黑棋盘更新
__global__ void BlackPixelUpdate(...) {
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                       blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = p.y * 2;      // 偶数行
    } else {
        p.y = p.y * 2 + 1;  // 奇数行
    }
    CheckerboardPropagation(...);
}

__global__ void RedPixelUpdate(...) {
    // 类似，但y坐标相反
}
```

**8邻域自适应搜索**:
```cuda
// 在远邻居中寻找最优
for (int i = 1; i < 11; ++i) {
    if (p.y > 2 + 2 * i) {
        int pointTemp = up_far - 2 * i * width;
        if (costs[pointTemp] < costMin) {
            costMin = costs[pointTemp];
            costMinPoint = pointTemp;
        }
    }
}
```

**优点**: 完全GPU并行，收敛更快
**缺点**: 实现复杂，需要两次kernel调用

---

## 4. 视图选择机制

### 4.1 openMVS: 静态预选择

**流程**:
1. **局部选择**: 基于共享点数和视角质量打分
2. **全局优化**: 使用MRF能量最小化
3. **固定不变**: 迭代过程中视图不变

```cpp
// libs/MVS/SceneDensify.cpp:150-267
bool SelectViews(IIndexArr& images, IIndexArr& imagesMap, IIndexArr& neighborsMap) {
    // 构建MRF图
    for (each image pair) {
        edges[pair] = overlap_area;
    }

    // 定义能量
    E_unary = avg_score / neighbor_score;  // 鼓励高分视图
    E_pairwise = fSamePairwise if same_view_selected;  // 惩罚重复选择

    // TRW-S优化
    energy->Minimize_TRW_S(options, lowerBound, energyVal);
}
```

**评分公式**:
```
score = num_shared_points × area_overlap / baseline_quality
```

### 4.2 ACMMP: 动态自适应视图选择

**流程**:
1. **计算每个视图的代价向量**
2. **基于代价动态计算采样概率**
3. **随机采样选择视图**
4. **每次迭代重新选择**

```cuda
// ACMMP.cu:946-1006
// 多假设联合视图选择
float view_weights[32] = {0.0f};
float sampling_probs[32] = {0.0f};

// 基于邻居的先验
for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < num_images - 1; ++j) {
        if (isSet(selected_views[neighbor], j)) {
            view_selection_priors[j] += 0.9f;  // 邻居选了,我也倾向选
        } else {
            view_selection_priors[j] += 0.1f;
        }
    }
}

// 基于代价的采样概率
float cost_threshold = 0.8 * expf((iter) * (iter) / (-90.0f));  // 自适应阈值
for (int i = 0; i < num_images - 1; i++) {
    float tmpw = 0;
    for (int j = 0; j < 8; j++) {
        if (cost_array[j][i] < cost_threshold) {
            tmpw += expf(cost_array[j][i] * cost_array[j][i] / (-0.18f));
        }
    }
    sampling_probs[i] = tmpw * view_selection_priors[i];
}

// 随机采样15次
TransformPDFToCDF(sampling_probs, num_images - 1);
for (int sample = 0; sample < 15; ++sample) {
    float rand_prob = curand_uniform(&rand_states[center]);
    for (int image_id = 0; image_id < num_images - 1; ++image_id) {
        if (sampling_probs[image_id] > rand_prob) {
            view_weights[image_id] += 1.0f;
            break;
        }
    }
}
```

**关键数学**:
```
采样概率 P(view_i) ∝ exp(-cost²/0.18) × prior(view_i)

其中:
- cost: 该视图的NCC代价
- prior: 基于邻居选择的先验概率
```

---

## 5. 几何一致性

### 5.1 openMVS: 后处理式几何一致性

**方式**: 在PatchMatch迭代后，通过FilterDepthMap进行一致性检查

```cpp
// libs/MVS/SceneDensify.cpp:1052-1200
bool FilterDepthMap(DepthData& depthData, const IIndexArr& idxNeighbors) {
    for (each pixel) {
        // 投影到邻居视图
        Point3 X = camera.TransformPointI2W(Point3(x, y, depth));

        int num_consistent = 0;
        for (neighbor in idxNeighbors) {
            Point3f pt = neighborCamera.ProjectPointP3(X);
            Depth depth_neighbor = neighborDepthMap(pt);

            // 深度一致性检查
            if (IsDepthSimilar(pt.z, depth_neighbor, threshold)) {
                // 法向一致性检查
                if (normal.dot(normal_neighbor) > cos(threshold)) {
                    num_consistent++;
                }
            }
        }

        // 一致性不足则移除
        if (num_consistent < nMinViewsFilter) {
            depthMap(x, y) = 0;
        }
    }
}
```

**可选**: 在PatchMatch中加入几何项（`nEstimationGeometricIters`参数）

```cpp
// libs/MVS/DepthMap.cpp:535-551
// 几何一致性代价
Point3f X1 = image1.Tl * Point3f(X0.x*depth, X0.y*depth, depth) + image1.Tm;
Point2f x1(X1);
Depth depth1 = image1.depthMap.sample(x1);
Point2f x0_back = ... // 反投影

float dist = norm(x0 - x0_back);  // 重投影误差
float consistency = min(sqrt(dist*(dist+2)), 4.0);
score += fEstimationGeometricWeight * consistency;  // 加入总代价
```

### 5.2 ACMMP: 多尺度几何一致性融入代价

**特点**: 几何一致性直接参与代价计算和视图权重

```cuda
// ACMMP.cu:507-532
__device__ float ComputeGeomConsistencyCost(
    const cudaTextureObject_t depth_image,
    const Camera ref_camera,
    const Camera src_camera,
    const float4 plane_hypothesis,
    const int2 p)
{
    const float max_cost = 3.0f;

    // 前向投影
    float depth = ComputeDepthfromPlaneHypothesis(ref_camera, plane_hypothesis, p);
    float3 forward_point = Get3DPointonWorld_cu(p.x, p.y, depth, ref_camera);

    // 投影到源图像
    float2 src_pt;
    float src_d;
    ProjectonCamera_cu(forward_point, src_camera, src_pt, src_d);
    float src_depth = tex2D<float>(depth_image, src_pt.x + 0.5f, src_pt.y + 0.5f);

    if (src_depth == 0.0f) return max_cost;

    // 反向投影
    float3 src_3D_pt = Get3DPointonWorld_cu(src_pt.x, src_pt.y, src_depth, src_camera);
    float2 backward_point;
    ProjectonCamera_cu(src_3D_pt, ref_camera, backward_point, ref_d);

    // 重投影误差
    float diff_col = p.x - backward_point.x;
    float diff_row = p.y - backward_point.y;
    return min(max_cost, sqrt(diff_col * diff_col + diff_row * diff_row));
}

// 融入最终代价
final_costs[i] = view_weights[j] * (
    cost_array[i][j] +
    0.2f * ComputeGeomConsistencyCost(...)  // 几何一致性权重0.2
);
```

**多尺度实现**:
- 在不同分辨率层都计算几何一致性
- 低分辨率层的深度图用于指导高分辨率层

---

## 6. 平面先验 (ACMMP独有)

### 6.1 原理

ACMMP使用稀疏点云构建Delaunay三角网，每个三角形拟合一个平面，作为深度估计的先验。

### 6.2 实现流程

**步骤1: 获取支撑点**
```cpp
// main.cpp:119-121
std::vector<cv::Point> support2DPoints;
acmmp.GetSupportPoints(support2DPoints);  // 从深度图获取可靠点
```

**步骤2: Delaunay三角化**
```cpp
// main.cpp:121
const auto triangles = acmmp.DelaunayTriangulation(imageRC, support2DPoints);
```

**步骤3: 拟合平面参数**
```cpp
// main.cpp:162
float4 n4 = acmmp.GetPriorPlaneParams(triangle, depths);
// n4 = (nx, ny, nz, d) 表示平面 nx*x + ny*y + nz*z + d = 0
```

**步骤4: 在PatchMatch中使用平面先验**
```cuda
// ACMMP.cu:1046-1088
if (params.planar_prior && plane_masks[center] > 0) {
    float gamma = 0.5f;
    float depth_sigma = (depth_max - depth_min) / 64.0f;
    float angle_sigma = M_PI * (5.0f / 180.0f);  // 5度

    // 计算与先验的差异
    float depth_prior = ComputeDepthfromPlaneHypothesis(cameras[0], prior_planes[center], p);
    float depth_diff = depth_now - depth_prior;
    float angle_cos = Vec3DotVec3(prior_planes[center], plane_hypotheses[positions[i]]);
    float angle_diff = acos(angle_cos);

    // 平面先验能量
    float prior = gamma + exp(-depth_diff² / two_depth_sigma²)
                        * exp(-angle_diff² / two_angle_sigma²);

    // 综合代价
    restricted_costs[i] = exp(-final_costs[i]² / beta) * prior;
}
```

**数学公式**:
```
Prior(d, n) = γ + exp(-(d - d_prior)² / 2σ_d²) × exp(-(θ)² / 2σ_θ²)

其中:
- d: 当前深度估计
- d_prior: 平面先验深度
- θ: 当前法向与先验法向的夹角
- γ = 0.5: 基础先验强度
- σ_d: 深度容差
- σ_θ = 5°: 法向容差
```

---

## 7. 多分辨率策略

### 7.1 openMVS: 简单线性插值

```cpp
// libs/MVS/SceneDensify.cpp:660-664
if (scaleNumber != totalScaleNumber) {
    cv::resize(lowResDepthMap, depthData.depthMap, size, 0, 0,
               OPTDENSE::nIgnoreMaskLabel >= 0 ? cv::INTER_NEAREST : cv::INTER_LINEAR);
    cv::resize(lowResNormalMap, depthData.normalMap, size, 0, 0, cv::INTER_NEAREST);
}
```

### 7.2 ACMMP: 联合双边上采样 (JBU)

**原理**: 利用高分辨率彩色图像引导低分辨率深度图的上采样

```cpp
// main.cpp:212-238
void JointBilateralUpsampling(const std::string &dense_folder,
                               const Problem &problem,
                               int acmmp_size)
{
    // 读取低分辨率深度图
    cv::Mat_<float> ref_depth;
    readDepthDmb(depth_path, ref_depth);

    // 读取高分辨率引导图像
    cv::Mat image_float;
    image_uint.convertTo(image_float, CV_32FC1);

    // 执行JBU
    RunJBU(scaled_image_float, ref_depth, dense_folder, problem);
}
```

**JBU公式**:
```
D_high(p) = Σ_q∈N(p) [ w_s(p,q) × w_r(I(p), I(q)) × D_low(q↓) ] / Σ weights

其中:
- w_s: 空间高斯权重
- w_r: 颜色相似性权重
- D_low(q↓): 低分辨率深度值
- I(p), I(q): 高分辨率引导图像的颜色
```

---

## 8. 代价聚合方式

### 8.1 openMVS: Top-K Mean

```cpp
// libs/MVS/DepthMap.cpp:594-610 (DENSE_AGGNCC_MINMEAN)
if (idxScore == 0)
    return *std::min_element(scores.cbegin(), scores.cend());

const float* pescore(&scores.GetNth(idxScore));  // 第K个元素
const float* pscore(scores.cbegin());
int n(1);
float score(*pscore);
do {
    const float s(*(++pscore));
    if (s >= thRobust) break;
    score += s;
    ++n;
} while (pscore < pescore);
return score / n;  // 前K个的平均
```

### 8.2 ACMMP: 加权平均

```cuda
// ACMMP.cu:1009-1027
float final_costs[8] = {0.0f};
for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < num_images - 1; ++j) {
        if (view_weights[j] > 0) {
            final_costs[i] += view_weights[j] * (
                cost_array[i][j] +
                0.2f * ComputeGeomConsistencyCost(...)
            );
        }
    }
    final_costs[i] /= weight_norm;  // 归一化
}
```

---

## 9. 融合策略

### 9.1 openMVS

```cpp
// libs/MVS/SceneDensify.cpp:1450-1646
void FuseDepthMaps(...) {
    for (each reference depth map) {
        for (each pixel) {
            // 检查邻居深度图一致性
            for (neighbor in neighbors) {
                if (IsDepthSimilar(pt.z, depth_neighbor, threshold)) {
                    if (normal.dot(normal_neighbor) > normalError) {
                        // 加权累加
                        X += X_neighbor * confidence;
                        num_consistent++;
                    }
                }
            }

            // 足够一致才保留
            if (num_consistent >= nMinViewsFuse) {
                point = X / total_weight;  // 加权平均位置
                pointcloud.add(point);
            }
        }
    }
}
```

**一致性阈值**:
- 深度: `fDepthDiffThreshold = 0.01` (1%)
- 法向: `fNormalDiffThreshold = 25°`
- 最小视图数: `nMinViewsFuse = 3`

### 9.2 ACMMP

```cpp
// main.cpp:296-386
void RunFusion(...) {
    for (each reference image) {
        for (each pixel) {
            // 投影到邻居
            float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[i]);

            int num_consistent = 0;
            float dynamic_consistency = 0;

            for (each neighbor) {
                ProjectonCamera(PointX, cameras[src_id], point, proj_depth);

                // 重投影误差
                float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
                float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
                float angle = GetAngle(ref_normal, src_normal);

                // 严格阈值
                if (reproj_error < 2.0f &&
                    relative_depth_diff < 0.01f &&
                    angle < 0.174533f) {  // 10度

                    float tmp_index = reproj_error + 200 * relative_depth_diff + angle * 10;
                    dynamic_consistency += exp(-tmp_index);
                    num_consistent++;
                }
            }

            // 动态一致性阈值
            if (num_consistent >= 1 && dynamic_consistency > 0.3 * num_consistent) {
                PointCloud.push_back(point);
            }
        }
    }
}
```

**关键区别**:
| 指标 | openMVS | ACMMP |
|------|---------|-------|
| 重投影误差阈值 | 无明确限制 | < 2.0 pixels |
| 深度差异阈值 | 1% | 1% |
| 法向角度阈值 | 25° | 10° |
| 动态一致性 | 无 | `exp(-error)` 加权 |

---

## 10. 性能与适用场景

### 10.1 性能对比

| 指标 | openMVS | ACMMP |
|------|---------|-------|
| **速度** | CPU较慢，CUDA中等 | 纯CUDA，非常快 |
| **内存** | 较低 | 较高（多图像并行） |
| **精度** | 高 | 更高（平面先验） |
| **完整度** | 好 | 更好（自适应视图选择） |
| **易用性** | 高（完整pipeline） | 中（需要预处理） |

### 10.2 适用场景

**openMVS更适合**:
- 通用场景重建
- 资源受限环境
- 需要完整pipeline（SfM→稠密→网格→纹理）
- 没有强GPU的情况

**ACMMP更适合**:
- 大规模建筑/城市场景
- 弱纹理区域（平面先验帮助大）
- 追求最高精度
- 有强GPU的研究环境

### 10.3 主要创新点总结

**ACMMP的核心创新**:

1. **棋盘传播** - 允许完全GPU并行
2. **自适应视图选择** - 每个像素动态选择最优视图组合
3. **平面先验** - 利用场景几何先验提高弱纹理区域精度
4. **多尺度几何一致性** - 在代价计算中融入多视图一致性
5. **联合双边上采样** - 保持边缘的高质量上采样

---

## 代码位置快速索引

### openMVS
| 功能 | 文件 | 行号 |
|------|------|------|
| PatchMatch迭代 | `libs/MVS/DepthMap.cpp` | 630-912 |
| NCC计算 | `libs/MVS/DepthMap.cpp` | 465-564 |
| 视图选择 | `libs/MVS/SceneDensify.cpp` | 150-293 |
| 深度图融合 | `libs/MVS/SceneDensify.cpp` | 1450-1646 |
| 几何一致性 | `libs/MVS/DepthMap.cpp` | 535-551 |

### ACMMP
| 功能 | 文件 | 行号 |
|------|------|------|
| 棋盘传播 | `ACMMP.cu` | 1126-1148 |
| 双边NCC | `ACMMP.cu` | 349-421 |
| 自适应视图选择 | `ACMMP.cu` | 946-1006 |
| 几何一致性 | `ACMMP.cu` | 507-532 |
| 平面先验 | `main.cpp` | 113-197 |
| JBU上采样 | `main.cpp` | 212-238 |
| 融合 | `main.cpp` | 240-390 |

---

## 总结

ACMMP相比openMVS的主要优势在于：
1. **更好的并行性** - 棋盘传播允许更高效的GPU利用
2. **更智能的视图选择** - 动态自适应而非静态预选
3. **更强的几何约束** - 多尺度几何一致性
4. **平面区域处理更好** - 平面先验辅助

openMVS的优势在于：
1. **完整的重建流程** - 从SfM到纹理映射的完整pipeline
2. **更好的可移植性** - CPU+CUDA双支持
3. **更成熟稳定** - 经过广泛使用验证
4. **更易于集成** - 丰富的接口和文档

**建议**: 如果追求最高精度且有GPU，用ACMMP；如果需要完整流程或通用性，用openMVS。
