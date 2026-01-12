# Quantitative Evaluation Metrics

This document describes all quantitative metrics used in the plane segmentation evaluation pipeline. Each metric is explained with its mathematical definition, implementation details, variable definitions, and interpretation guidelines.

---

## Table of Contents

1. [Segmentation Covering (SC)](#1-segmentation-covering-sc)
2. [Rand Index (RI)](#2-rand-index-ri)
3. [Variation of Information (VOI)](#3-variation-of-information-voi)
4. [Geometric Precision](#4-geometric-precision)
5. [Geometric Recall](#5-geometric-recall)
6. [Inlier Ratio](#6-inlier-ratio)
7. [Metric Summary Table](#7-metric-summary-table)
8. [Implementation Notes and Potential Issues](#8-implementation-notes-and-potential-issues)

---

## 1. Segmentation Covering (SC)

**Source:** `evaluator.py:54-114`

### Definition

Segmentation Covering measures how well predicted segments cover the ground-truth segments, weighted by segment size. For each ground-truth segment, it finds the best-matching prediction (highest IoU) and computes a size-weighted average.

### Mathematical Formula

$$
SC = \frac{1}{\sum_i |G_i|} \sum_i |G_i| \cdot \max_j \text{IoU}(G_i, P_j)
$$

Where:
- $G_i$ = Ground-truth segment $i$
- $P_j$ = Predicted segment $j$
- $|G_i|$ = Number of pixels in ground-truth segment $i$
- $\text{IoU}(G_i, P_j) = \frac{|G_i \cap P_j|}{|G_i \cup P_j|}$

### Variables

| Variable | Type | Description |
|----------|------|-------------|
| `gt_mask` | `np.ndarray (H,W)` | Ground-truth instance segmentation labels |
| `pred_mask` | `np.ndarray (H,W)` | Predicted instance segmentation labels |
| `ignore_label` | `int` | Label to ignore (default: 0, typically background) |
| `contingency` | `np.ndarray (n_gt, n_pr)` | Matrix where entry (i,j) = pixels in both $G_i$ and $P_j$ |
| `gt_areas` | `np.ndarray (n_gt,)` | Number of pixels per GT segment |
| `pr_areas` | `np.ndarray (n_pr,)` | Number of pixels per predicted segment |
| `best_iou` | `np.ndarray (n_gt,)` | Best IoU for each GT segment |

### Implementation Details

```python
# Build contingency matrix
combined = gt_inv * n_pr + pr_inv
counts = np.bincount(combined, minlength=n_gt * n_pr)
contingency = counts.reshape((n_gt, n_pr))

# Compute best IoU for each GT region
for i in range(n_gt):
    inter = contingency[i, :]
    union = gt_areas[i] + pr_areas - inter
    ious = inter / union  # where union > 0
    best_iou[i] = ious.max()

# Weighted average
sc = (best_iou * gt_areas).sum() / gt_areas.sum()
```

### Interpretation

| SC Value | Interpretation |
|----------|---------------|
| 1.0 | Perfect segmentation (each GT segment has a perfect match) |
| 0.7-0.9 | Good segmentation with minor boundary errors |
| 0.5-0.7 | Moderate performance, some over/under-segmentation |
| < 0.5 | Poor segmentation |

### Properties

- **Range:** [0, 1] (higher is better)
- **Asymmetric:** Measures how well predictions cover GT, not vice versa
- **Size-weighted:** Larger segments contribute more to the score
- **Ignores false positives:** Predicted segments with no GT match don't penalize

---

## 2. Rand Index (RI)

**Source:** `sklearn.metrics.rand_score`

### Definition

The Rand Index measures the similarity between two clusterings by considering all pairs of samples. It computes the fraction of pairs that are either:
- In the same cluster in both clusterings, OR
- In different clusters in both clusterings

### Mathematical Formula

$$
RI = \frac{a + b}{\binom{n}{2}}
$$

Where:
- $n$ = Total number of pixels
- $a$ = Number of pairs in the same cluster in both GT and prediction
- $b$ = Number of pairs in different clusters in both GT and prediction
- $\binom{n}{2} = \frac{n(n-1)}{2}$ = Total number of pixel pairs

### Variables

| Variable | Type | Description |
|----------|------|-------------|
| `labels_true` | `np.ndarray` | Flattened GT labels |
| `labels_pred` | `np.ndarray` | Flattened predicted labels |
| $a$ | `int` | True positives: pairs correctly grouped together |
| $b$ | `int` | True negatives: pairs correctly separated |

### Implementation

```python
from sklearn.metrics import rand_score
ri = rand_score(gt_plane.flatten(), pred.flatten())
```

### Interpretation

| RI Value | Interpretation |
|----------|---------------|
| 1.0 | Identical clusterings |
| 0.9+ | Very high agreement |
| 0.7-0.9 | Good agreement |
| 0.5 | Random agreement (for balanced data) |

### Properties

- **Range:** [0, 1] (higher is better)
- **Symmetric:** RI(A,B) = RI(B,A)
- **Not adjusted for chance:** Values close to 1 don't always indicate good clustering

---

## 3. Variation of Information (VOI)

**Source:** `skimage.metrics.variation_of_information`

### Definition

Variation of Information is an information-theoretic measure of the distance between two clusterings. It quantifies the amount of information lost and gained when going from one clustering to another.

### Mathematical Formula

$$
VOI(S, P) = H(S|P) + H(P|S)
$$

Where:
- $H(S|P)$ = Conditional entropy of GT given prediction ("under-segmentation")
- $H(P|S)$ = Conditional entropy of prediction given GT ("over-segmentation")

The conditional entropies are computed as:

$$
H(S|P) = -\sum_{i,j} \frac{|S_i \cap P_j|}{n} \log \frac{|S_i \cap P_j|}{|P_j|}
$$

$$
H(P|S) = -\sum_{i,j} \frac{|S_i \cap P_j|}{n} \log \frac{|S_i \cap P_j|}{|S_i|}
$$

### Variables

| Variable | Type | Description |
|----------|------|-------------|
| `labels_true` | `np.ndarray (H,W)` | Ground-truth segmentation |
| `labels_pred` | `np.ndarray (H,W)` | Predicted segmentation |
| `Hs` (`H_split`) | `float` | $H(S\|P)$ - measures under-segmentation |
| `Hm` (`H_merge`) | `float` | $H(P\|S)$ - measures over-segmentation |
| `voi_total` | `float` | $H_s + H_m$ - total VOI |

### Implementation

```python
from skimage.metrics import variation_of_information
Hs, Hm = variation_of_information(gt_plane, pred)
voi_total = Hs + Hm
```

### Interpretation

| Component | High Value Means |
|-----------|-----------------|
| $H(S\|P)$ | Under-segmentation: predictions merge GT segments |
| $H(P\|S)$ | Over-segmentation: predictions split GT segments |
| Total VOI | More disagreement between clusterings |

| VOI Value | Interpretation |
|-----------|---------------|
| 0 | Identical clusterings |
| 0-1 | Very similar clusterings |
| 1-2 | Moderate differences |
| > 2 | Significant differences |

### Properties

- **Range:** [0, $\log n$] (lower is better)
- **Symmetric:** VOI(A,B) = VOI(B,A)
- **Metric:** Satisfies triangle inequality
- **Decomposable:** Can analyze over/under-segmentation separately

---

## 4. Geometric Precision

**Source:** `metrics.py:12-65`

### Definition

Geometric Precision measures what fraction of points assigned to predicted plane segments are geometrically consistent with a fitted 3D plane (within a distance threshold).

### Mathematical Formula

$$
\text{Precision}@\tau = \frac{\sum_k \text{InlierPoints}_k}{\sum_k \text{TotalPoints}_k}
$$

Where:
- $k$ = Plane index
- $\text{InlierPoints}_k$ = Points within distance $\tau$ from the fitted plane
- $\text{TotalPoints}_k$ = All points assigned to plane $k$
- $\tau$ = Distance threshold (e.g., 1cm, 2cm, 5cm)

### Variables

| Variable | Type | Description |
|----------|------|-------------|
| `refined_inlier_num_points` | `int` | Points within threshold distance from refined plane |
| `num_points` | `int` | Total points in predicted segment |
| `distance_threshold` | `float` | $\tau$ in meters (0.01, 0.02, 0.05) |
| `global_precision` | `float` | Aggregated precision across all planes |

### Per-Plane vs Global Precision

```python
# Per-plane precision
df["precision"] = df["refined_inlier_num_points"] / df["num_points"]

# Global precision (aggregated)
total_inliers = df["refined_inlier_num_points"].sum()
total_predicted = df["num_points"].sum()
global_precision = total_inliers / total_predicted
```

### Interpretation

| Precision | Interpretation |
|-----------|---------------|
| 1.0 | All predicted plane points lie on actual planes |
| 0.8-1.0 | Most predictions are geometrically correct |
| 0.5-0.8 | Significant noise in predictions |
| < 0.5 | Poor plane fitting or incorrect segmentation |

### Properties

- **Range:** [0, 1] (higher is better)
- **Threshold-dependent:** Tighter thresholds yield lower precision
- **Measures:** Quality of plane predictions (are predicted planes real?)

---

## 5. Geometric Recall

**Source:** `metrics.py:12-65`

### Definition

Geometric Recall measures what fraction of **all scene points** are explained by geometrically consistent plane predictions.

### Mathematical Formula

$$
\text{Recall}@\tau = \frac{\sum_k \text{InlierPoints}_k}{N_{\text{total}}}
$$

Where:
- $N_{\text{total}}$ = Total number of 3D points in the scene (planar AND non-planar)
- $\text{InlierPoints}_k$ = Points within threshold from fitted plane $k$

### Variables

| Variable | Type | Description |
|----------|------|-------------|
| `refined_inlier_num_points` | `int` | Inlier points per plane |
| `total_scene_points` | `int` | Total 3D points in scene |
| `global_recall` | `float` | Fraction of scene explained by planes |

### Implementation

```python
total_inliers = df["refined_inlier_num_points"].sum()
global_recall = total_inliers / total_scene_points
```

### Interpretation

| Recall | Interpretation |
|--------|---------------|
| High (e.g., 0.7+) | Most of scene is explained by planes |
| Medium (0.3-0.7) | Mixed planar/non-planar scene |
| Low (< 0.3) | Scene is mostly non-planar or predictions miss planes |

### Important Note

**This recall definition differs from standard segmentation recall.** The denominator includes ALL scene points, not just ground-truth planar points. This means:
- Recall will be low for scenes with many non-planar regions, even with perfect plane detection
- It measures "scene coverage" rather than "planar region detection"

---

## 6. Inlier Ratio

**Source:** `planefit.py:180-210`

### Definition

Inlier Ratio measures the geometric consistency of a predicted plane segment. It's used as a quality filter to reject poor plane predictions.

### Two Types

#### 6.1 RANSAC Inlier Ratio

$$
\text{RANSAC Inlier Ratio} = \frac{|\text{RANSAC Inliers}|}{|\text{Segment Points}|}
$$

Points that were used to fit the initial RANSAC plane divided by total segment points.

#### 6.2 Refined Inlier Ratio

$$
\text{Refined Inlier Ratio} = \frac{|\text{Refined Inliers}|}{|\text{Segment Points}|}
$$

Points within distance threshold of the least-squares refined plane.

### Variables

| Variable | Type | Description |
|----------|------|-------------|
| `ransac_inlier_num_points` | `int` | Points fitting RANSAC model |
| `refined_inlier_num_points` | `int` | Points within threshold of LS plane |
| `num_points` | `int` | Total points in segment |
| `inlier_ratio_ransac` | `float` | RANSAC inlier ratio |
| `inlier_ratio_refined` | `float` | Refined inlier ratio |
| `inlier_ratio_threshold` | `float` | Quality gate (default: 0.5) |

### Usage as Quality Filter

```python
# Mark planes below threshold as outliers
_, df = mark_planes_below_threshold_as_outliers(_, df, inlier_ratio_threshold=0.5)
```

Planes with `refined_inlier_ratio < 0.5` have their inlier counts set to 0, effectively excluding them from precision/recall calculation.

### Interpretation

| Inlier Ratio | Interpretation |
|--------------|---------------|
| > 0.8 | Excellent plane fit, segment is highly planar |
| 0.5-0.8 | Acceptable fit, some noise or boundary effects |
| < 0.5 | Poor fit, segment is not truly planar (rejected) |

---

## 7. Metric Summary Table

| Metric | Range | Optimal | Measures | Source |
|--------|-------|---------|----------|--------|
| SC | [0, 1] | 1 (higher better) | Segment-level matching quality | Custom |
| RI | [0, 1] | 1 (higher better) | Pairwise clustering agreement | sklearn |
| VOI | [0, log n] | 0 (lower better) | Information distance between clusterings | skimage |
| Precision@τ | [0, 1] | 1 (higher better) | Geometric accuracy of predictions | Custom |
| Recall@τ | [0, 1] | varies | Scene coverage by planes | Custom |
| Inlier Ratio | [0, 1] | > 0.5 | Per-plane geometric consistency | Custom |

---

## 8. Implementation Notes and Potential Issues

### 8.1 Precision/Recall Definition (Potential Issue)

The current precision/recall implementation measures **geometric consistency** rather than **segmentation accuracy**:

**Current Implementation:**
- **Precision:** "Are predicted plane pixels actually planar?" (geometric check)
- **Recall:** "What fraction of ALL scene pixels are explained by planes?"

**Standard Segmentation Metrics Would Be:**
- **Precision:** TP / (TP + FP) where TP = correctly predicted plane pixels
- **Recall:** TP / (TP + FN) where FN = missed plane pixels

**Implications:**
1. Current recall is scene-dependent: a scene with 30% planar area has max recall ~0.3
2. Comparing across datasets with different planar coverage is misleading
3. Missing plane regions doesn't directly penalize recall if other planes are found

**Recommendation:** Consider adding standard segmentation precision/recall alongside geometric metrics.

### 8.2 Threshold Selection

Distance thresholds (1cm, 2cm, 5cm) should match the expected noise level:
- **1cm:** Strict, good for high-quality depth
- **2cm:** Balanced, typical for RGB-D sensors
- **5cm:** Lenient, for noisy depth or large scenes

### 8.3 Inlier Ratio Threshold

The 0.5 threshold for `inlier_ratio_refined` is a quality gate:
- Planes below this are marked as having 0 inliers
- This prevents noisy segments from artificially boosting recall
- May be too strict for scenes with depth noise

### 8.4 Label Resizing

When resizing prediction labels to match GT resolution:
```python
# CORRECT: Use INTER_NEAREST for label data
pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)

# WRONG: Linear interpolation creates invalid labels
pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR)
```

### 8.5 Edge Cases

1. **Empty predictions:** SC=0, RI=low, VOI=high, Precision=0, Recall=0
2. **Single giant segment:** May have high recall but poor precision
3. **Heavily over-segmented:** High precision (each piece is planar) but low SC

---

## Pipeline Flow

```
Input: RGB Image → MoGe Inference → Planarity + Depth + Normals
                         ↓
              Vectorized Segmentation
                         ↓
              Predicted Plane Labels
                         ↓
    ┌────────────────────┴────────────────────┐
    │                                          │
    ▼                                          ▼
2D Metrics                              3D Metrics
(Compare labels)                        (Fit planes to 3D points)
    │                                          │
    ├── Segmentation Covering           ├── Backproject to 3D
    ├── Rand Index                      ├── RANSAC + LS Plane Fit
    └── Variation of Information        ├── Compute Inlier Ratio
                                        ├── Filter by Quality
                                        └── Precision/Recall @ τ
```

---

## References

1. Arbelaez, P., et al. "Contour Detection and Hierarchical Image Segmentation." TPAMI 2011. (Segmentation Covering)
2. Rand, W. M. "Objective Criteria for the Evaluation of Clustering Methods." JASA 1971. (Rand Index)
3. Meilă, M. "Comparing Clusterings—an Information Based Distance." JMA 2007. (Variation of Information)
