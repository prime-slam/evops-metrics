<p style="font-size: 30pt; font-weight: bold;">
    metrics.iou
</p>

<p style="font-size: 20pt; font-weight: bold;">
    metrics.<span style="color: red;">iou</span>(pred_indices, gt_indices) <a href="https://github.com/prime-slam/evops-metrics/blob/release-1.0/src/evops/metrics/point_based.py#L28">[source]</a>
</p>

<p>
Evaluates IoU (<span style="font-weight: bold">I</span>ntersection <span style="font-weight: bold">o</span>ver <span style="font-weight: bold">U</span>nion)
for two planes where each plane is described
as an indices array of plane points in the point cloud. 
</p>

<dt style="font-size: 20pt;">Parameters:</dt>
<dd class="field-odd">
    <dl>
    <dt><strong>pred_indices: </strong>
    <span style="font-style: italic;">NDArray[Any, np.int32]</span>
    </dt>
    <dd>
        <p>Array containing indices of the plane corresponding points as a result of segmentation</p>
    </dl>
</dd>
<dd class="field-odd">
    <dl>
    <dt><strong>gt_indices: </strong>
    <span style="font-style: italic;">NDArray[Any, np.int32]</span>
    </dt>
    <dd>
        <p>Array containing indices of the plane corresponding points as ground truth segmentation</p>
    </dl>
</dd>
<dt style="font-size: 20pt;">Returns:</dt>
<dd class="field-odd">
    <dl>
    <dt><strong>iou_value: </strong>
    <span style="font-style: italic;">np.float64</span>
    </dt>
    <dd>
        <p>IoU value for planes</p>
    </dl>
</dd>

---

<p style="font-size: 20pt;">
    Example:
</p>

```bash
>>> pred_indices = np.array([1, 2, 3, 4])
>>> gt_indices = np.array([1, 2, 3, 4])
>>> iou(pred_indices, gt_indices)
1
```