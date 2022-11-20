<p style="font-size: 30pt; font-weight: bold;">
    metrics.iou
</p>

<p style="font-size: 20pt; font-weight: bold;">
    metrics.<span style="color: red;">iou</span>(pred_labels, gt_labels) <a href="https://github.com/MobileRoboticsSkoltech/evops/blob/release/0.1/src/evops/metrics/metrics.py#L31">[source]</a>
</p>

<dt style="font-size: 20pt;">Parameters:</dt>
<dd class="field-odd">
    <dl>
    <dt><strong>pred_labels: </strong>
    <span style="font-style: italic;">NDArray[Any, np.int32]</span>
    </dt>
    <dd>
        <p>Array containing the labels of the corresponding points as a result of segmentation</p>
    </dl>
</dd>
<dd class="field-odd">
    <dl>
    <dt><strong>gt_labels: </strong>
    <span style="font-style: italic;">NDArray[Any, np.int32]</span>
    </dt>
    <dd>
        <p>Array containing the labels of the corresponding points as grount truth segmentation</p>
    </dl>
</dd>
<dt style="font-size: 20pt;">Returns:</dt>
<dd class="field-odd">
    <dl>
    <dt><strong>iou_value: </strong>
    <span style="font-style: italic;">np.float64</span>
    </dt>
    <dd>
        <p>IoU value for point cloud.</p>
    </dl>
</dd>

---

<p style="font-size: 20pt;">
    Example:
</p>

```bash
>>> pred_labels = np.array([1, 2, 3, 4])
>>> gt_labels = np.array([1, 2, 3, 4])
>>> iou(pred_labels, gt_labels)
1
```