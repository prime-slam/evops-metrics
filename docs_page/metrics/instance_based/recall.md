<p style="font-size: 30pt; font-weight: bold;">
    metrics.recall
</p>

<p style="font-size: 20pt; font-weight: bold;">
    metrics.<span style="color: red;">recall</span>(pred_labels, gt_labels, tp_condition) <a href="https://github.com/prime-slam/evops-metrics/blob/release-1.0/src/evops/metrics/instance_based.py#L42">[source]</a>
</p>

Evaluates recall of plane detection algorithm.

<dt style="font-size: 20pt;">Parameters:</dt>
<dd class="field-odd">
    <dl>
    <dt><strong>pred_labels: </strong>
    <span style="font-style: italic;">NDArray[Any, np.int32]</span>
    </dt>
    <dd>
        <p>Array containing the labels of points obtained as a result of segmentation</p>
    </dl>
</dd>
<dd class="field-odd">
    <dl>
    <dt><strong>gt_labels: </strong>
    <span style="font-style: italic;">NDArray[Any, np.int32]</span>
    </dt>
    <dd>
        <p>Array containing the reference labels of point cloud</p>
    </dl>
</dd>
<dd class="field-odd">
    <dl>
    <dt><strong>tp_condition: </strong>
    <span style="font-style: italic;">string</span>
    </dt>
    <dd>
        <p>Helper function to match planes from predicted to reference ones. Possible values: <code>"iou"</code></p>
    </dl>
</dd>
<dt style="font-size: 20pt;">Returns:</dt>
<dd class="field-odd">
    <dl>
    <dt><strong>recall_value: </strong>
    <span style="font-style: italic;">np.float64</span>
    </dt>
    <dd>
        <p>Calculated recall</p>
    </dl>
</dd>

---

<p style="font-size: 20pt;">
    Example:
</p>

```bash
>>> pred_labels = np.array([1, 1, 3, 3])
>>> gt_labels = np.array([2, 2, 0, 3])
>>> tp_condition = "iou"
>>> recall(pred_labels, gt_labels, tp_condition)
0.5
```