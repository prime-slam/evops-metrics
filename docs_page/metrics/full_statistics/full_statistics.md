<p style="font-size: 30pt; font-weight: bold;">
    metrics.full_statistics
</p>

<p style="font-size: 20pt; font-weight: bold;">
    metrics.<span style="color: red;">full_statistics</span>(pred_labels, gt_labels, tp_condition) <a href="https://github.com/prime-slam/evops-metrics/blob/release-1.0/src/evops/metrics/full_statistics.py#L26">[source]</a>
</p>

Evaluates full statistics for provided prediction and ground truth data.
This function just calculates each detection metric and return dictionary with values for all of them.

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
    <dt><strong>metric: </strong>
    <span style="font-style: italic;">Callable[[NDArray[Any, np.int32], NDArray[Any, np.int32]], np.float64]</span>
    </dt>
    <dd>
        <p>Metric function for which you want to get the mean value.
Possible values from this library: <code>metrics.iou</code> and <code>metrics.dice</code></p>
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
    <dt><strong>full_statistics: </strong>
    <span style="font-style: italic;">Dict[str, np.float64]</span>
    </dt>
    <dd>
        <p>Dictionary with all supported plane detection metric names and their values</p>
    </dl>
</dd>

---

<p style="font-size: 20pt;">
    Example:
</p>

```bash
>>> pred_labels = np.array([1, 1, 1, 1])
>>> gt_labels = np.array([1, 1, 2, 4])
>>> metric = dice
>>> full_statistics(pred_labels, gt_labels, metric, "iou")
{
    'panoptic': 0.25,
    'precision': 1.0, 
    'recall': 0.33, 
    'fScore': 0.5,
    'usr': 1.0, 
    'osr': 0.0, 
    'noise': 0.0, 
    'missed': 0.66, 
    'mean': 0.5
}
```