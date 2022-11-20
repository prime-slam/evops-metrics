<p style="font-size: 30pt; font-weight: bold;">
    metrics.mean
</p>

<p style="font-size: 20pt; font-weight: bold;">
    metrics.<span style="color: red;">mean</span>(pred_labels, gt_labels, tp_condition) <a href="https://github.com/prime-slam/evops-metrics/blob/release-1.0/src/evops/metrics/point_based.py#L56">[source]</a>
</p>

Evaluates mean of the selected metric across all planes which can be matched
from prediction to ground truth using <code>constants.IOU_THRESHOLD_FULL</code>.
This metric can be used to show how exactly plane detection algorithms segment planes on the level of points.
It is pixel perfect metric that can be used for final algorithm tuning.

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
    <dt><strong>mean_value: </strong>
    <span style="font-style: italic;">np.float64</span>
    </dt>
    <dd>
        <p>Mean value of the selected metric calculated only for the matched planes</p>
    </dl>
</dd>

---

<p style="font-size: 20pt;">
    Example:
</p>

```bash
>>> pred_labels = np.array([1, 1, 1])
>>> gt_labels = np.array([1, 1, 1])
>>> metric = dice
>>> mean(pred_labels, gt_labels, metric, "iou")
1
```