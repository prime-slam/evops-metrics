<p style="font-size: 30pt; font-weight: bold;">
    metrics.panoptic
</p>

<p style="font-size: 20pt; font-weight: bold;">
    metrics.<span style="color: red;">panoptic</span>(pred_labels, gt_labels, tp_condition) <a href="https://github.com/prime-slam/evops-metrics/blob/release-1.0/src/evops/metrics/instance_based.py#L74">[source]</a>
</p>

Evaluates panoptic metric (adoptation from <a href="https://arxiv.org/abs/1801.00868">Kirillov A. "Panoptic Segmentation"</a>) for plane detection algorithm. 
It is like all-in-one metric which shows multiplication of two quality levels: pixel perfect and plane instance.
This metric can be used if you need 'single number' metric for your algorithm.

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
        <p>Metric function for which mean value will be calculated. It is used for pixel perfect part of this metric.
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
    <dt><strong>panoptic_value: </strong>
    <span style="font-style: italic;">np.float64</span>
    </dt>
    <dd>
        <p>Calculated panoptic metric value</p>
    </dl>
</dd>

---

<p style="font-size: 20pt;">
    Example:
</p>

```bash
>>> pred_labels = np.array([1, 2, 1, 1, 1])
>>> gt_labels = np.array([1, 1, 1, 1, 1])
>>> metric = dice
>>> tp_condition = "iou"
>>> panoptic(pred_labels, gt_labels, metric, tp_condition)
0.53
```