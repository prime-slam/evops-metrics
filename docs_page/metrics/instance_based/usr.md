<p style="font-size: 30pt; font-weight: bold;">
    metrics.usr
</p>

<p style="font-size: 20pt; font-weight: bold;">
    metrics.<span style="color: red;">usr</span>(pred_labels, gt_labels, tp_condition) <a href="https://github.com/prime-slam/evops-metrics/blob/release-1.0/src/evops/metrics/instance_based.py#L95">[source]</a>
</p>

Evaluates USR (<span style="font-weight: bold">U</span>nder <span style="font-weight: bold">S</span>egmented <span style="font-weight: bold">R</span>atio) for plane detection algorithm. 
It shows which part of detected planes covers more than one plane from ground truth data.

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
    <dt><strong>usr_value: </strong>
    <span style="font-style: italic;">np.float64</span>
    </dt>
    <dd>
        <p>Calculated USR</p>
    </dl>
</dd>

---

<p style="font-size: 20pt;">
    Example:
</p>

```bash
>>> pred_labels = np.array([1, 1, 3, 3])
>>> gt_labels = np.array([2, 2, 1, 3])
>>> tp_condition = "iou"
>>> usr(pred_labels, gt_labels, tp_condition)
0.5
```