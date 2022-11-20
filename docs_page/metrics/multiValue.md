<p style="font-size: 30pt; font-weight: bold;">
    metrics.multi_value
</p>

<p style="font-size: 20pt; font-weight: bold;">
    metrics.<span style="color: red;">multi_value</span>(pred_labels, gt_labels, overlap_threshold) <a href="https://github.com/MobileRoboticsSkoltech/evops/blob/release/0.1/src/evops/metrics/metrics.py#L132">[source]</a>
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
<dd class="field-odd">
    <dl>
    <dt><strong>overlap_threshold: </strong>
    <span style="font-style: italic;">np.float64</span>
    </dt>
    <dd>
        <p>minimum value at which the planes are considered intersected</p>
    </dl>
</dd>
<dt style="font-size: 20pt;">Returns:</dt>
<dd class="field-odd">
    <dl>
    <dt><strong>multi_value_result: </strong>
    <span style="font-style: italic;">Dict[string, np.float64]</span>
    </dt>
    <dd>
        <p>Dictionary with keys 
        <span style="font-style: italic;">"precision"</span>,
        <span style="font-style: italic;">"recall"</span>,
        <span style="font-style: italic;">"under_segmented"</span>,
        <span style="font-style: italic;">"over_segmented"</span>,
        <span style="font-style: italic;">"missed"</span>,
        <span style="font-style: italic;">"noise"</span>
        and corresponding values for point cloud.</p>
    </dl>
</dd>

---

<p style="font-size: 20pt;">
    Example:
</p>

```bash
TODO
```