<p style="font-size: 14pt;">
    The library also defines several constants to which you can assign your own values:
</p>

<ul style="font-size: 14pt;">
      <li>
        evops.metrics.constants.<span style="color: red;">UNSEGMENTED_LABEL</span>
        <p>This value allows you to filter out unallocated points in the point clouds.</p>
      </li>
      <li>
        evops.metrics.constants.<span style="color: red;">IOU_THRESHOLD_FULL</span>
        <p>This value is the threshold to define two planes as correctly matched by IoU metric.</p>
      </li>
      <li>
        evops.metrics.constants.<span style="color: red;">IOU_THRESHOLD_PART</span>
        <p>This value is the threshold to define that planes are partly overlapped by IoU metric.
It is used in oversegmentation and undersegmentation calculations</p>
      </li>
</ul>