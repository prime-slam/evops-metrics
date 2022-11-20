<p style="text-align:center">
    <img src="./docs/_static/logo.png" width="250" height="250"/>
</p>

# EVOPS: library for evaluating plane segmentation algorithms
[![Build and publish](https://github.com/Perception-Solutions/evops/actions/workflows/ci.yml/badge.svg)](https://github.com/Perception-Solutions/evops/actions/workflows/ci.yml)

<p style="font-size: 14pt;">
     EVOPS is an open-source python library that provides various metrics for evaluating the results of the algorithms for segmenting and associating planes from point clouds collected from LIDARs and RGBD devices. 
</p>

<p style="font-size: 14pt;">
     List of metrics implemented in the library:
</p>

<ul style="font-size: 14pt;">
    <li>Summary segmentation metrics <ul style="font-size: 14pt;">
        <li><a href="https://prime-slam.github.io/evops-metrics/instance_based/panoptic">Panoptic</a></li>
        <li><a href="https://prime-slam.github.io/evops-metrics/full_statistics/full_statistics">Full statistics</a></li>
    </ul></li>
    <li>Instance-based segmentation metrics
        <ul style="font-size: 14pt;">
            <li><a href="https://prime-slam.github.io/evops-metrics/#/metrics/instance_based/precision">Precision</a></li>
            <li><a href="https://prime-slam.github.io/evops-metrics/#/metrics/instance_based/recall">Recall</a></li>
            <li><a href="https://prime-slam.github.io/evops-metrics/#/metrics/instance_based/fScore">F-Score</a></li>
            <li><a href="https://prime-slam.github.io/evops-metrics/#/metrics/instance_based/usr">Under segmented ratio</a></li>
            <li><a href="https://prime-slam.github.io/evops-metrics/#/metrics/instance_based/osr">Over segmented ratio</a></li>
            <li><a href="https://prime-slam.github.io/evops-metrics/#/metrics/instance_based/noise">Noise ratio</a></li>
            <li><a href="https://prime-slam.github.io/evops-metrics/#/metrics/instance_based/missed">Missed ratio</a></li>
    </ul></li>
    <li>Point-based segmentation metrics
        <ul style="font-size: 14pt;">
            <li><a href="https://prime-slam.github.io/evops-metrics/#/metrics/point_based/iou">Intersection over Union (IoU)</a></li>
            <li><a href="https://prime-slam.github.io/evops-metrics/#/metrics/point_based/dice">Dice</a></li>
            <li><a href="https://prime-slam.github.io/evops-metrics/#/metrics/point_based/mean">Mean of some metric for matched instances</a></li>
    </ul></li>
</ul>

<p style="font-size: 14pt;">
    For more, please visit the <a href="https://prime-slam.github.io/evops-metrics">EVOPS documentation</a>.
</p>
<p style="font-size: 14pt;">
    You can also find full information about the project on the <a href="https://evops.netlify.app/">EVOPS project website</a>.
</p>

# Python quick start

<p style="font-size: 14pt;">
     Library can be installed using the pip package manager:
</p>

```bash
$ # Install package
$ pip install evops

$ # Check installed version of package
$ pip show evops
```

# Example of usage

<p style="font-size: 14pt;">
    Below is an example of using the precision metric:
</p>

```bash
>>> from evops.metrics import precision
>>> pred_labels = np.array([1, 1, 3, 3])
>>> gt_labels = np.array([2, 2, 0, 3])
>>> tp_condition = "iou"
>>> precision(pred_labels, gt_labels, tp_condition)
0.5
```

# Citation
```
@misc{kornilova2022evops,
      title={EVOPS Benchmark: Evaluation of Plane Segmentation from RGBD and LiDAR Data}, 
      author={Anastasiia Kornilova, Dmitrii Iarosh, Denis Kukushkin, Nikolai Goncharov, Pavel Mokeev, Arthur Saliou, Gonzalo Ferrer},
      year={2022},
      eprint={2204.05799},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# License

<p style="font-size: 14pt;">
    This project is licensed under the Apache License - see the <a href="https://github.com/Perception-Solutions/evops/blob/main/LICENSE">LICENSE</a> file for details.
</p>