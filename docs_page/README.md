# Introduction

<img src="./static/img/logo.png" width="350" height="350" />

<p style="font-size: 15pt; font-weight: bold;">
     EVOPS: library for evaluating plane segmentation algorithms.
</p>

<p style="font-size: 14pt;">
     EVOPS is an open-source python library that provides various metrics for evaluating the results of the algorithms for segmenting planes from point clouds collected from LIDARs and RGBD devices. 
</p>

[![Build and publish](https://github.com/MobileRoboticsSkoltech/evops/actions/workflows/ci.yml/badge.svg)](https://github.com/MobileRoboticsSkoltech/evops/actions/workflows/ci.yml)


<p style="font-size: 14pt;">
     List of metrics implemented in the library:
</p>

<ul style="font-size: 14pt;">
      <li>Intersection over Union (IoU)</li>
      <li>Dice </li>
      <li>Precision</li>
      <li>Recall</li>
      <li>Mean of some metric</li>
      <li>Under segmented percent</li>
      <li>Over segmented percent</li>
      <li>Noise percent</li>
      <li>Missed percent</li>
</ul>

# Python quick start

<p style="font-size: 14pt;">
     Library can be installed using the pip package manager:
</p>

```bash
$ pip install evops
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