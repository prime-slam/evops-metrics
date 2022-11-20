# Introduction

<img src="./static/img/logo.png" width="350" height="350" />

<p style="font-size: 15pt; font-weight: bold;">
     EVOPS: library for evaluating plane segmentation and association algorithms.
</p>

<p style="font-size: 14pt;">
     EVOPS is an open-source python library that provides various metrics for evaluating the results of the algorithms for segmenting and associating planes from point clouds collected from LIDARs and RGBD devices. 
</p>

[![Build and publish](https://github.com/prime-slam/evops-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/prime-slam/evops-metrics/actions/workflows/ci.yml)


<p style="font-size: 14pt;">
     List of metrics implemented in the library:
</p>

<ul style="font-size: 14pt;">
    <li>Summary segmentation metrics <ul style="font-size: 14pt;">
        <li><a href="#/metrics/instance_based/panoptic">Panoptic</a></li>
        <li><a href="#/metrics/full_statistics/full_statistics">Full statistics</a></li>
    </ul></li>
    <li>Instance-based segmentation metrics
        <ul style="font-size: 14pt;">
            <li><a href="#/metrics/instance_based/precision">Precision</a></li>
            <li><a href="#/metrics/instance_based/recall">Recall</a></li>
            <li><a href="#/metrics/instance_based/fScore">F-Score</a></li>
            <li><a href="#/metrics/instance_based/usr">Under segmented ratio</a></li>
            <li><a href="#/metrics/instance_based/osr">Over segmented ratio</a></li>
            <li><a href="#/metrics/instance_based/noise">Noise ratio</a></li>
            <li><a href="#/metrics/instance_based/missed">Missed ratio</a></li>
    </ul></li>
    <li>Point-based segmentation metrics
        <ul style="font-size: 14pt;">
            <li><a href="#/metrics/point_based/iou">Intersection over Union (IoU)</a></li>
            <li><a href="#/metrics/point_based/dice">Dice</a></li>
            <li><a href="#/metrics/point_based/mean">Mean of some metric for matched instances</a></li>
    </ul></li>
</ul>

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

# Metrics description
EVOPS-Metrics provides a lot of various segmentation and association metrics which are different in their implementation and usecases.

### Summary segmentation metrics
Summary segmentation metrics can be used to get full image of algorithm quality: by one number in panoptic or by values of all implemented metrics in full statistics.

#### Panoptic
Built on the base of description from metric described in <a href="https://arxiv.org/abs/1801.00868">Kirillov A. "Panoptic Segmentation"</a> is the best choice for one number description of the plane segmentation algorithm. It takes into account both instance and point level quality.

*Example*

| Original image                                                                         | Ground truth planes                                                                   | Predicted planes                                                                                |
|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------| 
| ![](https://github.com/prime-slam/evops-metrics/blob/docs_page/img/orig.png?raw=true) | ![](https://github.com/prime-slam/evops-metrics/blob/docs_page/img/gt.png?raw=true) | ![](https://github.com/prime-slam/evops-metrics/blob/docs_page/img/prediction_panoptic.png?raw=true) |

In this example you can see that algorithm skipped in his prediction yellow plane (cup board part).
Also, prediction has problems with some small parts of planes: it treated picture frame as a part of the picture and leaves from flower as a part of the wall.
Both of described issues influence metrics like F-Score and mean respectively. Moreover, you can see that panoptic metric gathers both influence and is lower than both Mean and F-Score.

| Metric   | Value |
|----------|-------|
| Mean     | 0.96  |
| F-Score  | 0.9   |
| Panoptic | 0.88  |

#### Full statistics
It is just gathering of all implemented metrics in this library: they are evaluated independently and concatenated into single-dictionary result.

### Instance-based segmentation metrics
Instance-based segmentation metrics can be used to define quality of plane segmentation algorithm in plane scale.
These metrics show how precisely it finds each plane and separates them from each other.
In each instance-based metric extra function (IoU by default) is used to match plane from prediction with its ground truth representation.
It is done by evaluation of this function for each pair of planes (one from prediction and the other one from gt) and choosing pair with evaluation result <code>> IOU_THRESHOLD_FULL</code>.

#### Precision, Recall, F-Score
Classic metrics which show main algorithm quality numbers: which part of segmented planes are found in ground truth data, which part of ground truth planes were segmented and their f-score combination.
These metrics can be used to rough description of the algorithm quality without taking into an account small pixel mistakes (according to matching threshold).

*Example*

| Original image                                                                         | Ground truth planes                                                                   | Predicted planes                                                                         |
|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------| 
| ![](https://github.com/prime-slam/evops-metrics/blob/docs_page/img/orig.png?raw=true) | ![](https://github.com/prime-slam/evops-metrics/blob/docs_page/img/gt.png?raw=true) | ![](https://github.com/prime-slam/evops-metrics/blob/docs_page/img/prediction_fscore.png?raw=true) |

In this example you can see that algorithm predicted one extra gray plane (picture frame) and skipped yellow plane (cup board part).
It was detected by metrics in the next way: precision becomes less than one as one of the predicted planes doesn't match with any plane from ground truth
and is treated as incorrectly detected. Recall is lower than 1 as for one plane from ground truth no match is found in prediction, so it is treated as not detected.

| Metric    | Value   |
|-----------|---------|
| Precision | 0.83    |
| Recall    | 0.83    |
| F-Score   | 0.83    |


#### Under/Over Segmented Ratio
Idea of these metrics is to show plane separation quality of the plane segmentation algorithm. 
Over Segmented Ratio (OSR) indicates how many planes from ground truth data were separated into small pieces by the algorithm.
Under Segmented Ratio (USR) vice versa describes ratio of planes from prediction which are represented by more than one plane in ground truth.

*Example*

| Original image                                                                          | Ground truth planes                                                             | Predicted planes                                                                           |
|-----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------| 
| ![](https://github.com/prime-slam/evops-metrics/blob/docs_page/img/orig.png?raw=true) | ![](https://github.com/prime-slam/evops-metrics/blob/docs_page/img/gt.png?raw=true) | ![](https://github.com/prime-slam/evops-metrics/blob/docs_page/img/prediction_osr.png?raw=true) |

In this example you can see that algorithm treated two plans as one (wall and picture) but separated two planes in the other wall.
It was detected by metrics in the next way: OSR becomes bigger than zero as one of the ground truth planes is partly matched (overlapped by at least 20%) with two plans from prediction (wall with two predicted planes). So we have 1 of 6 planes with over segmentation, so ratio is `1 / 6 = 0.17`
USR is 0.17 as one plane from prediction (wall with picture) partly matches two planes from ground truth, so we have 1 of 6 detected planes with under segmentation  and ratio is `1 / 6 = 0.17`.

| Metric | Value |
|--------|-------|
| OSR    | 0.17  |
| USR    | 0.17  |

#### Noise/Missed ratio
These metrics can be used to define ratio of fully not segmented/phantom segmented planes in the algorithm prediction

*Example*

| Original image                                                                         | Ground truth planes                                                                   | Predicted planes                                                                         |
|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------| 
| ![](https://github.com/prime-slam/evops-metrics/blob/docs_page/img/orig.png?raw=true) | ![](https://github.com/prime-slam/evops-metrics/blob/docs_page/img/gt.png?raw=true) | ![](https://github.com/prime-slam/evops-metrics/blob/docs_page/img/prediction_fscore.png?raw=true) |

In this example you can see that algorithm predicted one extra gray plane (picture frame) and skipped yellow plane (cup board part).
It was detected by metrics in the next way: noise becomes bigger than zero as one of the predicted planes doesn't match with any plane from ground truth
and is treated as incorrectly detected. As we totally have 6 detected planes and one of them is noise than `noise_ratio = 1 / 6 = 0.17`.
Missed is 0.17 as for one plane from ground truth no match is found in prediction, so it is treated as not detected and, as we have 6 planes in ground truth data, missed ratio is `1 / 6 = 0.17`. 
It can be seen that missed and noise are just inverted recall and precision. 

| Metric       | Value |
|--------------|-------|
| Noise_ratio  | 0.17  |
| Missed_ratio | 0.17  |

### Point-based segmentation metrics
Point-based segmentation metrics can be used as more precise tool for depicting if algorithm fully fit ground truth data with predicted planes.
It is evaluated only for matched planes (when we already know which plane from ground truth was predicted by this plane instance) and indicates
how much difference between prediction and ground truth we have in the level of pixels.

#### Mean of some metric for matched instances
This metric just provides mean value of other plane-to-plane metric application only to matched planes. IoU or Dice are examples of such plane-to-plane metrics.
We can use this metric to show how precise plane segmentation algorithm matches borders of plane in average.

*Example*

| Original image                                                                         | Ground truth planes                                                                   | Predicted planes 1                                                                            | Predicted planes 2                                                                          |
|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| ![](https://github.com/prime-slam/evops-metrics/blob/docs_page/img/orig.png?raw=true) | ![](https://github.com/prime-slam/evops-metrics/blob/docs_page/img/gt.png?raw=true) | ![](https://github.com/prime-slam/evops-metrics/blob/docs_page/img/prediction_fscore.png?raw=true) | ![](https://github.com/prime-slam/evops-metrics/blob/docs_page/img/prediction_mean.png?raw=true) |

In this example you can see two different predictions: in the first one algorithm predicted one extra gray plane (picture frame) and skipped yellow plane (cup board part);
in the second one algorithm guesses all planes correctly but has problems with some small parts of them: it treated picture frame as a part of the picture and leaves from flower as a part of the wall.
That is why mean value of the first prediction is `1` as all matched planes fits ideally. On the other side prediction two has only `0.97` mean value (ratio of correctly segmented pixels to totally segmented) as two planes have small differences in pixels with their gt prototypes.

| Metric                | Value |
|-----------------------|-------|
| Mean_IoU_prediction_1 | 1     |
| Mean_IoU_prediction_2 | 0.97  |

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
    This project is licensed under the Apache License - see the <a href="https://github.com/prime-slam/evops-metrics/blob/main/LICENSE">LICENSE</a> file for details.
</p>