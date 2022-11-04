# Copyright (c) 2022, Pavel Mokeev, Dmitrii Iarosh, Anastasiia Kornilova
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Label id which depicts that pixel is not a part of any plane
UNSEGMENTED_LABEL = 0

# Threshold for IoU overlap which defines planes as overlapped enough to be treated
# as a matched pair of gt and predicted plane for instance based metrics calculation
IOU_THRESHOLD_FULL = 0.75

# Threshold for IoU overlap which defines planes as overlapped enough to be treated
# as a partly matched pair of gt and predicted plane for over and under segmentation calculation
IOU_THRESHOLD_PART = 0.2
