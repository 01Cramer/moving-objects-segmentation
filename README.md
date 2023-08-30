This is software made for moving objects segmenatation build with detectron2 library. It is mainly used to improve depth maps.

## Results
![](https://github.com/01Cramer/moving-objects-segmentation/blob/main/test_on_sample_8.gif)

## Instalation
For instalation refer to the [following](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) and install detectron from a local clone (I had some issues when installing in other way).

Other requirements:
- pip install scikit-learn
- pip install scipy

Then download moving_objects_segmentation.py from this repository and put it in detectron2 folder.

## Overview
The algorithm is based on comparing two consecutive frames. Detected object masks from the two frames are compared using the jaccard_score function, which returns the similarity of masks. Masks with the highest similarity are then compared pixel by pixel through iteration of the mask with the smaller bounding box area. I assume that if the highest similarity is below 0.50, it means that there are no corresponding masks, and the highest score is a result of noise or other distractions, causing the object mask to be removed. After iterating through the pixels, the average correlation between masks is computed. Objects that are not moving exhibit a very high correlation, around 1.0, whereas the correlation is lower for objects in motion. I assume that a correlation under 0.99 serves as a threshold for identifying moving objects. Any mask with a higher correlation is subsequently removed, resulting in a final collection of masks representing only the moving objects.

Although there is still plenty of room for improvement.



