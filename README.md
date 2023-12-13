## Moving objects segmentation
This is software made for moving objects segmenatation build with [Detectron2](https://github.com/facebookresearch/detectron2) library.
It has been used to improve depth maps. [Link to paper](http://www.multimedia.edu.pl/publications/m64708-MIV-New-depth-maps-for-selected-CTC-sequences.pdf)

## Results
Results presented are based on a sample short video sequence consisting of 7 frames.
![](https://github.com/01Cramer/moving-objects-segmentation/blob/main/test_on_sample_8.gif)

## Overview
The algorithm is based on comparing two consecutive frames. Detected object masks from the two frames are compared using the jaccard_score function, which returns the similarity of masks. Masks with the highest similarity are then compared pixel by pixel through iteration of the mask with the smaller bounding box area. I assume that if the highest similarity is below 0.50, it means that there are no corresponding masks, and the highest score is a result of noise or other distractions, causing the object mask to be removed. After iterating through the pixels, the average correlation between masks is computed. Objects that are not moving exhibit a very high correlation, around 1.0, whereas the correlation is lower for objects in motion. I assume that a correlation under 0.99 serves as a threshold for identifying moving objects. Any mask with a higher correlation is subsequently removed, resulting in a final collection of masks representing only the moving objects.
Although there is still plenty of room for improvement.

## Citing Detectron2
If you use Detectron2 in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.

@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}



