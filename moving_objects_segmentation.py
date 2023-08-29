# basic setup: 
import torch, torchvision, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import common libraries
import numpy as np
import pandas as pd
import tqdm
import cv2
from scipy import stats
from sklearn.metrics import jaccard_score
from statistics import correlation

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes
from detectron2.projects import point_rend
import time

################# ADDITIONAL FUNCTIONS ##################

def convert(outputs): # easier access (syntax) to detected objects 
  cls = outputs['instances'].pred_classes
  masks = outputs['instances'].pred_masks
  boxes = outputs['instances'].pred_boxes

  # create new instance obj and set its fields
  obj = detectron2.structures.Instances(image_size=(height, width))
  obj.set('pred_classes', cls)
  obj.set('pred_masks', masks)
  obj.set('pred_boxes', boxes)
  return obj

def remove_given_indxes(outputs, indx_to_remove):
    cls = outputs.pred_classes
    masks = outputs.pred_masks
    boxes = outputs.pred_boxes

    # delete corresponding arrays
    cls = np.delete(cls.cpu().numpy(), indx_to_remove)
    masks = np.delete(masks.cpu().numpy(), indx_to_remove, axis=0)
    boxes = np.delete(boxes, indx_to_remove, axis=0)

    # convert back to tensor and move to cuda
    cls = torch.tensor(cls)
    masks = torch.tensor(masks)
    boxes = Boxes(torch.tensor(boxes))

    # create new instance obj and set its fields
    obj = detectron2.structures.Instances(image_size=(height, width))
    obj.set('pred_classes', cls)
    obj.set('pred_masks', masks)
    obj.set('pred_boxes', boxes)
    return obj

################# MODEL PARAMS ##################

video_path = 'sample_0.mp4'
video = cv2.VideoCapture(video_path)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# initialize video writer
video_writer = cv2.VideoWriter('test_sample_0.mp4', fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second), frameSize=(width, height), isColor=True)

# initialize predictor
cfg = get_cfg()
point_rend.add_pointrend_config(cfg)
cfg.MODEL.DEVICE = 'cpu'  # Ustawienie modelu w tryb CPU
cfg.merge_from_file("C:/Users/student/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95  # Set threshold for object detection
cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl"
predictor = DefaultPredictor(cfg)

# initialize visualizer
v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)

################# ALGORITHM ##################

# previous frame features holders
prev_frame = None
prev_boxes = None
prev_masks = None

def runOnVideo(video, max_frames):
    """ Runs the predictor on every frame in the video (unless max_frames is given),
    and returns the frame with the predictions drawn.
    """

    read_frames = 0
    while True:
        has_frame, frame = video.read()
        if not has_frame:
            break

        # get detectron2 prediction results for this frame
        outputs = predictor(frame)
        modified_outputs = convert(outputs)

        new_masks = modified_outputs.pred_masks
        new_boxes = modified_outputs.pred_boxes
        indx_to_remove = [] # list of indexes of objects that are not moving

        if read_frames == 0: # in first frame we cant detect moving objects because algorithm is based on previous frames 
            # updating previous frame features
            prev_frame = frame
            prev_masks = new_masks
            prev_boxes = new_boxes
        else:
            for i in range(len(new_masks)):# iterate through every mask in current frame
                jaccards = [] # list of jaccard scores for pairs of mask in current frame and every mask in previous frame 
                new_mask_array = new_masks[i].numpy() # converting to np.array for jaccard_score function input
                found_in_prev = False # flag in case if in previous frame corresponding mask was not detected  

                for j in range(len(prev_masks)):
                    prev_mask_array = prev_masks[j].numpy() # converting to np.array for jaccard_score function input
                    # calculating jaccard score
                    jaccards.append(jaccard_score(new_mask_array, prev_mask_array, average="micro")) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html
                #print("MAX JACCARD SCORE: ", max(jaccards))
                if max(jaccards) >= 0.5: # threshold for jaccard score
                    found_in_prev = True    
                    corresponding_mask_index = jaccards.index(max(jaccards)) # highest jaccard score -> corresponding mask

                    n_x1 = new_boxes.tensor[i][0]
                    n_y1 = new_boxes.tensor[i][1]
                    n_x2 = new_boxes.tensor[i][2]
                    n_y2 = new_boxes.tensor[i][3]

                    # corresponding mask bouding box
                    p_x1 = prev_boxes.tensor[corresponding_mask_index][0]
                    p_y1 = prev_boxes.tensor[corresponding_mask_index][1]
                    p_x2 = prev_boxes.tensor[corresponding_mask_index][2]
                    p_y2 = prev_boxes.tensor[corresponding_mask_index][3]

                    # compute area of corresponding bouding boxes
                    area_n = (n_x2 - n_x1) * (n_y2 - n_y1)
                    area_p = (p_x2 - p_x1) * (p_y2 - p_y1)

                    pixels_b = []
                    pixels_g = []
                    pixels_r = []

                    pixels_b_prev = []
                    pixels_g_prev = []
                    pixels_r_prev = []

                    # iterate every pixel in smaller bounding box
                    if area_p >= area_n:# box got smaller
                        for y in range(int(n_y1.item()), int(n_y2.item())):
                            for x in range(int(n_x1.item()), int(n_x2.item())):
                                # check if pixel is inside current mask and corresponding mask from previous frame
                                pixel_in_mask = new_masks[i][y][x]
                                pixel_in_prev_mask = prev_masks[corresponding_mask_index][y][x]

                                if pixel_in_mask and pixel_in_prev_mask:
                                    pixel = frame[y,x]             
                                    pixels_b.append(pixel[0])
                                    pixels_g.append(pixel[1])
                                    pixels_r.append(pixel[2])
                                    
                                    pixel_prev = prev_frame[y,x]
                                    pixels_b_prev.append(pixel_prev[0])
                                    pixels_g_prev.append(pixel_prev[1])
                                    pixels_r_prev.append(pixel_prev[2])
                    else:# box got bigger
                        for y in range(int(p_y1.item()), int(p_y2.item())):
                            for x in range(int(p_x1.item()), int(p_x2.item())):
                                # check if pixel is inside current mask and in corresponding mask from previous frame
                                pixel_in_mask = new_masks[i][y][x]
                                pixel_in_prev_mask = prev_masks[corresponding_mask_index][y][x]
                                
                                if pixel_in_mask and pixel_in_prev_mask:
                                    pixel = frame[y,x]
                                    pixels_b.append(pixel[0])
                                    pixels_g.append(pixel[1])
                                    pixels_r.append(pixel[2])
                                    
                                    pixel_prev = prev_frame[y,x]
                                    pixels_b_prev.append(pixel_prev[0])
                                    pixels_g_prev.append(pixel_prev[1])
                                    pixels_r_prev.append(pixel_prev[2])

                    # compute correlation between every color channel and average correlation
                    corr_b = stats.pearsonr(pixels_b, pixels_b_prev).statistic
                    corr_g = stats.pearsonr(pixels_g, pixels_g_prev).statistic
                    corr_r = stats.pearsonr(pixels_r, pixels_r_prev).statistic
                    avg_corr = (corr_b + corr_g + corr_r) / 3
                    print(i)
                    print(avg_corr)
                if avg_corr > 0.99 or not found_in_prev:
                    indx_to_remove.append(i)

        # updating previous frame features
        prev_boxes = new_boxes
        prev_masks = new_masks
        prev_frame = frame

        # removing objects that are not moving
        modified_outputs = remove_given_indxes(modified_outputs, indx_to_remove)

        if read_frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
 
            # Draw a visualization of the predictions using the video visualizer
            visualization = v.draw_instance_predictions(frame, modified_outputs.to("cpu"))
        

            # Convert Matplotlib RGB format to OpenCV BGR format
            visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)

            yield visualization

        read_frames += 1
        if read_frames > max_frames:
            break

# create a cut-off for debugging
num_frames = 17

# enumerate the frames of the video
for visualization in tqdm.tqdm(runOnVideo(video, num_frames), total=num_frames):

    # write test image
    cv2.imwrite('POSE detectron2.png', visualization)

    # write to video file
    video_writer.write(visualization)

# release resources
video.release()
video_writer.release()
cv2.destroyAllWindows()
