#video2image
import os
import cv2
from imageai.Detection import ObjectDetection
execution_path = os.getcwd()


def extract_keyframe(video_file,output_file,fps):
    out_keyframe_file = os.path.join(output_file,"keyframe")
    if not os.path.exists(out_keyframe_file):
        os.makedirs(out_keyframe_file)

    videos_list = os.listdir(video_file)
    for video in videos_list:
        video_dir = os.path.join(video_file,video)
        keyframe_video_dir = os.path.join(out_keyframe_file,video[0:-4])
        if not os.path.exists(keyframe_video_dir):
            os.makedirs(keyframe_video_dir)
        comment = "ffmpeg -i " + video_dir + " -f image2 -r " + str(fps) + " " + keyframe_video_dir + "/%5d.jpg"
        os.system(comment)


def detection(output_file):
    # keyframe file for detection
    keyframe_file = os.path.join(output_file,"keyframe")

    # save detection results file for keyframe
    output_full_image_file = os.path.join(output_file, "result_keyframe")
    if not os.path.exists(output_full_image_file):
        os.makedirs(output_full_image_file)

    # save detection results file for cropped person
    output_crop_person_file = os.path.join(output_file, "result_crop")
    if not os.path.exists(output_crop_person_file):
        os.makedirs(output_crop_person_file)

    # detection
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath( os.path.join(execution_path, "yolo.h5"))
    detector.loadModel()
    videos_list = os.listdir(keyframe_file)
    videos_list.sort()
    for video in videos_list:
        # mkdir files for result
        output_video_keyframe_file = os.path.join(output_full_image_file,video)
        if not os.path.exists(output_video_keyframe_file):
            os.makedirs(output_video_keyframe_file)

        output_video_crop_file = os.path.join(output_crop_person_file,video)
        if not os.path.exists(output_video_crop_file):
            os.makedirs(output_video_crop_file)

        # read keyframe and detect
        video_keyframe_dir = os.path.join(keyframe_file,video)
        keyframes_list = os.listdir(video_keyframe_dir)
        keyframes_list.sort()
        for keyframe in keyframes_list:
            keyframe_dir = os.path.join(video_keyframe_dir,keyframe)
            res_keyframe_dir = os.path.join(output_video_keyframe_file,keyframe)

            # detect and save results for keyframes
            detections = detector.detectObjectsFromImage(input_image=keyframe_dir,
                                                     output_image_path=res_keyframe_dir)
            ori_image = cv2.imread(keyframe_dir)
            count = 1
            for eachObject in detections:
                if eachObject["name"] == 'person':
                    bb = eachObject['box_points']
                    person_image = ori_image[bb[1]:bb[3],bb[0]:bb[2]]
                    person_image_dir = os.path.join(output_video_crop_file, keyframe[0:-4] + "_" + str(count).zfill(2) + ".jpg")
                    cv2.imwrite(person_image_dir, person_image)
                    count += 1
def result2video(output_file):
    # save detection results file for videos
    output_video_file = os.path.join(output_file,"result_video")
    if not os.path.exists(output_video_file):
        os.makedirs(output_video_file)
    res_keframe_file = os.path.join(output_file,"result_keyframe")
    res_keframe_videos_list = os.listdir(res_keframe_file)
    res_keframe_videos_list.sort()
    for res_keframe_video in res_keframe_videos_list:
        res_video_dir = os.path.join(output_video_file, res_keframe_video + ".mp4")
        res_keframe_video_dir = os.path.join(res_keframe_file,res_keframe_video,"%5d.jpg")
        comment = "ffmpeg -f image2 -i " + res_keframe_video_dir + " -vcodec libx264 -r 25 " + res_video_dir
        os.system(comment)

def detector(video_file,output_file,fps):
    extract_keyframe(video_file,output_file,fps)
    detection(output_file)
    result2video(output_file)