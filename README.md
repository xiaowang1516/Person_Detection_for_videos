## Person Detection
## Installation
apt-get install ffmpeg
pip install opencv-python keras numpy pillow scipy h5py tensorflow
https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl 

## person detection
python main.py 
"from detector import detector as Det
input = "/home/wx/work/videos/"
output = "/home/wx/work/result/"
fps = 8
Det(input,output,fps)
"
input % videos file
output % genetated files 1)keyframe 2)result_crop 3)result_keyframe 4)result_video