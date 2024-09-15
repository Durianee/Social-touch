Here is a description of each file: 

emotion_classification_pipeline.py: This is the main script of the project, responsible for implementing the fused model that combines features from 3D ResNet50 and OpenPose for emotion classification.

emotion_openpose_only.py: This script contains the implementation of the OpenPose-only model, which utilizes body keypoint data extracted via OpenPose for emotion classification.

emotion_resnet_only.py: This script contains the code for the ResNet-only model, which uses features extracted solely from a 3D ResNet50 model for emotion classification.

extract_features_resnet.py: This script is used to extract features from the Global Average Pooling (GAP) layer of the 3D ResNet50 model. The script requires a pretrained 3D ResNet50 model, which can be found at the following repository: 3D ResNet50 Moments Models.

extract_keypoints_openpose.py: This script is used to extract keypoint features using OpenPose. It is designed to run in a Windows environment with a GPU configuration. The script processes video frames and extracts keypoints, such as body, face, and hand keypoints, for emotion analysis.

emotion_label.xlsx: An Excel file that contains the emotion labels for the corresponding video data, where emotions are categorized for each video.
