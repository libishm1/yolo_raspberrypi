Deploying YOLOv5 Model using TorchScript on a Raspberry Pi
Overview
This guide walks you through deploying a YOLOv5 model trained in WSL on a Raspberry Pi. The steps include converting the trained model to TorchScript and running inference on the Raspberry Pi.

Prerequisites

Hardware
- Raspberry Pi 4 Model B (4GB or 8GB recommended)
- A microSD card with Raspberry Pi OS
- USB Camera (optional, for real-time inference)
  
Software
- Python 3.x
- PyTorch
- OpenCV
Necessary libraries: numpy, argparse
Steps
    1. Converting the Model to TorchScript
After training your YOLOv5 model, convert it to TorchScript for deployment.

    2. Setting Up the Raspberry Pi
Install the necessary software on your Raspberry Pi:

    sh
Copy code
-   sudo apt-get updatesu
-   sudo apt-get install python3-pip
-    pip3 install torch torchvision opencv-python numpy

   install necessary software,

sudo apt-get update

sudo apt-get install python3-pip

-    sudo apt install python3-torch python3-torchvision python3-opencv python3-numpy

   3. Navigating to the correct directory
      
sudo apt update
sudo apt install git

Make a path and clone to the directory

cd home

mkdir image_detection

cd image_detection

git clone https://github.com/libishm1/yolo_raspberrypi.git

cd yolo_raspberrypi


Give permission to the python script 

chmod +x inference-rpi.py

  
4. Running Inference on the Raspberry Pi

Running Inference
To run inference, use the following command:

    python3 inference_rpi.py --model best.torchscript.pt --image metal55.jpg --output detected.jpg
Conclusion
You have now successfully deployed a YOLOv5 model using TorchScript on a Raspberry Pi. Adjust the confidence threshold and other parameters as needed to optimize the performance for your specific use case.
