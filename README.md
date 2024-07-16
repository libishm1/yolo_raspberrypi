Deploying YOLOv5 Model using TorchScript on a Raspberry Pi
    Overview
This guide walks you through deploying a YOLOv5 model trained in WSL on a Raspberry Pi. The steps include converting the trained model to TorchScript and running inference on the Raspberry Pi.

    Prerequisites

Hardware
-     Raspberry Pi 4 Model B (4GB or 8GB recommended)
-     A microSD card with Raspberry Pi OS
-     USB Camera (optional, for real-time inference)
  
Software
-     Python 3.x
-     PyTorch
-     OpenCV
Necessary libraries: numpy, argparse
Steps
    1. Converting the Model to TorchScript
After training your YOLOv5 model, convert it to TorchScript for deployment.

    2. Setting Up the Raspberry Pi
Install the necessary software on your Raspberry Pi:

sh
Copy code
sudo apt-get update
sudo apt-get install python3-pip
pip3 install torch torchvision opencv-python numpy
3. Running Inference on the Raspberry Pi
Copy the best.torchscript.pt model file to your Raspberry Pi. Create an inference script inference_rpi.py.

Running Inference
To run inference, use the following command:

sh
Copy code
python3 inference_rpi.py --model best.torchscript.pt --image /path/to/input_image.jpg --output /path/to/output_image.jpg
Conclusion
You have now successfully deployed a YOLOv5 model using TorchScript on a Raspberry Pi. Adjust the confidence threshold and other parameters as needed to optimize the performance for your specific use case.
