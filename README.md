
# Convolution Operations

This repository contains Python scripts for performing various image processing tasks including convolution operations and license plate recognition using EasyOCR.

## Description

The following Python scripts are included in this repository:

### convolution.py
This script performs a convolution operation on a given image matrix using a defined kernel. The convolution function slices the original matrix into fragments, multiplies each fragment by the kernel, and computes the output. It is mainly used for image processing tasks such as edge detection, blurring, or sharpening.

### padding.py
This script applies padding to an image before performing the convolution operation. Padding helps maintain the size of the output image when the kernel is applied. It adds zeros around the original image, and then the convolution operation is performed with the kernel.

### plate-recognition.py
This script detects and recognizes license plates from a given image. It processes the image using several filters to reduce noise, detects the license plate's contour, and uses EasyOCR to extract text from the plate. The final result is displayed along with the detected number plate location on the original image.

## Installing Dependencies

First you need to navigate to the according directory, for example ```cd Padding```. Then you should create a virtual environment ``` python3 -m venv venv ``` and activate it ```source venv/bin/activate```.

Before running any of the scripts, you need to install the required dependencies. Each script has a `requirements.txt` file that lists the necessary libraries. To install the dependencies, navigate to the directory where the `requirements.txt` file is located and run the following command:

```bash
pip install -r requirements.txt
```

## Usage

- `convolution.py`: Can be used to apply convolution on any image matrix using a predefined kernel. You can modify the matrix and kernel values to apply different filters.
- `padding.py`: Adds padding to the image before applying convolution, ensuring the output image size remains consistent.
- `plate-recognition.py`: Detects and reads text from vehicle number plates using EasyOCR. Make sure to replace the image path with your own image to detect the license plate in your image.

