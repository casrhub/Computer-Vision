import numpy as np
import cv2
import matplotlib.pyplot as plt


def convolution_helper(fragment, kernel):
    fragment_row, fragment_col = fragment.shape #Tuple of fragment to store size of n * m matrix
    kernel_row, kernel_col = kernel.shape #Tuple of kernel to store size of n * m matrix
    # Matrix multiplication 
    multiplication_result = 0.0
    for row in range(fragment_row):
        for column in range(fragment_col):
            multiplication_result += fragment[row,column] * kernel[row, column]
    return multiplication_result
