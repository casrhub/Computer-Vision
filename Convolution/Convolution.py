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

# Main convolution function 
def convolution(image, kernel): 
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    # Output sizes
    output_row = image_row - kernel_row + 1
    output_col = image_col - kernel_col + 1

    output = np.zeros((output_row,output_col))

    for row in range (output_row):
        for column in range (output_col):
            # We slice the original matix to extract the fragment of the kernel size
            fragment = image[row: row + kernel_row, column:column + kernel_col]
            output [row, column] = convolution_helper(fragment, kernel)

    plt.imshow(output, cmap='gray')  # Display the result
    plt.title("Output Image using {}x{} Kernel".format(kernel_row, kernel_col))
    plt.show()
    
    return output


# Define the Original Matrix (Exercise matrix)
original_matrix = np.array([[10, 4, 50, 30, 20],
                            [80, 0, 0, 0, 6],
                            [0, 0, 1, 16, 17],
                            [0, 1, 0, 7, 23],
                            [1, 0, 6, 0, 4]])

# Define the Filter
filter_matrix = np.array([[1, 0, 1],
                          [0, 0, 0],
                          [1, 0, 3]])

output_matrix = convolution(original_matrix, filter_matrix)
print("Output Matrix:")
print(output_matrix)

#Steps on what the code does: 
# 1.  Create a helper function to compute the multipication between the kernel and the fragment
# 2.  Create a convolution function that slices the original matrix to match the size of the kernel which will produce a fragment
# 3.  Multiply a each fragment of the original matrix whith the kernel 
