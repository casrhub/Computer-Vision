import numpy as np

# Helper function to apply the convolution operation between the fragment and the kernel
def convolution_helper(fragment, kernel):
    result = 0.0
    fragment_row, fragment_col = fragment.shape
    for row in range(fragment_row):
        for col in range(fragment_col):
            result += fragment[row, col] * kernel[row, col]
    return result

# Function to apply padding to the matrix
def add_padding(matrix, padding_size):
    matrix_row, matrix_col = matrix.shape
    
    # Create a padded matrix filled with zeros
    padded_matrix = np.zeros((matrix_row + 2 * padding_size, matrix_col + 2 * padding_size))
    
    # Place the original matrix in the center of the padded matrix
    padded_matrix[padding_size: padding_size + matrix_row, padding_size: padding_size + matrix_col] = matrix
    
    return padded_matrix

# Convolution function with padding
def convolution_with_padding(image, kernel, padding_size):
    # Add padding to the image
    padded_image = add_padding(image, padding_size)
    
    image_row, image_col = padded_image.shape
    kernel_row, kernel_col = kernel.shape
    
    # Output sizes (adjusted for padding)
    output_row = image_row - kernel_row + 1
    output_col = image_col - kernel_col + 1
    
    # Create an empty output matrix
    output = np.zeros((output_row, output_col))
    
    # Perform convolution by sliding the kernel over the padded image
    for row in range(output_row):
        for col in range(output_col):
            # Extract the fragment from the padded image
            fragment = padded_image[row: row + kernel_row, col: col + kernel_col]
            
            # Compute the convolution for this fragment
            output[row, col] = convolution_helper(fragment, kernel)
    
    return output

# Example matrices
image = np.array([[10, 4, 50, 30, 20],
                  [80, 0, 0, 0, 6],
                  [0, 0, 1, 16, 17],
                  [0, 1, 0, 7, 23],
                  [1, 0, 6, 0, 4]])

kernel = np.array([[1, 0, 1],
                   [0, 0, 0],
                   [1, 0, 3]])

# Apply convolution with padding (padding size = 1)
padding_size = 1
output_matrix = convolution_with_padding(image, kernel, padding_size)

# Print the output matrix
print("Padded Convolution Output:")

print(add_padding(image, 3))
print(output_matrix)
