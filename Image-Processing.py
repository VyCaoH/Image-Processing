import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to adjust brightness of the image
def process_Brightness(img, brightness):
    brightness = int(brightness)
    #Change picture to numpy array with 32 bytes to enough space to store
    img = np.array(img, dtype=np.int32)
    img += brightness
    #Clip value into (0,255) to remain origin picture
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

# Function to adjust contrast of the image
def process_Contrast(img, contrast_factor):
    contrast_factor = float(contrast_factor)
    img = np.array(img, dtype=np.int32)
    img = img.astype(np.uint16) * contrast_factor
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

# Function to flip the image vertically or horizontally
def flip_Picture(img, flip_type):
    flip_type = int(flip_type)
    img = np.array(img, dtype=np.int32)
    #0 for vertical flip and 1 for horizontal flip
    if flip_type == 0:
        return np.flip(img, axis=0).astype(np.uint8)
    elif flip_type == 1:
        return np.flip(img, axis=1).astype(np.uint8)
    else:
        raise ValueError("Invalid flip type. Use 0 for vertical flip and 1 for horizontal flip.")
    
# Function to convert the image to grayscale
def process_GreyScale(img):
  # RGB to grayscale conversion weights
    weights = [0.299, 0.587, 0.144]
    # Compute the grayscale intensity for each pixel
    gray_intensity = np.dot(img[:, :, :3], weights)
    grayscale_image = np.clip(gray_intensity, 0, 255).astype(np.uint8)
    return grayscale_image

# Function to apply sepia filter to the image
def apply_sepia_filter(img):
    #using sepia matrix to apply all matrix of picture so that change color of it
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    img_array = np.array(img)
    sepia_img_array = np.dot(img_array[:, :, :3], sepia_matrix.T)
    sepia_img_array = np.clip(sepia_img_array, 0, 255).astype(np.uint8)
    return sepia_img_array.astype(np.uint8)

# Function to blur the image using a given kernel
def blur_image(image, kernel):
    kernel_size = kernel.shape[0]
    # Calculate the padding size to ensure proper convolution without boundary issues
    padding = (kernel_size - 1) // 2
    padded_image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    blurred_image = np.zeros_like(image, dtype=np.float32)
    #go through every channel, width, height of picture to use convolution
    for c in range(image.shape[2]):
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                ROI = padded_image[y:y+kernel_size, x:x+kernel_size, c]
                blurred_pixel = np.sum(ROI * kernel)
                blurred_image[y, x, c] = blurred_pixel
    return blurred_image.astype(np.uint8)

# Function to sharpen the image using a specific kernel
def sharpen_image(image):
    image_array = np.array(image)
    #using this kernel to sharpen image
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    kernel_size = kernel.shape[0]
    padding = (kernel_size - 1) // 2
    padded_image = np.pad(image_array, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    sharpened_image = np.zeros_like(image_array, dtype=np.int32)
    for c in range(image_array.shape[2]):
        for y in range(image_array.shape[0]):
            for x in range(image_array.shape[1]):
                ROI = padded_image[y:y+kernel_size, x:x+kernel_size, c]
                sharpened_pixel = np.sum(ROI * kernel)
                sharpened_image[y, x, c] = sharpened_pixel
    sharpened_image = np.clip(sharpened_image, 0, 255)
    return sharpened_image.astype(np.uint8)


# Function to crop the center of the image
def crop_center(img_array, crop_size_x, crop_size_y):
    height, width, channel = img_array.shape
    crop_size_x = int(crop_size_x)
    crop_size_y = int(crop_size_y)
    if(crop_size_x > width or crop_size_y > height):
        raise ValueError("Invalid size")
    # Take height and width of picture
    startx = height//2-(crop_size_x//2)
    starty = width//2-(crop_size_y//2)    
    return img_array[starty:starty+crop_size_y,startx:startx+crop_size_x]

# Function to crop a circular region from the image
def crop_circle(img_array, cx, cy, radius):
    radius = int(radius)
    # Tính toán tọa độ của các điểm ảnh trong ảnh
    y, x = np.ogrid[:img_array.shape[0], :img_array.shape[1]]
    radius = min(radius, min(img_array.shape[0], img_array.shape[1]))
    # Tạo mặt nạ cho các điểm ảnh thuộc hình tròn
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    # Áp dụng mặt nạ để crop hình tròn từ ảnh
    cropped_img_array = np.zeros_like(img_array)
    cropped_img_array[mask] = img_array[mask]
    return cropped_img_array

# Function to draw a mask for an ellipse and crop the region from the image
def draw_rotated_ellipse_mask(center_x, center_y, a, b, angle_degrees):
    y, x = np.ogrid[:img_array.shape[0], :img_array.shape[1]]
    angle_rad = np.deg2rad(angle_degrees)
    x_c = x - center_x
    y_c = y - center_y
    x_new = x_c * np.cos(angle_rad) + y_c * np.sin(angle_rad)
    y_new = + y_c * np.cos(angle_rad) - x_c * np.sin(angle_rad) 
    mask = (x_new**2 / a**2) + (y_new**2 / b**2) <= 1
    return mask.astype(np.uint8)

def draw_two_elips(center_x, center_y):
    rotated_ellipse_img = draw_rotated_ellipse_mask(center_x, center_y, 300, 200, 45)
    rotated_ellipse_img2 = draw_rotated_ellipse_mask(center_x, center_y, 300, 200, -45)
    combined_ellipse_img = np.logical_or(rotated_ellipse_img, rotated_ellipse_img2)
    croppedE_img_array = np.zeros_like(img_array)
    croppedE_img_array[combined_ellipse_img] = img_array[combined_ellipse_img]
    return croppedE_img_array

if __name__ == "__main__":
    file_name = input("Input file name to process: (VD: 1.jpg) ")
    img = Image.open(file_name)
    img_array = np.array(img)
    print("0. All functions will be process")
    print("1. Change brightness ")
    print("2. Change contrast ")
    print("3. Flip picture ")
    print("4. Grayscale/sepia convert ")
    print("5. Blurred/Sharpen image ")
    print("6. Crop picture in center")
    print("7. Crop picture in circular")
    print("8. Crop picture in ellipse")
    func = input("Input number of function you want to process: (Example: 1) " )
    index_dot = file_name.rfind(".")
    file_name_without_extension = file_name[:index_dot]
    file_extension = file_name[index_dot:]

    match func:
        case "0":
            print("0. All functions will be process")
            # Option 0: Process all functions

            brightness = input("Input brightness you want to apply (positive number for brighter and negative for darker): ")
            brightened_img_array = process_Brightness(img_array, brightness)
            brightened_img = Image.fromarray(brightened_img_array)
            brightened_img_output =  f"{file_name_without_extension}_brightness{file_extension}"
            brightened_img.save(brightened_img_output)
            
            contrast_factor = input("Input contrast factor you want to apply (Ex: 1.5):  ")
            contrast_img_array = process_Contrast(img_array,  contrast_factor)
            contrast_img = Image.fromarray(contrast_img_array)
            contrast_img_output =  f"{file_name_without_extension}_contrast{file_extension}"
            contrast_img.save(contrast_img_output)
            
            
            flippedVertical_img_array = flip_Picture(img_array, 0)
            flippedVertical_img = Image.fromarray(flippedVertical_img_array)
            flippedVertical_img_output =  f"{file_name_without_extension}_flippedVertical{file_extension}"
            flippedVertical_img .save(flippedVertical_img_output)
            
            flippedHorizontally_img_array = flip_Picture(img_array, 1)
            flippedHorizontally_img = Image.fromarray(flippedHorizontally_img_array)
            flippedHorizontally_img_output = f"{file_name_without_extension}_flippedHorizontally{file_extension}"
            flippedHorizontally_img .save(flippedHorizontally_img_output)
            
            greyScale_img_array = process_GreyScale(img_array)
            greyScale_img = Image.fromarray(greyScale_img_array)
            greyScale_img_output = f"{file_name_without_extension}_greyScale{file_extension}"
            greyScale_img.save(greyScale_img_output)
            
            Sepia_img = apply_sepia_filter(img)
            Sepia_img = Image.fromarray(Sepia_img)
            Sepia_img_output = f"{file_name_without_extension}_Sepia{file_extension}"
            Sepia_img.save(Sepia_img_output)
            
            
            kernel_size = 5
            kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
            blurred_image = blur_image(img_array, kernel)
            blurred_img = Image.fromarray(blurred_image)
            blurred_image_output = f"{file_name_without_extension}_blurred{file_extension}"
            blurred_img.save(blurred_image_output)
            
            
            sharpen_image = sharpen_image(img_array)
            sharpen_img = Image.fromarray(sharpen_image)
            sharpen_image_output = f"{file_name_without_extension}_sharpen{file_extension}"
            sharpen_img.save(sharpen_image_output)
            
            crop_size_x= input("Input width size of picture crop (Ex: 50) ")
            crop_size_y= input("Input height size of picture crop (Ex: 50) ")
            Cropped_img_array = crop_center(img_array, crop_size_x, crop_size_y)
            Crop_img = Image.fromarray(Cropped_img_array)
            Crop_img_output = f"{file_name_without_extension}_CroppedCenter{file_extension}"
            Crop_img.save(Crop_img_output)

            
            circle_radius = input("Input radius of circle you want to crop (Ex: 50):  ")
            center_x, center_y = img_array.shape[1] // 2, img_array.shape[0] // 2 
            CroppedCir_img_array = crop_circle(img_array, center_x, center_y, circle_radius)
            CropCir_img = Image.fromarray(CroppedCir_img_array)
            CropCir_img_output =  f"{file_name_without_extension}_CropCir{file_extension}"
            CropCir_img.save(CropCir_img_output)
            
            center_x2, center_y2 = img_array.shape[1] // 2, img_array.shape[0] // 2
            croppedE_img_array = draw_two_elips(center_x, center_y)
            croppedE_img = Image.fromarray(croppedE_img_array)
            croppedE_img_output =  f"{file_name_without_extension}_CropEllipse{file_extension}"
            croppedE_img.save(croppedE_img_output)
            
        case "1":
            print("1. Change brightness ")
            brightness = input("Input brightness you want to apply (positive number for brighter and negative for darker): ")
            brightened_img_array = process_Brightness(img_array, brightness)
            brightened_img = Image.fromarray(brightened_img_array)
            
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(brightened_img)
            plt.title("Darkened Image")
            plt.axis("off")

            plt.show()

        case "2":
            print("2. Change contrast ")
            contrast_factor = input("Input contrast factor you want to apply (Ex: 1.5):  ")
            contrast_img_array = process_Contrast(img_array,  contrast_factor)
            contrast_img = Image.fromarray(contrast_img_array)
                        
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(contrast_img)
            plt.title("Contrast changed Image")
            plt.axis("off")

            plt.show()
            
        case "3":
            print("3. Flip picture")
            type = input("Input type of flip you want to process (0. Flip vertically, 1. Flip horizontally): " )
            flipped_img_array = flip_Picture(img_array, type)
            flipped_img = Image.fromarray(flipped_img_array)
                        
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(flipped_img)
            plt.title("Flipped Image")
            plt.axis("off")

            plt.show()

        case "4":
            print("4. Grayscale/sepia convert")
            type = input("Input process you want to do (0. Grayscale, 1.Sepia): ")
            match type:
                case "0":
                    greyScale_img_array = process_GreyScale(img_array)
                    greyScale_img = Image.fromarray(greyScale_img_array)
                    
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(img)
                    plt.title("Original Image")
                    plt.axis("off")

                    plt.subplot(1, 2, 2)
                    plt.imshow(greyScale_img, cmap='gray')
                    plt.title("Grayscale Image")
                    plt.axis("off")

                    plt.show()
                case "1":
                    Sepia_img_array = apply_sepia_filter(img_array)
                    Sepia_img = Image.fromarray(Sepia_img_array)
                                
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(img)
                    plt.title("Original Image")
                    plt.axis("off")

                    plt.subplot(1, 2, 2)
                    plt.imshow(Sepia_img)
                    plt.title("Sepia convert Image")
                    plt.axis("off")
                    
                    plt.show()
                    
        case "5":
            print("5. Blurred/Sharpen image")
            type = input("Input process you want to do (0. Blurred, 1.Sharpen): ")
            match type:
                case "0":
                    # Define the kernel (you can adjust the values or use different kernels)
                    kernel_size = 50
                    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)

                    # Blur the image using the kernel
                    blurred_image = blur_image(img_array, kernel)
                                                
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(img)
                    plt.title("Original Image")
                    plt.axis("off")

                    plt.subplot(1, 2, 2)
                    plt.imshow(blurred_image)
                    plt.title("Blurred Image")
                    plt.axis("off")

                    plt.show()
                    
                case "1":
                    sharpen_image = sharpen_image(img_array)

                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(img)
                    plt.title("Original Image")
                    plt.axis("off")

                    plt.subplot(1, 2, 2)
                    plt.imshow(sharpen_image)
                    plt.title("Sharpen Image")
                    plt.axis("off")

                    plt.show()

        case "6": 
            print("6. Crop picture in center")
            crop_size_x= input("Input width size of picture crop (Ex: 50) ")
            crop_size_y= input("Input height size of picture crop (Ex: 50) ")

            Cropped_img_array = crop_center(img_array, crop_size_x, crop_size_y)
            Crop_img = Image.fromarray(Cropped_img_array)
                        
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(Crop_img)
            plt.title("Cropped Image")
            plt.axis("off")

            plt.show()
        
        case "7": 
            print("7. Crop picture in circular")
            circle_radius = input("Input radius of circle you want to crop (Ex: 50):  ")
            center_x, center_y = img_array.shape[1] // 2, img_array.shape[0] // 2 
            CroppedCir_img_array = crop_circle(img_array, center_x, center_y, circle_radius)
            CropCir_img = Image.fromarray(CroppedCir_img_array)
                        
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(CropCir_img)
            plt.title("Cropped Image")
            plt.axis("off")

            plt.show()
        case "8": 
            print("8. Crop picture in ellipse")
            center_x, center_y = img_array.shape[1] // 2, img_array.shape[0] // 2
            croppedE_img_array = draw_two_elips(center_x, center_y)
            croppedE_img = Image.fromarray(croppedE_img_array)
                        
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(croppedE_img)
            plt.title("Cropped Image")
            plt.axis("off")

            plt.show()
                        
        
       
            