import os
import cv2

# Set the folder path containing the images and the name for the output video
folder_path = "C://FTDT/heatmaps2"  # Adjust this path if needed
video_name = 'heatmaps_animation.mp4'  # Output video filename

# Get the list of image files in the specified folder
image_files = [f for f in os.listdir(folder_path) if f.startswith('heatmap2_') and f.endswith('.png')]

# Sort the image files numerically
image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

# Check if there are images to process
if not image_files:
    print("No images found in the specified folder.")
else:
    # Read the first image to get the frame size
    first_image_path = os.path.join(folder_path, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Initialize video writer with OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(video_name, fourcc, 20, (width, height))

    # Read and save images to video
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)
        out.write(img)

    out.release()  # Release the video writer
    print(f"Video saved as {video_name}")
