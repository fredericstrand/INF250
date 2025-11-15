import os
from PIL import Image

def slice_images_into_four_parts(input_dir):
    # Create the output folder inside the input directory
    output_dir = os.path.join(input_dir, "sliced_pictures")
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all jpeg files
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpeg', '.jpg')):
            filepath = os.path.join(input_dir, filename)

            # Open the image
            with Image.open(filepath) as img:
                width, height = img.size

                # Coordinates for 4 quadrants
                mid_x, mid_y = width // 2, height // 2
                boxes = [
                    (0, 0, mid_x, mid_y),                # top-left
                    (mid_x, 0, width, mid_y),            # top-right
                    (0, mid_y, mid_x, height),            # bottom-left
                    (mid_x, mid_y, width, height)         # bottom-right
                ]

                # Slice and save each part
                base_name, ext = os.path.splitext(filename)
                for i, box in enumerate(boxes, 1):
                    part = img.crop(box)
                    new_filename = f"{base_name}_part{i}{ext.lower()}"
                    part.save(os.path.join(output_dir, new_filename), quality=95)

                print(f"Sliced {filename} into 4 parts.")

    print(f"\nAll images sliced! Saved in: {output_dir}")

if __name__ == "__main__":
    # Use current directory
    current_dir = os.path.join(os.getcwd(), "images")
    slice_images_into_four_parts(current_dir)
