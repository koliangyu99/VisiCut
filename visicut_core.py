# visicut_core.py

# Consolidated Imports for the module
from google.colab import files, output
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image as PILImage # Alias PIL.Image to avoid conflict
from ipywidgets import Image as IPYImage # Alias ipywidgets.Image
import numpy as np
from ipycanvas import MultiCanvas
from IPython.display import display
from io import BytesIO
import ipywidgets as widgets
import matplotlib.pyplot as plt

# Global or Class-level variable for the predictor and points (for interaction)
# We'll initialize these in the setup function and modify them through returned values
_predictor = None
_interactive_points = [] # To store points from the interactive canvas

def setup_sam_predictor():
    """
    Sets up the SAM-2 predictor. This function runs the initial setup
    and should be called once.
    Corresponds to: Cell 1 (initial setup part)
    """
    global _predictor # Declare that we are modifying the global _predictor
    print("Initializing SAM-2 predictor (this may take a moment)...")
    output.enable_custom_widget_manager()
    _predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2-hiera-large", mask_threshold=0.0
    )
    print("✅ SAM-2 Predictor initialized.")
    return _predictor # Return it for explicit passing in Colab

def upload_and_prepare_image(predictor_instance):
    """
    Handles image upload and prepares it for SAM-2.
    Corresponds to: Cell 2
    
    Args:
        predictor_instance: An initialized SAM2ImagePredictor instance.

    Returns:
        tuple: (filename, pil_image_obj, image_np)
    """
    print("Please upload an image file.")
    uploaded = files.upload()
    filename = next(iter(uploaded.keys()))
    pil_image_obj = PILImage.open(filename)
    image_np = np.array(pil_image_obj.convert("RGB"))
    predictor_instance.set_image(image_np)
    print(f"✅ Image '{filename}' uploaded and set for prediction.")
    return filename, pil_image_obj, image_np

def setup_interactive_canvas(pil_image_obj):
    """
    Sets up the interactive canvas for point clicking.
    Corresponds to: Cell 3

    Args:
        pil_image_obj (PIL.Image.Image): The PIL Image object to display.

    Returns:
        list: The list of points collected from user clicks.
    """
    global _interactive_points # Declare that we are modifying the global _interactive_points
    _interactive_points = [] # Reset points for a new image

    dot_radius = 8
    dot_color = "lime"
    # overlay_guide_alpha = 0.3 # Not directly used in drawing logic here
    # grid_size = 50 # Not used in drawing logic here

    w, h = pil_image_obj.size
    f = BytesIO()
    pil_image_obj.save(f, format='PNG')
    f.seek(0)

    image_widget = IPYImage(
        value=f.read(),
        format='png',
        width=w,
        height=h,
    )

    canvases = MultiCanvas(2, width=w, height=h)
    display(canvases) # Display the canvases first

    base, overlay = canvases[0], canvases[1]
    base.draw_image(image_widget, 0, 0)
    overlay.global_alpha = 0.5
    overlay.fill_style = dot_color

    def on_mouse_down(x, y):
        _interactive_points.append((x, y))
        overlay.fill_circle(x, y, dot_radius)
        # Optional: print points in real-time or update a widget with point count
        # print(f"Collected points: {_interactive_points}") # Can be too verbose
        # Consider using a widget for better feedback

    overlay.on_mouse_down(on_mouse_down)
    print(f"Click directly on object/image; The Dot Color is {dot_color.capitalize()}.")
    print(f"Canvas dimensions (W, H): {w}, {h}")
    print(f"Type of object drawn: {type(pil_image_obj)} (converted to {type(image_widget)})")
    
    # We don't return points directly here because they are collected interactively
    # The Colab notebook will access _interactive_points after this cell is run.
    print("Please click on the image to select points, then run the next cell.")


def run_sam_and_visualize(predictor_instance, image_np):
    """
    Runs SAM-2 prediction based on collected points and visualizes results.
    Corresponds to: Cell 4 (prediction and visualization part)

    Args:
        predictor_instance: An initialized SAM2ImagePredictor instance.
        image_np (np.array): The original image as a NumPy array.

    Returns:
        tuple: (masks, result_pil_image)
    """
    if not _interactive_points:
        print("No points collected. Please click on the image in the previous cell.")
        return None, None

    coords = np.array(_interactive_points)
    labels = np.ones(len(coords), dtype=int) # All points are positive prompts

    print(f"Predicting mask with {len(coords)} points...")
    masks, scores, _ = predictor_instance.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=False
    )

    # Visualize results
    plt.figure(figsize=(8,8))
    plt.imshow(image_np)
    plt.imshow(masks[0], cmap="jet", alpha=0.5)
    for (x, y), l in zip(coords, labels):
        plt.scatter(x, y, c="lime", s=50, edgecolors="black")
    plt.axis("off")
    plt.show()

    # Create transparent-background PNG
    alpha = (masks[0] > 0.5).astype(np.uint8) * 255

    if image_np.ndim == 3 and image_np.shape[2] == 3:
        rgba = np.dstack([image_np, alpha])
    elif image_np.ndim == 3 and image_np.shape[2] == 4:
        rgba = image_np.copy()
        rgba[:,:,3] = alpha
    else:
        print("Warning: image_np has unexpected shape for compositing RGBA. Creating a new RGBA image.")
        # Create a 3-channel base (e.g., black) to stack with alpha
        rgba = np.dstack([np.zeros_like(image_np[:,:,0:3]), alpha])

    result_pil_image = PILImage.fromarray(rgba, mode="RGBA")
    display(result_pil_image)
    print("✅ Mask generated and cutout displayed.")
    return masks, result_pil_image

def save_cutout(filename_original, result_pil_image):
    """
    Saves the generated transparent cutout image.
    Corresponds to: Cell 5

    Args:
        filename_original (str): The original filename used to derive the output name.
        result_pil_image (PIL.Image.Image): The PIL Image object of the cutout.
    """
    if result_pil_image is None:
        print("No image to save. Run previous steps first.")
        return

    out_name = filename_original.rsplit(".",1)[0] + "_sam2.png"
    result_pil_image.save(out_name)
    print(f"✅ Saved cutout as {out_name}")

# --- Helper to access points for the Colab notebook ---
def get_interactive_points():
    """Returns the globally stored interactive points."""
    return _interactive_points
