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
# These are used to maintain state across interactive Colab cells.
_predictor = None
_interactive_points = [] # To store points collected from the interactive canvas

def setup_sam_predictor():
    """
    Sets up the SAM-2 predictor. This function performs the initial setup
    and should be called once per Colab session.
    Corresponds to: Cell 1 (initial setup part)
    """
    global _predictor # Declare that we are modifying the global _predictor
    print("Initializing SAM-2 predictor (this may take a moment)...")
    output.enable_custom_widget_manager() # Enable custom widget manager for interactive elements
    _predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2-hiera-large", mask_threshold=0.0
    )
    print("✅ SAM-2 Predictor initialized.")
    return _predictor # Return the predictor instance for explicit passing

def upload_and_prepare_image(predictor_instance):
    """
    Handles image upload from the user and prepares it for SAM-2 processing.
    Corresponds to: Cell 2
    
    Args:
        predictor_instance: An initialized SAM2ImagePredictor instance.

    Returns:
        tuple: (filename, pil_image_obj, image_np)
               filename (str): The name of the uploaded file.
               pil_image_obj (PIL.Image.Image): The PIL Image object of the uploaded image.
               image_np (np.array): The NumPy array representation of the image (RGB).
    """
    print("Please upload an image file (e.g., JPG, PNG, GIF, BMP, TIFF).")
    uploaded = files.upload()
    filename = next(iter(uploaded.keys())) # Get the name of the first uploaded file
    pil_image_obj = PILImage.open(filename) # Open the image using PIL
    image_np = np.array(pil_image_obj.convert("RGB")) # Convert to NumPy array for SAM-2, ensuring RGB

    # Set the image in the SAM-2 predictor for subsequent mask prediction
    predictor_instance.set_image(image_np)
    print(f"✅ Image '{filename}' uploaded and set for prediction.")
    return filename, pil_image_obj, image_np

def setup_interactive_canvas(pil_image_obj):
    """
    Sets up the interactive canvas in Colab for users to click on objects
    to define segmentation points.
    Corresponds to: Cell 3

    Args:
        pil_image_obj (PIL.Image.Image): The PIL Image object to display on the canvas.
    """
    global _interactive_points # Declare that we are modifying the global _interactive_points
    _interactive_points = [] # Reset points list for a new image

    dot_radius = 8
    dot_color = "lime" # Green color for the dots

    w, h = pil_image_obj.size # Get original image dimensions
    f = BytesIO()
    pil_image_obj.save(f, format='PNG') # Save PIL image to BytesIO for ipywidgets
    f.seek(0) # Rewind the BytesIO object to the beginning

    image_widget = IPYImage( # Create an ipywidgets Image widget for display
        value=f.read(),
        format='png',
        width=w,
        height=h,
    )

    canvases = MultiCanvas(2, width=w, height=h) # Create a MultiCanvas with 2 layers (base and overlay)
    display(canvases) # IMPORTANT: Display the canvases to render them in the notebook

    base, overlay = canvases[0], canvases[1] # Access the base and overlay layers
    base.draw_image(image_widget, 0, 0) # Draw the static image onto the base layer
    overlay.global_alpha = 0.5 # Set transparency for the overlay (for drawing dots)
    overlay.fill_style = dot_color # Set fill color for the dots

    def on_mouse_down(x, y):
        """Callback function executed when a mouse click occurs on the overlay canvas."""
        _interactive_points.append((x, y)) # Add the click coordinates to the global points list
        overlay.fill_circle(x, y, dot_radius) # Draw a circle at the clicked point

    overlay.on_mouse_down(on_mouse_down) # Attach the callback to mouse down events on the overlay

    print(f"Click directly on object/image; The Dot Color is {dot_color.capitalize()}.")
    print(f"Canvas dimensions (W, H): {w}, {h}")
    print(f"Type of object drawn: {type(pil_image_obj)} (converted to {type(image_widget)})")
    print("\nInstructions: Click on the object(s) in the image above to define your segmentation points. Then, run the next cell.")


def run_sam_and_visualize(predictor_instance, image_np):
    """
    Runs SAM-2 prediction based on collected points, visualizes results (original,
    masked, and flattened cutout), and creates the final flattened cutout image.
    Corresponds to: Cell 4

    Args:
        predictor_instance: An initialized SAM2ImagePredictor instance.
        image_np (np.array): The original image as a NumPy array (RGB).

    Returns:
        tuple: (masks, result_pil_image_flattened)
               masks (np.array): The raw segmentation masks from SAM-2.
               result_pil_image_flattened (PIL.Image.Image): The final PIL Image object
                                                                with the cutout flattened onto a white background.
    """
    if not _interactive_points:
        print("❌ No points collected. Please click on the image in the previous cell (Cell 3) before running this cell.")
        return None, None

    coords = np.array(_interactive_points) # Convert collected points to a NumPy array
    labels = np.ones(len(coords), dtype=int) # All points are positive prompts (foreground)

    print(f"Predicting mask with {len(coords)} points...")
    masks, scores, _ = predictor_instance.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=False # Request a single, most confident mask
    )

    # --- Create the transparent-background RGBA image ---
    alpha = (masks[0] > 0.5).astype(np.uint8) * 255 # Create alpha channel from the mask

    # Handle different image channel counts for stacking RGBA
    if image_np.ndim == 3 and image_np.shape[2] == 3: # Original is RGB
        pil_rgba_image = PILImage.fromarray(np.dstack([image_np, alpha]), mode="RGBA")
    elif image_np.ndim == 3 and image_np.shape[2] == 4: # Original is already RGBA
        rgba_copy = image_np.copy()
        rgba_copy[:,:,3] = alpha # Just replace its alpha channel
        pil_rgba_image = PILImage.fromarray(rgba_copy, mode="RGBA")
    else:
        print("Warning: image_np has unexpected shape for compositing RGBA. Creating a new RGBA image with black base.")
        # Create a 3-channel black base to stack with the alpha channel
        pil_rgba_image = PILImage.fromarray(np.dstack([np.zeros_like(image_np[:,:,0:3]), alpha]), mode="RGBA")

    # Flatten the transparent image onto a white background
    background_color = (255, 255, 255) # White background (RGB tuple)
    background = PILImage.new('RGB', pil_rgba_image.size, background_color) # Create a new white background image
    # Composite the RGBA image onto the RGB background. Convert background to RGBA temporarily for alpha_composite.
    # Then convert the result back to RGB for a flattened image.
    result_pil_image_flattened = PILImage.alpha_composite(background.convert('RGBA'), pil_rgba_image).convert('RGB')
    # --- End create flattened image ---

    # --- Visualize results with three labeled subplots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # Create 1 row, 3 columns of plots for visualization
    
    # Plot 1: Original Image with User Points
    axes[0].imshow(image_np)
    for (x, y) in coords: # Iterate directly over coords
        axes[0].scatter(x, y, c="lime", s=50, edgecolors="black") # Show clicked points
    axes[0].set_title("1. Original Image + Points")
    axes[0].axis("off") # Turn off axis labels and ticks

    # Plot 2: Mask Overlay
    axes[1].imshow(image_np)
    axes[1].imshow(masks[0], cmap="jet", alpha=0.5) # Overlay mask with transparency
    for (x, y) in coords: # Iterate directly over coords
        axes[1].scatter(x, y, c="lime", s=50, edgecolors="black") # Show points on mask too
    axes[1].set_title("2. Mask Overlay")
    axes[1].axis("off")

    # Plot 3: Final Cutout (Flattened on White Background)
    axes[2].imshow(result_pil_image_flattened) # Display the generated flattened cutout
    axes[2].set_title("3. Final Cutout (Flattened)")
    axes[2].axis("off")

    plt.tight_layout() # Adjust subplot parameters for a tight layout
    plt.show() # Display the Matplotlib figure

    # Display the single flattened cutout image separately below the plots for easy saving/copying
    display(result_pil_image_flattened)
    print("✅ Mask generated and flattened cutout displayed.")
    return masks, result_pil_image_flattened # Return the masks and the flattened PIL Image

def save_cutout(filename_original, result_pil_image_to_save):
    """
    Saves the generated cutout image (flattened onto white background)
    and initiates a direct download to the user's local machine.
    Corresponds to: Cell 5

    Args:
        filename_original (str): The original filename used to derive the output name.
        result_pil_image_to_save (PIL.Image.Image): The PIL Image object of the cutout
                                                       (expected to be flattened).
    """
    if result_pil_image_to_save is None:
        print("❌ No image to save. Please ensure Cell 4 ran successfully and produced a result.")
        return

    # MODIFIED: Construct the output filename with "sam2_" as a prefix
    # e.g., "chair.jpg" becomes "sam2_chair.png"
    out_name = "sam2_" + filename_original.rsplit(".", 1)[0] + ".png"

    # Save the image to the Colab environment's file system
    result_pil_image_to_save.save(out_name)
    print(f"✅ Saved cutout to Colab files as {out_name}")

    # Initiates a direct download to the user's local computer
    files.download(out_name)
    print(f"⬇️ Initiating download for {out_name}...")

# --- Helper to access points for the Colab notebook (for debugging/inspection) ---
def get_interactive_points():
    """Returns the globally stored interactive points list."""
    return _interactive_points
