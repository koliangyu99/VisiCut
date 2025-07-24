# VisiCut

## Interactive Image Segmentation and Cutout with SAM-2

VisiCut is a Google Colab-friendly project that leverages the powerful Segment Anything Model 2 (SAM-2) for interactive image segmentation and transparent background cutouts. Simply upload an image, click points on the objects you want to segment, and VisiCut will generate a precise mask and a ready-to-use transparent PNG.

This repository holds the core Python utility functions, designed for clean integration into a Google Colab notebook workflow.

## Features

* **Point-Based Interaction:** Define segmentation regions by simply clicking points on your target object.
* **SAM-2 Powered:** Utilizes the state-of-the-art Segment Anything Model 2 for highly accurate masking.
* **Transparent Cutouts:** Generates clean PNG images with the segmented object and a transparent background.
* **Google Colab Ready:** Optimized for easy setup and execution within Google Colab environments.
* **Modular Code:** Core logic is separated into Python modules for reusability and clarity.

## How to Use in Google Colab

To use VisiCut, you'll run a Google Colab notebook that clones this repository and executes the interactive segmentation.

1.  **Open the Colab Notebook:**
    * [Link to your VisiCut Colab Notebook here]
        *(You will create this notebook later, after setting up your GitHub repo and putting the .py files in it. Once created, paste its public share link here.)*

2.  **Run Cell 1 (Setup and Clone Repository):**
    * This cell will install necessary libraries (SAM-2, ipycanvas, etc.) and clone this `VisiCut` GitHub repository into your Colab environment.
    * It will also add the repository path to Python's `sys.path` so you can import functions from `my_image_utils.py`.

3.  **Run Cell 2 (Upload Image):**
    * Execute this cell to upload the image you wish to segment.

4.  **Run Cell 3 (Interactive Pointing):**
    * An interactive canvas will appear. **Click directly on the object** you want to segment. You will see a small lime green dot appear for each click.
    * The canvas is clickable even if the image doesn't immediately load (due to widget timing). Ensure you see the image *before* clicking for accurate point placement. If the image doesn't appear, try running the cell again or re-running Cell 1 and Cell 3.

5.  **Run Cell 4 (Generate Cutout):**
    * This cell will send your clicked points to SAM-2, generate the mask, visualize the result, and finally produce the transparent PNG cutout.
    * The cutout image will be displayed, and a `.png` file will be saved to your Colab environment (visible in the file browser on the left sidebar).

---

## Project Structure

* `README.md`: This file.
* `my_image_utils.py`: Contains core Python functions for image processing, SAM-2 interaction helpers, and result visualization.
    * `process_image_for_sam`: Prepares images for SAM-2.
    * `display_mask_on_image`: Visualizes the segmentation mask.
    * `create_rgba_cutout`: Generates the final transparent cutout.
    * *(Add any other `.py` files and their main functions here)*

## Installation (for local development, not required for Colab)

If you wish to run parts of this project locally (outside of Colab), you would typically:

```bash
