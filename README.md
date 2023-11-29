# ```                AI & ML Project                ```

# Image_Segmentation_with_SAM_and_OWL-ViT_Model
This project contains a Python script for image processing using Meta AI's SAM (Sample Augmentation Module) for segmentation and OWL-ViT (Object and Word Learning Vision Transformer) for object detection. The script takes an input image, performs segmentation using SAM, and then detects objects using OWL-ViT.

## Dependencies

Make sure to install the required dependencies before running the script:

```bash
pip install sam owl-vit opencv-python numpy matplotlib
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/NoorMahammad-S/Image_Segmentation_with_SAM_and_OWL-ViT_Model.git
cd Image_Segmentation_with_SAM_and_OWL-ViT_Model
```

2. Install dependencies:

```bash
pip install sam owl-vit opencv-python numpy matplotlib
```

3. Run the script:

```bash
python main.py
```

## Configuration

- Adjust the `image_path` variable in the script to point to the image you want to process.

- Customize the target size, confidence threshold, and other parameters in the script according to your specific requirements.

## Acknowledgments

- Meta AI for providing SAM and OWL-ViT models.
