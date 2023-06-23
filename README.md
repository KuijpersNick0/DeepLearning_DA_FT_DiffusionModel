## Prerequisites

Before running the project, ensure that you have the following dependencies installed:

- Python (version 3.7 or higher)
- Jupyter Notebook
- TensorFlow (version 2.0 or higher)
- Keras (version 2.4 or higher)
- NumPy
- SciPy
- Matplotlib

## Getting Started

1. Clone the project repository to your local machine.
2. Navigate to the project directory.

## Fine-Tuning the Diffusion Model

To create a new fine-tuned model and save its weights for future use, follow these steps:

1. Open `DiffusionFineTuning.ipynb` in Jupyter Notebook.
2. Run the notebook to perform the fine-tuning process.
3. Save the generated model weights for later use.

## Discrete cosine transformation (DCT)

The Discrete Cosine Transformation is a mathematical transformation that converts a signal or an image from the spatial domain to the frequency domain. Traditionally used in image compression, DCT has found application in data augmentation due to its ability to preserve essential information while modifying the input data. By applying DCT, the original data is transformed into a series of cosine functions with different frequencies. These transformed features can then be altered to generate augmented data points.\

## DCT 1 and DCT 2

Inside the `DCT_1_&DCT_2.ipynb` you will find 2 different DCT metods

1. DCT 1 (method1_DCT): In this method the DCT is used to create new images, but in this case the pixels setted to zero are determined by the following formula: pixelvalue > (mode + std x 2) or < (mode - std x 2). Std is the standard deviation
2. DCT 2 (method2_DCT): some of the features at a random value extracted from a Gaussian distribution are reset. After that, the inverse of the transform is performed


## Generating Artificial Images

### Diffusuion model 
To generate artificial images using the fine-tuned diffusion model and save them to a `.mat` file, perform the following:

1. Open `generateArtificialImages.ipynb` in Jupyter Notebook.
2. Modify the path of the model weights to match the path of the previously generated model weights.
3. Run the notebook to generate the artificial images.
4. Save the generated images to a `.mat` file.

### Discrete cosine transformation (DCT)
To generate new synthetic images using DCT and create new dataset (`.mat` files) that contain some of the new generated images: 

1. Open`DCT_1_&DCT_2.ipynb`for the methods
2. Modify the paths and run the notebook to generate 2 new `.mat` files with only synthetic images, for DCT 1 and DCT 2.
3. Modify the paths and run the notebook `creation_augmented_dataset_DCT_1_DCT_2.ipynb` to create the augmented dataset.
4. Run notebooks `Main_CNN_DCT_method_1.ipynb` and `Main_CNN_DCT_method_2.ipynb` to run the CNN to the new generated datasets.
5. .pt files will be automatically saved. 

## Running the Simple CNN

To execute a simple CNN without data augmentation techniques, use the following steps:

1. Open `mainCNN.py` in a Python IDE or editor.
2. Run the script to execute the simple CNN.

## Running the CNN with Artificial Images

To run a CNN with the newly generated artificial images, follow these steps:

1. Open `ArtificialCNN.py` in a Python IDE or editor.
2. Modify the path to the generated `.mat` file obtained from `generateArtificialImages.ipynb`.
3. Run the script to execute the CNN with artificial images.

## Notes

- Ensure that the necessary dataset files are available and properly formatted before running the scripts.
- Make sure the dependencies are installed correctly and up to date.
- Adjust any relevant paths or configurations within the scripts to match your specific environment and requirements.

## Conclusion

By utilizing the provided scripts and following the instructions outlined in this readme, you can explore the potential of fine-tuning the diffusion model and generating artificial images for improving lymphoma classification accuracy. Feel free to experiment with different parameters, architectures, or fine-tuning techniques to further enhance the performance of the models.

If you have any questions or encounter any issues while running the project, please refer to the project documentation or reach out to the project contributors for assistance.
