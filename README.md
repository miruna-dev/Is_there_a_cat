<h1 align="center">Cat Detector - From Image to Prediction</h1>

<p align="center">
  <i>An end-to-end MATLAB/Octave project for detecting cats in images using RGB processing and machine learning.</i>
</p>


---

## Project Overview

> **Note:**  
> After completing the full implementation, we found that **logistic regression is not powerful enough** to reliably classify cat images.  
> The **best training accuracy** achieved with this method was approximately **56%**, which is not sufficient for real-world image classification tasks.  
> For higher performance, more advanced models (e.g., CNNs) would be required.

---

This project implements a full image classification system that detects whether a **cat** is present in an image — using basic image processing and **logistic regression**, without relying on deep learning.

The pipeline includes:

- **Reading and resizing images** to a fixed 64x64 dimension
- **Splitting RGB channels** into red, green, and blue matrices
- **Flattening and stacking channels** to convert each image into a feature vector
- **Building a data matrix `X`**, where each row is an image
- **Labeling and saving the dataset** into a CSV file
- **Normalizing features** to a [0, 1] range
- **Training a logistic regression model** using gradient descent
- **Making predictions** on the training set
- **Evaluating performance** using accuracy and cost

---

While the system is simple and educational, the results show that **logistic regression struggles with complex image data**. Still, this project provides a solid end-to-end pipeline for understanding how image classification works from raw pixels to model evaluation.


> **Note for Octave users:**  
> You must run `pkg load image` before using image-related functions.


---

## Functions

### `image_to_rgb_matrix`

**Purpose:**  
This function reads an image from a given path, resizes it to **64x64 pixels**, and separates it into its three individual color channels: **Red**, **Green**, and **Blue**.

---

#### Step-by-step behavior:

- The function starts by reading the image using `imread`.
- It resizes the image to `64x64` pixels using `imresize`.

**Why 64x64?**
- It simplifies computations by significantly reducing the number of pixels.
- It produces a smaller and easier-to-handle data matrix.
- It maintains sufficient visual information for the kind of lightweight ML model intended for this project.
---

#### RGB Channel Extraction:

After resizing, the function separates the image into its 3 color channels:
- `R` — contains the red intensity values
- `G` — contains the green intensity values
- `B` — contains the blue intensity values

Each of these is a **64x64 matrix** that represents the saturation of that color across the image.

---

#### Visualizing the Output in Octave:

To test this function and visualize its result in Octave:

```matlab
pkg load image
[R, G, B] = image_to_rgb_matrix('path/to/your/image.jpg');
imshow(R)  % Visualizes the Red channel
```
---

### <code>rgb_to_vector</code>

<p><strong>Purpose:</strong><br>
This function takes three <code>64x64</code> matrices representing the <strong>Red</strong>, <strong>Green</strong>, and <strong>Blue</strong> channels of an image, and transforms them into a single column vector.
</p>

---

#### <ins>Step-by-step behavior:</ins>

- Each RGB matrix is <code>flattened</code> using the <code>(:)</code> operator.
- The flattened vectors are <strong>concatenated vertically</strong> to form a single vector of size <code>12288x1</code>.

---

#### <ins>Why vectorize?</ins>

<p>
Machine learning models operate on numerical vectors. This transformation allows the color information of an image to be represented in a single structure, which simplifies training and prediction.
</p>

- Each image is converted into a feature vector.
- These vectors can be stacked horizontally into a matrix <code>X = [x1, x2, x3, ...]</code> for training.
---

### <code>build_X</code>

<p><strong>Purpose:</strong><br>
This function scans a folder of images and constructs the data matrix <code>X</code>, where each column represents the vectorized RGB content of one image.
</p>

---

#### <ins>Step-by-step behavior:</ins>

- The function searches for all <code>.jpg</code> files in the specified <code>folder_path</code>.
- It reads the **first image** to determine the size of the feature vector (<code>n = 64 * 64 * 3 = 12288</code>).
- It initializes the matrix <code>X</code> with size <code>n x m</code>, where:
  - <code>n</code> = number of features (pixels)
  - <code>m</code> = number of images
- For each image:
  - It is resized and split into RGB channels using <code>image_to_rgb_matrix</code>.
  - Each channel is flattened and stacked into a single vector with <code>rgb_to_vector</code>.
  - The resulting vector is stored as a **column** in <code>X</code>.

---

#### <ins>What does X contain?</ins>

- <code>X</code> is a <code>12288 x m</code> matrix, where:
  - Each **column** is an image.
  - Each **row** corresponds to a pixel across the dataset.
- This structure is ideal for training a machine learning model.

---


### <code>export_images_to_csv</code>

<p><strong>Purpose:</strong><br>
This function processes all images in a folder, converts them to vectors, assigns a label, and saves the data into a CSV file — ready for training or analysis.
</p>

> **Note from experience:**  
> During experimentation, it became clear that working with **`.mat` files** is significantly more efficient than CSVs — especially when handling large image datasets.  
> Saving and loading data in `.mat` format is **faster** and avoids issues with numerical precision and memory consumption that can occur when using CSV.  
> Although this function uses CSV for simplicity and portability, switching to `.mat` is recommended for better performance in practical applications.

---

#### <ins>Step-by-step behavior:</ins>

- Calls <code>build_X</code> to get the matrix <code>X</code> of image vectors.
- Transposes <code>X</code> so that each image becomes a <strong>row</strong> in the final CSV.
- Creates a label vector <code>y</code> with the same number of rows as images:
  - If <code>label = 1</code> → images are labeled as cats
  - If <code>label = 0</code> → images are labeled as non-cats
- Combines <code>X</code> and <code>y</code> into a single matrix <code>data</code>.

---

#### <ins>Appending to existing data:</ins>

- If the output CSV file already exists:
  - It reads the existing data.
  - Appends the new <code>data</code> to it.
- Then it writes the final dataset to the specified <code>output_file</code> using <code>csvwrite</code>.

---

#### <ins>CSV structure:</ins>

Each row in the CSV file contains:
- **12288 values** (flattened RGB image data)
- **1 label** (0 or 1)

So the total number of columns = **12289**.

---

#### <ins>Example usage in Octave:</ins>

```matlab
pkg load image
export_images_to_csv('dataset/cats', 'data.csv', 1);
export_images_to_csv('dataset/not_cats', 'data.csv', 0);
```
---

### <code>get_characteristics</code>

<p><strong>Purpose:</strong><br>
This function reads the CSV dataset and returns two outputs: the feature matrix <code>X</code> and the label vector <code>y</code>, with features normalized between 0 and 1.
</p>

---

#### <ins>Step-by-step behavior:</ins>

- Loads the dataset from the CSV file using <code>dlmread</code>.
- Separates the data:
  - <code>X</code> contains all columns except the last (image features).
  - <code>y</code> contains the last column (labels).
- Applies **normalization** to <code>X</code>:
  - Each feature is scaled to the [0, 1] range using:
    <br><code>X = (X - min(X)) ./ (max(X) - min(X))</code>

---

#### <ins>Why normalize?</ins>

<p>
Feature normalization ensures that all pixel values are on the same scale, which improves convergence and performance of many machine learning algorithms.
</p>

---

### <code>compute_cost</code>

<p><strong>Purpose:</strong><br>
This function calculates the logistic regression cost <code>J</code>, given the predicted probabilities <code>a</code> and the true labels <code>y</code>.
</p>

---

#### <ins>Step-by-step behavior:</ins>

- Computes the number of training examples: <code>m = length(y)</code>
- Applies the **log-loss** (cross-entropy) formula for binary classification:
  
  <p align="center">
    <code>J = (-1/m) * sum(y .* log(a) + (1 - y) .* log(1 - a))</code>
  </p>

---

#### <ins>What does this cost represent?</ins>

- Measures how far the predicted probabilities <code>a</code> are from the actual labels <code>y</code>.
- A **lower cost** means the model is predicting more accurately.
- It is the objective function that the training algorithm will minimize.

---

### <code>sigmoid</code>

<p><strong>Purpose:</strong><br>
This function applies the sigmoid activation function to its input <code>z</code>. It maps any real number to a value between <code>0</code> and <code>1</code>, which is useful for binary classification tasks like detecting cats.
</p>

---

#### <ins>Mathematical definition:</ins>

<p align="center">
  <code>sigmoid(z) = 1 / (1 + exp(-z))</code>
</p>

---

#### <ins>Usage in machine learning:</ins>

- Converts raw model outputs into **probabilities**
- In this project, it transforms the linear output of <code>X * w + b</code> into a probability between 0 and 1
- Used during both **training** and **prediction**

---

### <code>predict</code>

<p><strong>Purpose:</strong><br>
This function makes binary predictions based on learned weights <code>w</code> and bias <code>b</code> using logistic regression. It outputs <code>1</code> for images predicted to contain a cat and <code>0</code> otherwise.
</p>

---

#### <ins>Step-by-step behavior:</ins>

- Computes the linear combination: <code>z = X * w + b</code>
- Applies the sigmoid activation: <code>a = sigmoid(z)</code>
- Applies a threshold at <code>0.5</code>:
  - If <code>a ≥ 0.5</code> → predict <code>1</code> (cat)
  - If <code>a &lt; 0.5</code> → predict <code>0</code> (not a cat)

The output <code>y_pred</code> is a binary vector with the same number of rows as <code>X</code>.

---

### <code>gradient_descent</code>

<p><strong>Purpose:</strong><br>
This function performs gradient descent to learn the optimal weights <code>w</code> and bias <code>b</code> for logistic regression based on a dataset <code>X</code> and labels <code>y</code>.
</p>

---

#### <ins>Step-by-step behavior:</ins>

- Initializes:
  - <code>w</code>: weights (as a column vector)
  - <code>b</code>: bias (scalar)
  - <code>alpha</code>: learning rate
  - <code>num_iters</code>: number of iterations
- Repeats for each iteration:
  1. Computes the linear combination: <code>z = X * w + b</code>
  2. Applies the sigmoid function: <code>a = sigmoid(z)</code>
  3. Calculates the error: <code>dz = a - y</code>
  4. Computes the gradients:
     - <code>dw = (1/m) * X' * dz</code>
     - <code>db = (1/m) * sum(dz)</code>
  5. Updates weights and bias:
     - <code>w = w - alpha * dw</code>
     - <code>b = b - alpha * db</code>

---

#### <ins>Why use gradient descent?</ins>

<p>
Gradient descent minimizes the logistic regression cost function by adjusting the parameters step-by-step in the direction of steepest descent. It is an essential optimization technique for training models on large datasets.
</p>

- Iteratively improves predictions.
- Can handle thousands of features efficiently (like our 12288-pixel image vectors).

---

### <code>train_model</code>

<p><strong>Purpose:</strong><br>
This function trains a logistic regression model using gradient descent and returns the learned parameters <code>w</code> and <code>b</code>, along with the final cost <code>J</code>.
</p>

---

#### <ins>Step-by-step behavior:</ins>

- Initializes:
  - <code>w</code> as a zero vector of size <code>n</code>
  - <code>b</code> as <code>0</code>
- Calls <code>gradient_descent</code> to optimize the parameters using:
  - Input features <code>X</code>
  - Labels <code>y</code>
  - Learning rate <code>alpha</code>
  - Number of iterations <code>num_iters</code>
- Computes the final predictions and cost:
  - <code>z = X * w + b</code>
  - <code>a = sigmoid(z)</code>
  - <code>J = compute_cost(a, y)</code>

---

#### <ins>What do the outputs mean?</ins>

- <code>w</code>: trained weights for the features (12288x1)
- <code>b</code>: trained bias (scalar)
- <code>J</code>: final cost after training — lower is better

---

### <code>run_training</code>

<p><strong>Purpose:</strong><br>
This is the main function that runs the entire training pipeline. It prepares the dataset, trains the model, and reports the final cost and training accuracy.
</p>

---

#### <ins>Step-by-step behavior:</ins>

1. Calls <code>export_images_to_csv</code>:
   - Loads images from <code>cat_folder</code> and <code>non_cat_folder</code>
   - Converts and labels them (1 for cat, 0 for non-cat)
   - Appends them to <code>output_csv</code>
2. Calls <code>get_characteristics</code> to:
   - Load the image data and labels from CSV
   - Normalize the feature matrix <code>X</code>
3. Calls <code>train_model</code> to:
   - Train the logistic regression model
   - Output parameters <code>w</code>, <code>b</code> and the final cost <code>J</code>
4. Calls <code>predict</code> to:
   - Predict labels for the training data
   - Calculate accuracy based on true labels

---

#### <ins>Inputs:</ins>

- <code>cat_folder</code> – path to folder with cat images
- <code>non_cat_folder</code> – path to folder with non-cat images
- <code>output_csv</code> – path to CSV where training data is saved
- <code>alpha</code> – learning rate (e.g. <code>0.01</code>)
- <code>num_iters</code> – number of training iterations (e.g. <code>1000</code>)

---

#### <ins>Outputs:</ins>

- Prints the final cost <code>J</code>
- Prints the training set accuracy as a percentage

---

#### <ins>Example usage in Octave:</ins>

```matlab
run_training('dataset/cats', 'dataset/not_cats', 'data.csv', 0.01, 1000);
```

