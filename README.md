# Data Preparation

Using the provided **WikiArt-Emotions-Ag4.tsv** and **WikiArt-info.tsv**, we can create a labeled version of the dataset **wikiart_labels_imageonly.csv** by running **generate_labels_imageonly.py**.

In order to download the images, run the **download_images.py** script.

I did some experiments on the dataset filtered down to the most common 10 emotions (the other 10 had <%1 representation in the label space). To create a filtered version of the dataset, run **filter_dataset.py** (I would first do an evaluation on the base dataset before this). To see the dataset distribution, run **analyze_dataset.py**.

To train models on the dataset, we use a predetermined train-test-split written to disk. To prepare this, run **prepare_split.py**.

# Model Training

### CNN

To train the CNN model, first ensure that the desired training split is in ./splits/

Then, run the **train.sh script**, changing any hyperparameters if desired (ensure you are using the right dataset). This will run the **wikiart_emotions_cnn.py** file and generate a model and logs for it.

### ViT

To train the Vision Transformer model, first ensure that the desired training split is in ./splits/

Then, run the **train_vit.sh script**, changing any hyperparameters if desired (ensure you are using the right dataset). This will run the **wikiart_emotions_vit.py** file and generate a model and logs for it.

# Evaluation

First, check the hyperparameters (including the dataset file) that are set in **evaluate.sh**. Run the file, which will run **evaluate.py** to output an evaluation of the model performance.

