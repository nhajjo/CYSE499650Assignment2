# Assignment #2: Sentiment Classification with Neural Language Models

---------------------------------------------------------------------------------------

## Assignment Overview

This project creates a binary sentiment classifier that is used for movie reviews by using a retrained DistilBERT transformer model.  This classifier also predicts if the review is going to be either 'Negative = 0' or 'Positive = 1'. 

This model was fine-tuned by using the provided training dataset and it also was evaluated on the provided public test dataset.

----------------------------------------------------------------------------------------

## Assignment Files

The following files for this assignment are:

- README.md
- stage1_notebook.ipynb
- requirements.txt
- model_checkpoint/ **(Contains the saved required DistilBERT model and tokenizer files into order to reload the trained model.)**
- public_test_predictions.csv

---------------------------------------------------------------------------------------

## Assignment Model/Training Details
  - **Model Used:** DistilBERT ('distilbert-base-uncased')
  - **Frameworks:** Hugging Face Transformers and PyTorch
  - **Task:** Binary Sentiment Classification
  - **Optimizer:** AdamW
  - **Learning Rate:** 2E-5 or 2 * 10^-5
  - **Batch Size:** 8
  - **Epochs:** 5
  - **Weight Decay:** 0.01
  - **Maximum Sequence Length:** 256

Note: This model was fine-tuned for binary sentiment classification.  In order to train this imbalanced training dataset, a weighted cross-entropy loss was used during the training.  And also, a created train/validation split was used to keep this class distribution.

---------------------------------------------------------------------------------------

## Dataset

**Training Data:** There are 240 Movie Reviews, 180 Positive Reviews, and 60 Negative Reviews.  Again, the created train/validation split was used while training all of this data.

---------------------------------------------------------------------------------------

## Evaluation

This trained model was evaluated by using the given public test dataset called "public_test.csv".  The Public Test Accuracy was also 63.25% while running the code.

---------------------------------------------------------------------------------------

## Running the Jupyter Notebook

#1. Install the required Python packages from "requirements.txt" by using the following command:

'''
pip install -r requirements.txt
'''

#2. Launch the 'Jupyter Notebook' and place the following files inside of the main assignment directory: "train.csv" and "public_test.csv"

#3 Open up the following file: "stage1_notebook.ipynb"

#4. Run all of the notebook cells in order from the top all the way down to the bottom.

#5. The notebook will do all of the following: 
  - Train the DistilBERT model.
  - Save the best checkpoint inside of the folder "model_checkpoint/".
  - Evaluate the public test set model.
  - Generate a new csv file called "public_test_predictions.csv".

---------------------------------------------------------------------------------------

## Model Checkpoint

**This trained model can be reloaded by writing the following chunk of code:**

'''
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("model_checkpoint")

tokenizer = AutoTokenizer.from_pretrained("model_checkpoint")
'''

---------------------------------------------------------------------------------------

## Code Output

It generated "public_test_predictions.csv" with the 'id, predicted, label' format where 'Negative = 0' or 'Positive = 0'.

---------------------------------------------------------------------------------------

## Public Test Accuracy/Result

63.25%

---------------------------------------------------------------------------------------

## For Stage #2 of this Assignment

The "model_checkpoint/" folder will be loaded directly as a method for the hidden test set without the need of having to retrain any of this model's data.

---------------------------------------------------------------------------------------

## Use of Generative A.I.

ChatGPT Atlas was used to help me recognize, debug, troubleshoot, and solve the tasks and to improve the data/results.  All of the code was reviewed, tested, and executed before submitting everything.

---------------------------------------------------------------------------------------
