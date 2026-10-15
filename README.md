# Banking Customer Intent Classifier

A Natural Language Processing (NLP) machine learning project that classifies banking customer messages into 77 intent categories using TF-IDF and Linear Support Vector Classification (LinearSVC).

## Project Overview

Customer support teams receive messages about card problems, transfers, payments, account access, and other banking services. Automatically identifying the intent behind each message can help organize support requests and route them to the appropriate team.

This project builds a text classification pipeline that predicts the intent of a banking customer's message.

## Results

**Best model:** TF-IDF + LinearSVC

| Metric | Development Set | Test Set |
|---|---:|---:|
| Accuracy | 88.93% | 88.93% |
| Macro F1-score | 88.93% | 88.93% |

The model was selected based on its development-set performance and evaluated on the held-out test set.

## Dataset

This project uses the **BANKING77** dataset, which contains banking-related customer queries grouped into 77 intent categories.

The dataset provides separate training, development, and test splits.

Dataset source: [BANKING77 on Hugging Face](https://huggingface.co/datasets/PolyAI/banking77)

Please review the dataset's license and terms before redistributing any data.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- LinearSVC
- Joblib
- Kaggle Notebooks

## Project Workflow

1. Load and inspect the dataset.
2. Check the dataset structure, missing values, and intent categories.
3. Convert customer messages into numerical features using TF-IDF.
4. Train a Logistic Regression baseline.
5. Train a LinearSVC model.
6. Compare both models using development accuracy and macro F1-score.
7. Select LinearSVC based on development-set results.
8. Evaluate the selected model on the test set.
9. Test predictions using new customer messages.
10. Save the fitted classification pipeline.

## Example Predictions

| Customer Message | Predicted Intent |
|---|---|
| I lost my bank card and need a replacement. | `lost_or_stolen_card` |
| I transferred money but it has not arrived. | `transfer_not_received_by_recipient` |
| I want to change my account password. | `passcode_forgotten` |

These are example predictions from the trained model. The final example illustrates that related intents can sometimes be confused.

## How It Works

The pipeline has two main components:

**1. TF-IDF Vectorizer**

Converts customer messages into numerical features based on the importance of words and word combinations.

**2. LinearSVC Classifier**

Uses the numerical features to predict one of the 77 banking intent categories.

The fitted pipeline includes both components, allowing raw text messages to be passed directly to the model.

## Current Project Status

- [x] Dataset exploration
- [x] Logistic Regression baseline
- [x] LinearSVC model training
- [x] Development-set model comparison
- [x] Test-set evaluation
- [x] Example predictions
- [x] Model saved locally
- [ ] Repository notebook and supporting files uploaded
- [ ] Prediction API
- [ ] Interactive application
- [ ] Deployment

## Future Improvements

- Analyze misclassified messages and per-category performance.
- Improve predictions for similar customer intents.
- Build a REST API using FastAPI.
- Develop a simple interface for submitting customer messages.
- Deploy the model as a usable application.

## Limitations

This model was trained on BANKING77, a benchmark dataset of banking-related queries. Its performance on real customer messages may differ, especially when messages use unfamiliar wording, contain multiple requests, or come from a different domain.

The current project performs intent classification; it does not independently resolve banking issues or provide financial advice.

## Author

**Feyisa Habtamu Jember**

GitHub: [@feyisahabtamu](https://github.com/feyisahabtamu)
