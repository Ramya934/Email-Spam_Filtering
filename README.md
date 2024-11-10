# Email-Spam-Filtering

This project implements a Spam Filtering System using machine learning to classify emails as "Spam" or "Ham" (not spam). The system applies Natural Language Processing (NLP) techniques to analyze the content of emails and uses various classification algorithms to detect spam emails with high accuracy.

**Project Overview**

Email spam filtering is an essential tool in modern communication systems, helping to reduce unwanted messages in users' inboxes. This project preprocesses email text, converts it into a numerical format, and applies supervised learning techniques to train a classifier on labeled data.

**Key Features**


Text Preprocessing: Cleans and prepares email text for analysis.

Feature Extraction: Converts email text into numerical features using techniques like TF-IDF (Term Frequency-Inverse Document Frequency).

Classification Models: Uses algorithms like Naive Bayes, Logistic Regression, or Support Vector Machines (SVM) for classification.

Evaluation Metrics: Assesses the performance of the models using metrics such as accuracy, precision, recall, and F1-score.

**Project Workflow**
1. Import Libraries

2. Load and Preprocess the Dataset
   
The dataset contains labeled email messages, where each message is marked as either "spam" or "ham". The preprocessing steps include:

Removing special characters and numbers to clean the text.
Tokenization to break text into words.
Stop-word removal to eliminate common but uninformative words 
Lemmatization or stemming to reduce words to their root forms.

**3. Train Classification Models**

Train one or more classification algorithms to detect spam emails:

Naive Bayes: Commonly used for text classification due to its simplicity and effectiveness.


Logistic Regression: A linear model useful for binary classification.

Support Vector Machine (SVM): Effective for text data classification.

**5. Evaluate Models**

Evaluate the models on test data using metrics such as:

**Accuracy:** The percentage of correctly classified emails.

**Precision:** The proportion of true spam emails among all emails classified as spam.

**Recall: ** The proportion of correctly identified spam emails among all actual spam emails.

**F1-Score:** The harmonic mean of precision and recall, providing a balanced evaluation metric.

**Output**

The code outputs the accuracy and other evaluation metrics for each model, helping to identify the most effective algorithm for spam detection.
