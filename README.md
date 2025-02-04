# **EmoNet: Advanced Emotion Classification Using NLP Techniques** 

## **Overview**
**EmoNet** is a Natural Language Processing (NLP) project designed to classify **emotions** expressed in textual data. The model predicts emotions such as **happiness, sorrow, rage, scare, care, and amaze** using state-of-the-art NLP techniques. This project applies **machine learning models** to achieve high accuracy in emotion classification, with potential applications in **sentiment analysis, customer feedback analysis, and mood detection in conversational interfaces**.

---

## **Dataset**
- **Source**: A labeled dataset containing **15,000 text samples** annotated with six distinct emotions. Find this in the [dataset](https://github.com/gandharpansare/Emonet_Advanced_Emotion_Classification/tree/main/dataset) folder.
- **Attributes**:
  - **Text Data**: Statements expressing various emotions.
  - **Emotion Labels**: One of six emotions assigned to each statement.
- **Imbalance Considerations**:
  - **Happiness** and **sorrow** were the most common emotions.
  - **Amaze** had the fewest examples, requiring special handling to balance the dataset.

---

## **Data Preprocessing & Feature Engineering**
- **Text Cleaning**:
  - Removed URLs, emails, non-letter characters, and stopwords.
  - Converted text to lowercase.
  - Tokenized sentences into words.
- **Feature Representation**:
  - **TF-IDF Vectorization** to convert text into numerical format.
  - **CountVectorizer** to analyze word frequency.
  - **Word2Vec embeddings** to capture semantic meanings.
- **Visualization**:
  - Generated a **word cloud** to identify dominant words in the dataset.
  - Applied **Latent Dirichlet Allocation (LDA)** for topic modeling, uncovering **8 themes** in the text.

---

## **Model Training**
### **1. Traditional Machine Learning Models**
- **Logistic Regression** – Best-performing model with **90% accuracy** and **0.99 ROC-AUC score**.
- **Random Forest Classifier** – Achieved **89% accuracy**, performed well on balanced classes.
- **Support Vector Classifier (SVC)** – Accuracy of **88%**, slightly behind Logistic Regression.

### **2. Model Selection & Evaluation**
- **Grid search and cross-validation** were used to optimize hyperparameters.
- **ROC-AUC curves** were plotted to assess classification performance across all emotion categories.
- **Final Model**: **Logistic Regression (C=1, L1 penalty, liblinear solver)** selected due to its high precision and recall.

---

## **Final Testing & Deployment**
- **Test Dataset**:
  - A hidden-label dataset was provided for external validation.
  - Predictions were generated using the best-trained model.
  - Accuracy on the test dataset - **85.7%**

---

## **Technologies Used**
- **Programming Language**: Python
- **NLP Libraries**: NLTK, SpaCy, WordCloud, Word2Vec
- **Machine Learning Frameworks**: Scikit-Learn
- **Visualization Tools**: Matplotlib, Seaborn

---

## **Contributors**
- Gandhar Ravindra Pansare
- Raj Dhake
- Sarthak Choudhary
