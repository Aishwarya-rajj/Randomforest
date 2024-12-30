## Project Overview

This project implements a Decision Tree model to classify the Iris dataset using Scikit-learn. The model visualizes decision boundaries and predicts the species of iris flowers based on input features such as petal and sepal dimensions.

## Objective

The goal of this project is to:

- Build and visualize a Decision Tree to classify the Iris dataset.
- Evaluate the performance of the model through accuracy and classification reports.
- Use the trained model to predict flower species based on new data.
- Create a Jupyter Notebook to showcase model building, visualization, and analysis.

## Dataset

The dataset used is the **Iris** dataset, a classic dataset in machine learning often used for classification tasks. It consists of 150 samples of iris flowers with three species (Setosa, Versicolor, Virginica).

### Features Used:

- **Sepal Length (cm)** - Length of the sepal.
- **Sepal Width (cm)** - Width of the sepal.
- **Petal Length (cm)** - Length of the petal.
- **Petal Width (cm)** - Width of the petal.

### Target Variable:

- **Species** - Class label representing the species of the iris flower.

## Technologies and Libraries Used:

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn, Jupyter Notebook

## Implementation

### Key Steps:

1. **Data Loading and Exploration**: The Iris dataset is loaded using `load_iris()` from Scikit-learn. Basic exploratory data analysis (EDA) is conducted to understand the structure of the data.

2. **Train-test Split**: The dataset is split into 70% training and 30% testing data to evaluate the model's performance on unseen data.

3. **Model Training and Visualization**:

   - A Decision Tree classifier is trained on the training data.
   - The depth of the tree is limited to avoid overfitting (e.g., `max_depth=1`).
   - The trained tree is visualized using `plot_tree()` to illustrate the decision boundaries and rules.

4. **Jupyter Notebook Creation**:
   - The entire workflow (data loading, training, visualization, and evaluation) is implemented in a Jupyter Notebook.
   - Markdown cells are used to explain each step for better readability and understanding.

5. **Evaluation**:

   - Predictions are made on the test set.
   - Accuracy and a classification report are generated to evaluate the model's performance.

## Results

- **Accuracy** – Displays the accuracy of the model on the test set.
- **Classification Report** – Provides precision, recall, and F1-score for each class (species).
- **Decision Tree Visualization** – Shows the structure of the tree and decision paths.

## Project Structure
```
|-- iris_decision_tree
    |-- data_exploration.ipynb
    |-- decision_tree_iris.py
    |-- results
        |-- decision_tree_plot.png
    |-- README.md
```

## How to Run the Project

1. Clone the repository:
```
git clone https://github.com/username/iris_decision_tree.git
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the Python script or Jupyter Notebook:
```
jupyter notebook data_exploration.ipynb
```

## Future Enhancements

- Tune hyperparameters (e.g., `max_depth`, `min_samples_split`) to improve accuracy.
- Experiment with different datasets to generalize the model.
- Implement pruning techniques to reduce overfitting.
- Deploy the model using Flask or Streamlit for real-time classification.

