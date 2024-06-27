# Cross Validation with Multiple Classifiers

This project demonstrates the use of cross-validation with different classifiers on the Iris dataset using Python and scikit-learn. Cross-validation is a statistical method used to estimate the skill of machine learning models. The Iris dataset is a classic dataset in machine learning and statistics, which contains 150 samples of iris flowers with four features each.

## Output
![image](https://github.com/sarvesh-2109/Cross-Validation/assets/113255836/31a838b6-dd0a-430e-8aa7-288f7cfed20e)


## Project Structure

- **Cross_Validation.ipynb**: The Jupyter notebook file containing the code for loading the dataset, applying cross-validation, and evaluating different classifiers.

## Classifiers Used

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Support Vector Classifier (SVC)**

## Requirements

- Python 3.x
- Jupyter Notebook
- Libraries:
  - scikit-learn
  - numpy
  - pandas

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/sarvesh-2109/Cross-Validation.git
    ```
2. Change the directory:
    ```bash
    cd Cross-Validation
    ```
3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Jupyter notebook:
    ```bash
    jupyter notebook Cross_Validation.ipynb
    ```

## Usage

1. **Load the Iris dataset:**
    ```python
    from sklearn.datasets import load_iris
    iris = load_iris()
    ```

2. **Cross-validation with Logistic Regression:**
    ```python
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    cross_val_score(LogisticRegression(), iris.data, iris.target)
    ```

3. **Cross-validation with Random Forest Classifier:**
    ```python
    from sklearn.ensemble import RandomForestClassifier
    cross_val_score(RandomForestClassifier(), iris.data, iris.target)
    ```

4. **Cross-validation with Support Vector Classifier (SVC):**
    ```python
    from sklearn.svm import SVC
    cross_val_score(SVC(), iris.data, iris.target)
    ```

## Results

The cross-validation scores for each classifier provide an estimate of their performance on the Iris dataset. These scores help in understanding which classifier performs the best for this specific dataset.

## Conclusion

Cross-validation is a powerful technique to evaluate the performance of different machine learning models. In this project, we have used three different classifiers and compared their performance on the Iris dataset. The results can guide the selection of the best model for similar classification tasks.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

