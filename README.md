# Customer Retention Predictor

This project implements a machine learning model to predict customer retention. The model analyzes various customer data to determine the likelihood of a customer staying with a company. The project aims to help businesses improve their customer retention strategies by identifying at-risk customers.

## Table of Contents
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Tech Stack

- **Python**: The main programming language used for the implementation.
- **Jupyter Notebook**: For interactive code development and visualization.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib**: For plotting and visualization.
- **Seaborn**: For statistical data visualization.
- **Scikit-learn**: For machine learning algorithms.

## Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Installation

To run this project, you need to have Python installed along with several libraries. You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. **Clone the repository:**

    ```bash
    git clone https://github.com/ayushp-18/Customer-Retention-Predictor.git
    cd Customer-Retention-Predictor
    ```

2. **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Jupyter Notebook:**

    Open `Customer_Retention_Predictor.ipynb` using Jupyter Notebook or Jupyter Lab and run all the cells to see the model in action.

## Examples

Below are some examples of how to use the model:

1. **Loading the Data:**

    ```python
    import pandas as pd

    data = pd.read_csv('path_to_data.csv')
    ```

2. **Preprocessing the Data:**

    ```python
    from sklearn.model_selection import train_test_split

    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

3. **Training the Model:**

    ```python
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    ```

4. **Making Predictions:**

    ```python
    predictions = model.predict(X_test)
    ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
