
### MLOps Zoomcamp - Deployment Homework

This README file provides the answers and steps to the deployment homework exercise.

Questions and Answers

Question 1. Notebook (1 point)
- Answer: 6.24

Question 2. Preparing the output (1 point)
- Answer: 46M

Question 3. Creating the scoring script (1 point)
jupyter nbconvert --to script notebook.ipynb

Question 4. Virtual environment. Hash for Scikit-Learn (1 point)
- Answer: 22914eb0db878c39d1a8c0ec4b17ff31b448fa1bb82b321889ef7b00b8f1e0d0

Question 5. Parametrize the script (1 point)
- Answer: 7.29

Question 6. Docker container (1 point)
- Answer: 0.19

Running the Script with Poetry

To run the provided script using Poetry, use the following command:

poetry run python 04_deployment/starter.py --year 2023 --month 4

Steps to Run the Script

1. Install Poetry: If you haven’t installed Poetry yet, you can do so by following the official installation guide (https://python-poetry.org/docs/#installation).

2. Install Dependencies: Navigate to your project directory and run:
   poetry install

3. Run the Script: Use the command mentioned above to run the script with the required parameters:
   poetry run python 04_deployment/starter.py --year 2023 --month 4

This should execute the script and process the data as described in the homework exercise.

Additional Notes

- Make sure to replace 04_deployment/starter.py with the correct path to your script if it's different.
- Ensure that all required dependencies are listed in your pyproject.toml file for Poetry to manage them correctly.

Feel free to reach out if you have any questions or need further assistance!
