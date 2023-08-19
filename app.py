import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load and preprocess the data
data = pd.read_csv("d.csv")
data = data.drop(columns=['Name'])

# Encode Role using LabelEncoder
le = LabelEncoder()
data['Role'] = le.fit_transform(data['Role'])

# Splitting the data
X = data.drop("Role", axis=1)
y = data["Role"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
lr_model = LogisticRegression(max_iter=10000)
lr_model.fit(X_train, y_train)

def predict_job_role(scores):
    job_role_code = lr_model.predict([scores])
    job_role = le.inverse_transform(job_role_code)[0]
    return job_role

# Test questions and correct answers for each subject
fields_questions = {
    "C": [
        ("Which of the following is not a keyword in C language?", ["class", "int", "float", "char"]),
        ("What is the output of printf(\"%d\")?", ["Undefined behavior", "0", "Segmentation fault", "Compiler error"]),
        ("How would you correctly allocate memory for an array of 10 ints in C?", ["int* arr = (int*) malloc(10 * sizeof(int));", "int arr[10];", "int* arr = malloc(10);", "None of the above"]),
        ("What keyword is used to deallocate dynamically allocated memory?", ["free", "delete", "remove", "None of the above"]),
        ("What header file needs to be included for using malloc() function in C?", ["stdlib.h", "stdio.h", "malloc.h", "None of the above"]),
    ],
    "C++": [
        ("Which keyword is used to prevent any changes in the variable in C++?", ["const", "static", "volatile", "extern"]),
        ("What is the full form of STL in C++?", ["Standard Template Library", "Standard Type Language", "Static Typing Language", "Syntax Tree Language"]),
        ("In C++, what does the 'this' pointer refer to?", ["The object itself", "The class itself", "The method that is currently being called", "The module in which the class was defined"]),
        ("Which C++ library would you use for common mathematical operations?", ["cmath", "math.h", "math", "stdlib"]),
        ("Which keyword is used in C++ for exception handling?", ["try-catch", "throw", "error", "exception"]),
    ],
    "Problem Solving Skills": [
        ("What is a common approach to breaking down a complex problem into smaller manageable subproblems?", ["Divide and Conquer", "Merge Sort", "Bubble Sort", "Greedy Algorithm"]),
        ("What does 'algorithmic thinking' involve?", ["Designing a step-by-step process to solve a problem", "Drawing flowcharts", "Writing code without planning", "Using pre-built libraries"]),
        ("What is the purpose of pseudocode in problem-solving?", ["Describing the algorithm in human-readable language before coding", "Writing code in a specific programming language", "Debugging code", "Optimizing code for performance"]),
        ("When might you use a 'greedy algorithm'?", ["When you want to make the locally optimal choice at each step", "When you want to explore all possible solutions", "When you want to use recursion", "When you want to perform random operations"]),
        ("What is the significance of 'time complexity' in problem-solving?", ["It measures the amount of time an algorithm takes to run as a function of input size", "It measures the number of statements in an algorithm", "It measures the number of variables used in an algorithm", "It measures the efficiency of an algorithm"]),
    ],
    "Python": [
        ("What will be the output of the following Python code?\n\nx = [10, [3.141, 20, [30, 'baz', 2.718]], 'foo']\nprint(x[1][2][1])", ["baz", "foo", "3.141", "20"]),
        ("What data type is this in Python: {1:'one', 2:'two'}?", ["Dictionary", "List", "Tuple", "String"]),
        ("Which Python data structure is used for storing homogeneous sequential data and allows duplicates?", ["List", "Set", "Dictionary", "Tuple"]),
        ("In Python, what does the 'self' keyword refer to?", ["The instance of the class", "The class itself", "The method that is currently being called", "The module in which the class was defined"]),
        ("Which of the following functions would you use to read a file line by line in Python?", ["readlines()", "readline()", "read()", "None of the above"]),
    ],
    "ML": [
        ("What does the 'ML' in 'Machine Learning' stand for?", ["Machine Learning", "Markup Language", "Maximum Likelihood", "Manual Labor"]),
        ("Which of the following is a type of machine learning?", ["Supervised Learning", "Superior Learning", "Substantial Learning", "Sufficient Learning"]),
        ("What is the purpose of a validation set in machine learning?", ["Used for tuning the parameters (hyperparameter optimization) and providing an unbiased evaluation of a model", "Used for final testing of the model", "Used for training the model", "None of the above"]),
        ("What does 'underfitting' mean in the context of Machine Learning?", ["The model is too simple and does not capture the complexity of the data", "The model is too complex and is fitted too much to the training data", "The model is perfect", "None of the above"]),
        ("What is 'cross-validation' in Machine Learning?", ["A technique for assessing how the results of a statistical analysis will generalize to an independent data set", "A technique for making the model more complex", "A technique for simplifying the model", "None of the above"]),
    ],
    "AI": [
        ("What does 'AI' stand for?", ["Artificial Intelligence", "Automated Input", "Automatic Installer", "Active Interface"]),
        ("Which of the following fields is AI most likely to impact significantly?", ["Healthcare", "Retail", "Agriculture", "All of the above"]),
        ("What is the term for an AI that has the ability to understand, learn, and apply knowledge?", ["Artificial General Intelligence (AGI)", "Artificial Limited Intelligence (ALI)", "Artificial Super Intelligence (ASI)", "Artificial Narrow Intelligence (ANI)"]),
        ("Which technique does AI use to simulate human intelligence?", ["Neural networks", "Natural networks", "Artificial networks", "Digital networks"]),
        ("What is 'Deep Learning' in the context of AI?", ["A subset of machine learning that is based on artificial neural networks with representation learning", "A method of AI that is based on binary decision processes", "A method of AI that uses rule-based algorithms", "None of the above"]),
    ],
    "Pandas": [
        ("Which Python library is commonly used for data manipulation and analysis?", ["Pandas", "Numpy", "Matplotlib", "Scikit-learn"]),
        ("What is a DataFrame in Pandas?", ["A two-dimensional, size-mutable, and heterogeneous tabular data structure", "A one-dimensional array", "A multi-dimensional array", "A dictionary"]),
        ("How can you drop a column named 'Age' from a Pandas DataFrame df?", ["df.drop('Age', axis=1, inplace=True)", "df.remove('Age')", "df.drop_column('Age')", "df.delete('Age')"]),
        ("What Pandas function is used to fill missing values in a DataFrame?", ["fillna()", "fill()", "replace()", "missing()"]),
        ("Which method is used to group data in a Pandas DataFrame?", ["groupby()", "aggregate()", "combine()", "categorize()"]),
    ],
    "SQL": [
        ("What does 'SQL' stand for?", ["Structured Query Language", "Sequential Query Language", "Statistical Query Language", "Single Query Language"]),
        ("Which SQL keyword is used to retrieve data from a database?", ["SELECT", "FETCH", "RETRIEVE", "SEARCH"]),
        ("What is a 'JOIN' clause used for in SQL?", ["To combine rows from two or more tables based on a related column", "To create a new table", "To delete data from a table", "To update records in a table"]),
        ("Which SQL clause is used to filter the results of a query?", ["WHERE", "FILTER", "SORT", "GROUP"]),
        ("What is the purpose of the 'ORDER BY' clause in SQL?", ["To sort the result set based on specified columns", "To group the result set based on specified columns", "To filter the result set based on conditions", "To perform mathematical operations on the result set"]),
    ],
    "Statistics": [
        ("What is the mean of a set of numbers?", ["The average of all the numbers", "The middle number", "The most frequently occurring number", "The difference between the largest and smallest numbers"]),
        ("What is the median?", ["The middle number when the numbers are sorted", "The average of all the numbers", "The most frequently occurring number", "The difference between the largest and smallest numbers"]),
        ("What does the 'standard deviation' measure?", ["The dispersion or spread of a set of values", "The central tendency of a set of values", "The highest value in a set", "The difference between the median and mode"]),
        ("What is a 'normal distribution' in statistics?", ["A symmetric, bell-shaped distribution that follows a specific mathematical formula", "A distribution with extremely high variance", "A distribution with no variance", "A distribution with negative skewness"]),
        ("What is the 'P-value' in hypothesis testing?", ["The probability of observing a test statistic as extreme as the one computed from the sample data", "The probability of making a Type II error", "The probability of making a Type I error", "The probability of a two-tailed test"]),
    ],
    "Debugging": [
        ("What is debugging?", ["The process of identifying and fixing errors in software code", "The process of writing new code", "The process of documenting code", "The process of optimizing code"]),
        ("What does 'print debugging' involve?", ["Inserting print statements in the code to track the flow and values of variables", "Using a debugger tool to analyze the code", "Rewriting the entire code", "Deleting parts of the code"]),
        ("What is a 'stack trace'?", ["A list of function calls and their sequence at the point of an error", "A graphical representation of data flow", "A way to measure CPU usage", "A type of data structure"]),
        ("Which type of error occurs when the code violates the rules of the programming language?", ["Syntax error", "Logic error", "Runtime error", "Semantic error"]),
        ("What is a 'breakpoint' in the context of debugging?", ["A point in the code where the debugger will pause execution for inspection", "A point in the code where the program terminates", "A point where the code breaks due to an error", "A point where the code stops running"]),
    ],
}

correct_answers = {
    "C": ["class", "Undefined behavior", "int* arr = (int*) malloc(10 * sizeof(int));", "free", "stdlib.h"],
    "C++": ["const", "Standard Template Library", "The object itself", "cmath", "try-catch"],
    "Problem Solving Skills": ["Divide and Conquer", "Designing a step-by-step process to solve a problem", "Describing the algorithm in human-readable language before coding", "When you want to make the locally optimal choice at each step", "It measures the amount of time an algorithm takes to run as a function of input size"],
    "Python": ["baz", "Dictionary", "List", "The instance of the class", "readlines()"],
    "ML": ["Machine Learning", "Supervised Learning", "Used for tuning the parameters (hyperparameter optimization) and providing an unbiased evaluation of a model", "The model is too simple and does not capture the complexity of the data", "A technique for assessing how the results of a statistical analysis will generalize to an independent data set"],
    "AI": ["Artificial Intelligence", "All of the above", "Artificial General Intelligence (AGI)", "Neural networks", "A subset of machine learning that is based on artificial neural networks with representation learning"],
    "Pandas": ["Pandas", "A two-dimensional, size-mutable, and heterogeneous tabular data structure", "df.drop('Age', axis=1, inplace=True)", "fillna()", "groupby()"],
    "SQL": ["Structured Query Language", "SELECT", "To combine rows from two or more tables based on a related column", "WHERE", "To sort the result set based on specified columns"],
    "Statistics": ["The average of all the numbers", "The middle number when the numbers are sorted", "The dispersion or spread of a set of values", "A symmetric, bell-shaped distribution that follows a specific mathematical formula", "The probability of observing a test statistic as extreme as the one computed from the sample data"],
    "Debugging": ["The process of identifying and fixing errors in software code", "Inserting print statements in the code to track the flow and values of variables", "A list of function calls and their sequence at the point of an error", "Syntax error", "A point in the code where the debugger will pause execution for inspection"],
}


def take_test():
    st.title('Skill Test')
    skillset_scores = {}
    for field in fields_questions:
        st.subheader(f"Exam for {field}")
        score = 0
        for i in range(len(fields_questions[field])):
            question, options = fields_questions[field][i]
            answer = st.radio(question, options)
            if answer == correct_answers[field][i]:
                score += 1
        skillset_scores[field] = score

    if st.button("Submit"):
        return skillset_scores

def show_results(skillset_scores):
    st.title('Results')

    total_score = sum(skillset_scores.values())
    
    if total_score == 0:
        st.warning("Prepare well and retake the test!")
        return  # Exit the function early
    
    # If there are correct answers, continue with job role prediction and score display
    scores = list(skillset_scores.values())
    job_role = predict_job_role(scores)
    st.markdown(f"<h2 style='text-align: center; color: violet; font-weight: bold;'>Suggested job role based on your skillset: <br>{job_role}</h2>", unsafe_allow_html=True)

    st.subheader('Your scores:')
    for field, score in skillset_scores.items():
        st.text(f"{field}: {score}")

    df_scores = pd.DataFrame.from_dict(skillset_scores, orient='index', columns=['Score'])
    df_scores = df_scores.reset_index()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.pie(df_scores['Score'], labels=df_scores['index'], autopct='%1.1f%%')
    ax.set_title('Skillset scores')
    st.pyplot(fig)

def main():
    skillset_scores = take_test()
    if skillset_scores is not None:
        st.session_state.skillset_scores = skillset_scores
        st.write('', '', '')
        st.markdown(":white_check_mark: You've completed the test! Here are your results:")
        show_results(st.session_state.skillset_scores)

if __name__ == "__main__":
    main()
