import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
def categorical_relationship(dataframe, var1, var2, alpha=0.05):
    # Create a bar plot with var1 on the x-axis and var2 on the y-axis
    sns.barplot(data=dataframe, x=var1, y=var2)
    plt.show()

    # Create a crosstab between var1 and var2 and print it
    observed = pd.crosstab(dataframe[var1], dataframe[var2])
    print("Crosstab between {} and {}:".format(var1, var2))
    print(observed)

    # Print the null and alternative hypotheses
    print("\nH0: There is NO relationship between {} and {}".format(var1, var2))
    print("Ha: There IS a relationship between {} and {}".format(var1, var2))

    # Perform the Chi-Squared test
    chi2, p, dof, expected = stats.chi2_contingency(observed)

    # Print the results of the Chi-Squared test
    if p < alpha:
        print(f"\nReject the null hypothesis. My p-value is {p}, which is less than alpha {alpha}.")
        print(f"We can conclude that there is a relationship between {var1} and {var2}.")
    else:
        print(f"\nFail to reject the null hypothesis. My p-value is {p}, which is greater than alpha {alpha}.")
        print(f"We cannot conclude that there is a relationship between {var1} and {var2}.")

# Example usage
# analyze_relationship(dataframe, 'var1_name', 'var2_name')

# Example usage
# analyze_relationship(dataframe, 'var1_name', 'var2_name')
