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

def continuous_categorical_relationship(dataframe, continuous_var, categorical_var, alpha=0.05):
    # Create a box plot with categorical_var on the x-axis and continuous_var on the y-axis
    sns.boxplot(data=dataframe, x=categorical_var, y=continuous_var)
    plt.title(f'{continuous_var} vs {categorical_var}')
    plt.show()

    # Check if there are more than two categories
    categories = dataframe[categorical_var].unique()
    if len(categories) > 2:
        # Perform ANOVA for more than two categories
        print("Performing ANOVA:")
        anova_result = stats.f_oneway(
            *(dataframe[dataframe[categorical_var] == category][continuous_var] for category in categories))
        p = anova_result.pvalue
    else:
        # Perform t-test for two categories
        print("Performing T-test:")
        group1 = dataframe[dataframe[categorical_var] == categories[0]][continuous_var]
        group2 = dataframe[dataframe[categorical_var] == categories[1]][continuous_var]
        t_test_result = stats.ttest_ind(group1, group2)
        p = t_test_result.pvalue

    # Print the null and alternative hypotheses
    print("\nH0: There is NO difference in means of {} across different categories of {}".format(continuous_var,
                                                                                                 categorical_var))
    print("Ha: There IS a difference in means of {} across different categories of {}".format(continuous_var,
                                                                                              categorical_var))

    # Print the results of the test
    if p < alpha:
        print(f"\nReject the null hypothesis. The p-value is {p}, which is less than alpha {alpha}.")
        print(
            f"There is a statistically significant difference in {continuous_var} across different categories of {categorical_var}.")
    else:
        print(f"\nFail to reject the null hypothesis. The p-value is {p}, which is greater than alpha {alpha}.")
        print(
            f"There is no statistically significant difference in {continuous_var} across different categories of {categorical_var}.")

# Example usage
# analyze_relationship(dataframe, 'var1_name', 'var2_name')

