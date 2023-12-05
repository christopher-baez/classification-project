from sklearn.model_selection import train_test_split

def prep_telco(df):
    """
    this  function takes in the telco database and cleans it
    deletes
        -payment type id
        -internet service id
        -contract type id
        -customer id
    turns total charges into a float by deleting empty spaces
    """
    df = df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id','Unnamed: 0', 'paperless_billing'])

    df = df[df.total_charges != ' ']

    df['total_charges'] = df.total_charges.astype('float64')

    # combining the add on columns into 1
    # adding 1 for addons and 0 for no addons
    df['has_add_ons'] = df[['online_security'
        , 'online_backup', 'device_protection', 'tech_support']].apply(
        lambda row: 1 if any(x == 'Yes' for x in row) else 0,
        axis=1
    )
    # now dropping the combined columns
    df = df.drop(columns=['online_security','online_backup', 'device_protection', 'tech_support'])

    return df

def preprocess_data(df):
    """
    Preprocesses the data by encoding various features into binary format and removing the original columns.

    Parameters:
    df (DataFrame): The DataFrame to be preprocessed.

    Returns:
    DataFrame: The preprocessed DataFrame.
    """
    # Switch 'internet_service_type' to binary: 1 for having internet, 0 for none
    df['has_internet'] = df['internet_service_type'].apply(lambda x: 0 if x == 'None' else 1)

    # Convert 'multiple_lines', 'streaming_tv', 'streaming_movies' to binary: 1 for 'Yes', 0 for 'No'
    lines = ['multiple_lines', 'streaming_tv', 'streaming_movies']
    for col in lines:
        df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)

    # Convert 'contract_type' to binary: 1 for 'Month-to-month', 0 for others
    df['monthly_contract'] = df['contract_type'].apply(lambda x: 1 if x == 'Month-to-month' else 0)

    # Convert 'payment_type' to binary: 1 for 'automatic' in string, 0 otherwise
    df['automatic_payment'] = df['payment_type'].apply(lambda x: 1 if 'automatic' in x else 0)

    # Convert 'partner', 'dependents', 'phone_service' to binary: 1 for 'Yes', 0 for 'No'
    yn = ['partner', 'dependents', 'phone_service']
    for col in yn:
        df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)

    # Encode 'gender' as binary: 1 for 'Male', 0 for others
    df['is_male'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)

    # Convert 'churn' to binary: 1 for 'Yes', 0 for 'No'
    df['churn'] = df['churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Drop original columns that were encoded
    columns_to_drop = ['internet_service_type', 'contract_type', 'payment_type', 'gender']
    df = df.drop(columns=columns_to_drop)

    return df


def categorize_columns(df):
    """
    Categorizes columns of a DataFrame into binary categorical variables or multiclass/continuous variables.

    Parameters:
    df (DataFrame): The DataFrame to categorize columns from.

    Returns:
    tuple: Two lists containing the names of binary categorical and multiclass/continuous variables, respectively.
    """
    cat_var = []
    cont_var = []

    for column in df.columns:
        unique_values = df[column].unique()
        print(f"'{column}': {unique_values}")

        if df[column].nunique() == 2:
            cat_var.append(column)
        elif df[column].nunique() > 2:
            cont_var.append(column)

    return cat_var, cont_var


def splitting_data(df, col):
    '''
    this function splits my data focusing on my target variable
    parameters
        df= datafram
        col= column
    returns
        train
        validate
        test
    '''

    # first split
    train, validate_test = train_test_split(df,
                                            train_size=0.6,
                                            random_state=123,
                                            stratify=df[col]
                                            )

    # second split
    validate, test = train_test_split(validate_test,
                                      train_size=0.5,
                                      random_state=123,
                                      stratify=validate_test[col]

                                      )
    return train, validate, test