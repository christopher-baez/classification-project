import os
import numpy as np
import pandas as pd
import env

def check_file_exists(filename, query, url):
    ''' this function is for python to check if the file you are looking for exist in the computer'''
    if os.path.exists(filename):
        print('this file exists, reading csv')
        df = pd.read_csv(filename)
    else:
        print('this file doesnt exist, read from sql, and export to csv')
        df = pd.read_sql(query, url)
        df.to_csv(filename)

    return df
def get_telco_data():
    ''' function to get data from SQL  and turning it to a dataframe'''
    url = env.get_db_url('telco_churn')
    query = '''
    select *
    from customers
        left join contract_types
            using (contract_type_id)
        left join internet_service_types
            using (internet_service_type_id)
        left join payment_types
            using (payment_type_id)
    '''

    filename = 'telco_churn.csv'


    # call the check_file_exists fuction
    df = check_file_exists(filename, query, url)

    df.set_index('customer_id', inplace=True)
    return df

