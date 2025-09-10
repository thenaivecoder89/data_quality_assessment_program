import os
os.environ['DISABLE_PANDERA_IMPORT_WARNING'] = 'True'
from dotenv import load_dotenv
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check, DataFrameSchema
from openai import OpenAI

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:,.2f}'.format)

# Initialize environment
load_dotenv()
finance_dataset = os.getenv('FINANCE_DATA')
ERROR_DATA_PATH = os.getenv('ERROR_DATA_FINANCE')
model_key = os.getenv('key')

def tb_analysis():
    # Load data into dataframe
    df = pd.read_excel(finance_dataset)
    df.index.name = 'RowID'

    # Define pandera schema
    pa_schema = DataFrameSchema(
        {
            'Company_Code': Column(pa.Int),
            'Account_Type': Column(pa.String),
            'Starting_Balance': Column(pa.Int),
            'Debit_Balance': Column(pa.Float64),
            'Credit_Balance': Column(pa.Float64),
            'Ending_Balance': Column(pa.Float64)
        },
        checks=[
            Check(lambda d: d['Company_Code']>=1010, error='Company code cannot be less than 1010'),
            Check(lambda d: d['Company_Code'].notna().all(), error='Company code cannot be NULL'),
            Check(lambda d: d['Starting_Balance']==0, error='Starting balance cannot be less than or greater than 0.'),
            Check(lambda d: d['Starting_Balance'].notna().all(), error='Starting balance cannot be null'),
            Check(lambda d: ~((d['Account_Type']=='Asset') & (d['Credit_Balance']<0)), error='Asset accounts cannot have credit side balances.'),
            Check(lambda d: ~((d['Account_Type']=='Liability') & (d['Debit_Balance']>0)), error='Liability accounts cannot have debit side balances.'),
            Check(lambda d: ~((d['Account_Type']=='Revenue') & (d['Debit_Balance']>0)), error='Revenue accounts cannot have debit side balances.'),
            Check(lambda d: ~((d['Account_Type']=='Expense') & (d['Credit_Balance']<0)), error='Expense accounts cannot have credit side balances.'),
            Check(lambda d: ~((d['Account_Type']=='Equity') & (d['Debit_Balance']>0)), error='Equity accounts cannot have debit side balances.'),
            Check(lambda d: ~((d['Account_Type']=='Other System/Control Accounts') & (d['Ending_Balance']>0)), error='Other system/ control accounts should not have a balance.'),
            Check(lambda d: ~((d['Account_Type']=='Other System/Control Accounts') & (d['Ending_Balance']<0)), error='Other system/ control accounts should not have a balance.')
        ]
    )

    # Handle pandera exceptions - bad records
    try:
        pa_schema.validate(df, lazy=True)
        print('Data passed all checks')
    except pa.errors.SchemaErrors as err:
        print('Data failed validation checks!')
        err_df = err.failure_cases
        err_dff = pd.DataFrame()
        err_dff['Errors'] = err_df['check']
        err_dff['Failed_Index'] = err_df['index']
        err_dff.sort_values('Failed_Index', inplace=True)
        err_dff.drop_duplicates(subset=['Failed_Index'], inplace=True)
        merged_err_dff = pd.merge(df, err_dff, left_on='RowID', right_on='Failed_Index', how='left', validate='one_to_one')
        merged_err_dff.index.name = 'SNo.'
        merged_err_dff.to_excel(ERROR_DATA_PATH)

    # Generate summarized analysis of the errors
    count = merged_err_dff['Errors'].value_counts()

    return merged_err_dff, count


# Create datasets
payload_full, payload_errors = tb_analysis()

# Call model and print output
model = OpenAI(api_key=model_key)
response = model.chat.completions.create(
    model='chatgpt-4o-latest',
    messages=[
        {
            'role': 'system',
            'content': 'You are an experienced data analyst.'
         },
        {
            'role': 'user',
            'content': f"""
                            Analyze the data in {payload_errors} to generate key insights.
                       """
        }
    ],
    temperature=0.2,
    top_p=0.2,
    max_tokens=2048,
    stream=False
)
print(f'LLM output on financial data quality checks:{response.choices[0].message.content}')