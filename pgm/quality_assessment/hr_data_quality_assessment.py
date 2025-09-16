import os
os.environ['DISABLE_PANDERA_IMPORT_WARNING']='True'
import pandera.pandas as pa
from pandera import DataFrameSchema, Check
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:,.2f}'.format)

# Initialize environment
load_dotenv()
hr_dataset = os.getenv('HR_DATA')
ERROR_DATA_PATH = os.getenv('ERROR_DATA_HR')
model_key = os.getenv('key')

# Load Data
df = pd.read_excel(hr_dataset)
df['Calculated_Gross_Salary'] = df['Basic Salary'] + df['Housing Allowance'] + df['Social Allowance'] + df['Living Allowance'] + df['Transportation Allowance'] + df['Supplementary Allowance'] + df['Kids Allowance'] + df['Ticket Allowance'] + df['Telephone Allowance'] + df['Other Allowance'] + df['Over Time'] + df['Supervisory Allowance'] + df['Language Allowance'] + df['Union Allowance'] + df['Placement Allowance'] + df['Gratification'] + df['Deputation Allowance'] + df['Monthly Reward'] + df['Consolidate Allowance'] + df['Acting Allowance'] + df['Match Fee'] + df['Basic Salary Part Timer']
df_dataset = df[['EMPL_ID', 'Company Description', 'Gross Salary', 'Calculated_Gross_Salary', 'Joining Date', 'Date of Birith', 'National Expiry Date', 'Emirates Id', 'Email Address', 'Phone Number', 'bank (bankName)', 'iban']]
df_dataset.index.name = 'Row_ID'
df_empl_bank_dataset = df[['EMPL_ID', 'iban']].groupby('EMPL_ID', as_index=False).agg(iban_count=('iban', 'nunique'))

# Data quality assessments
def data_quality_assessment():
    try:
        pa_schema_1 = DataFrameSchema(
            checks=[
                Check(lambda d: d['Calculated_Gross_Salary'] == d['Gross Salary'], error='Calculated gross salary does not match with system gross salary.'),
                Check(lambda d: ~(d['Emirates Id'].isna()), error='Emirates ID cannot be empty.'),
                Check(lambda d: ~(d['Phone Number'].isna()), error='Phone number should not be empty.'),
                Check(lambda d: ~(d['Email Address'].isna()), error='Email address should not be empty.'),
                Check(lambda d: ~(d['bank (bankName)'].isna()), error='Bank name should not be empty.'),
                Check(lambda d: ~(d['iban'].isna()), error='IBAN should not be empty.'),
                Check(lambda d: d['Date of Birith'].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}$'), error='Date of birth not in YYYY-MM-DD format.'),
                Check(lambda d: d['Joining Date'].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}$'), error='Joining date not in YYYY-MM-DD format.'),
                Check(lambda d: d['National Expiry Date'].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}$'), error='Emirates ID expiry date not in YYYY-MM-DD format.')
            ]
        )
        pa_schema_2 = DataFrameSchema(
            checks=[
                Check(lambda d: ~(d['iban_count']>1), error='One employee cannot have more than one account.')
            ]
        )
        try:
            pa_schema_1.validate(df_dataset, lazy=True)
            pa_schema_2.validate(df_empl_bank_dataset, lazy=True)
            print('Data passed validation checks!')
        except pa.errors.SchemaErrors as err:
            print('Data failed validation checks!')
            err_df = err.failure_cases
            err_dff = pd.DataFrame()
            err_dff['Errors'] = err_df['check']
            err_dff['Failure_Index'] = err_df['index']
            merged_dataset = pd.merge(df_dataset, err_dff, left_on='Row_ID', right_on='Failure_Index', how='left', validate='many_to_many')
            merged_dataset.to_excel(ERROR_DATA_PATH)
            return err_dff
    except Exception as e:
        error_message = f'Error encountered: {e}'
        return error_message

# Generate unique errors dataset
failed_data = data_quality_assessment()
unique_errors_count = failed_data[['Errors', 'Failure_Index']].groupby('Errors', as_index=False).agg(Error_Count=('Failure_Index', 'nunique'))

# Display chart
X_values = unique_errors_count['Errors'].tolist()
X_values_f = ['Empty Bank Name', 'Calculated vs System Gross Salary Mismatch', 'Date Format Incorrect - DOB', 'Empty Email Address', 'Empty Emirates ID', 'Date Format Incorrect - EID Expiry', 'Empty IBAN', 'Date Format Incorrect - Joining Date', 'Empty Phone Number']
Y_values = unique_errors_count['Error_Count'].tolist()
print(X_values_f)
print(Y_values)
fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True, dpi=120)
bars = ax.bar(X_values_f, Y_values)
ax.bar_label(bars, fontsize=8, padding=2)
ax.tick_params(axis='x', labelsize=6)
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
    tick.set_ha('right')
    tick.set_rotation_mode('anchor')

plt.tight_layout()
plt.show()

# Call LLM and Generate Output
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
                        Analyze the data in {unique_errors_count} to generate key insights.
                        """
        }
    ],
    max_tokens=2048,
    temperature=0.2,
    top_p=0.2,
    stream=False
)
output = response.choices[0].message.content
print(f'Output of data validation for HR:\n{output}')