import os
from openai import OpenAI
os.environ['DISABLE_PANDERA_IMPORT_WARNING'] = 'True'
from dotenv import load_dotenv
import pandera.pandas as pa
from pandera import Check, DataFrameSchema, Column
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', '{:,.2f}'.format)

# Initialize environment
load_dotenv()
procurement_dataset = os.getenv('PROCUREMENT_DATA')
ERROR_DATA_PATH = os.getenv('ERROR_DATA_PROCUREMENT')
model_key = os.getenv('key')

# Calculating vendor payment delays
df = pd.read_excel(procurement_dataset)
df['Payment_Delays'] = df['Payment Date'] - df['Invoice Date']
df_vendor_payments = df[['Purchase_Requisition', 'Purchase_order', 'Name 1', 'Payment_Delays', 'PayT', 'Payment year']].drop_duplicates()
df_vendor_payments['Payment_Delays'] = df_vendor_payments['Payment_Delays'].dt.days
df_vendor_payments['PO_Payment_Term'] = df_vendor_payments['PayT'].str[-2:].fillna(0).astype(int)
df_vendor_payments.index.name = 'Row_ID'

# Data quality assessment
def missing_values():
    try:
        pa_schema = DataFrameSchema(
            checks=[
                Check(lambda d: ~(d['PayT'].isna()), error='Payment terms of a purchase order cannot be blank.'),
                Check(lambda d: ~(d['Purchase_Requisition'].isna()), error='Purchase orders should have a corresponding purchase requisition.'),
                Check(lambda d: d['Payment_Delays']!=d['PO_Payment_Term'], error='Payments made are not compliant with PO payment terms.')
            ]
        )
        try:
            pa_schema.validate(df_vendor_payments, lazy=True)
            print('Data passed validation checks!')
        except pa.errors.SchemaErrors as err:
            print('Data failed validation checks!')
            err_df = err.failure_cases
            err_dff = pd.DataFrame()
            err_dff['Errors'] = err_df['check']
            err_dff['Failure_Index'] = err_df['index']
            err_dff.drop_duplicates(subset='Failure_Index', inplace=True)
            merged_errors = pd.merge(df_vendor_payments, err_dff, left_on='Row_ID', right_on='Failure_Index', how='left', validate='one_to_one')
            merged_errors.index.name = 'SNo.'
            merged_errors.to_excel(ERROR_DATA_PATH)
            return err_dff
    except Exception as e:
        error_message = f'Error encountered: {e}'
        return error_message

# Generate Chart of Detected Errors
final_df = missing_values()
X_values = final_df['Errors'].drop_duplicates().tolist()
Y_values = final_df['Errors'].value_counts().tolist()
X_values_f = ['PO Missing Payment Terms', 'PO Missing Requisition', 'Payment Date and Term Mis-match']
model_input = pd.DataFrame()
model_input['Errors'] = X_values
model_input['Count_of_Errors'] = Y_values
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(X_values_f, Y_values)
ax.bar_label(bars, fontsize=8, padding=2)
ax.tick_params(axis='x', labelsize=8)
for tick in ax.get_xticklabels():
    tick.set_rotation(45)

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
            'content':f"""
                        Analyze the data in {model_input} to generate key insights.
                        """
        }
    ],
    temperature= 0.2,
    top_p= 0.2,
    max_tokens=2048,
    stream= False
)
output = response.choices[0].message.content
print(f'Output of data validation for procurement:\n{output}')