import os
os.environ['DISABLE_PANDERA_IMPORT_WARNING'] = 'True'
from dotenv import load_dotenv
import pandera.pandas as pa
from pandera import Check, DataFrameSchema
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:,.2f}'.format)

# Initialize environment
load_dotenv()
events_dataset = os.getenv('EVENTS_DATA')
ERROR_DATA_PATH = os.getenv('ERROR_DATA_EVENTS')
model_key = os.getenv('key')

# Load data
df = pd.read_excel(events_dataset)
df.index.name = 'Row_ID'

# Perform data validations
def data_validation():
    try:
        pa_schema = DataFrameSchema(
            checks=[
                Check(lambda d: ~(d['Event_Name'] == 'DNP'), error='Missing event name'),
                Check(lambda d: ~(d['Event_Category'] == 'DNP'), error='Missing event category'),
                Check(lambda d: ~(d['Event_Type'] == 'DNP'), error='Missing event type'),
                Check(lambda d: ~(d['Event_Venue'] == 'DNP'), error='Missing event venue'),
                Check(lambda d: ~(d['Event_Age_Group'] == 'DNP'), error='Missing event age group'),
                Check(lambda d: ~(d['Emiratis_Winner'] == 'DNP'), error='Missing event Emirati winners'),
                Check(lambda d: ~((d['Event_Start_Date'].isna()) & (d['Event_End_Date'].isna())), error='Missing event start and end dates'),
                Check(lambda d: ~((d['Gold_Medals'] + d['Silver_Medals'] + d['Bronze_Medals'] + d['No_Medals']) == 0), error='Missing medal information'),
                Check(lambda d: ~((d['Gold_Medals']=='DNP') & (d['Silver_Medals']=='DNP') & (d['Bronze_Medals']=='DNP') & (d['No_Medals']=='DNP')), error='Missing medal information')
            ]
        )
        try:
            pa_schema.validate(df, lazy=True)
            print('Data passed validation checks!')
        except pa.errors.SchemaErrors as err:
            print('Data failed validation checks!')
            err_df = err.failure_cases
            err_dff = pd.DataFrame()
            err_dff['Errors'] = err_df['check']
            err_dff['Failure_Index'] = err_df['index']
            merged_df = pd.merge(df, err_dff, left_on='Row_ID', right_on='Failure_Index', how='left', validate='many_to_many')
            merged_df.index.name = 'SNo.'
            merged_df.to_excel(ERROR_DATA_PATH)
            return err_dff
    except Exception as e:
        error_message = f'Error encountered: {e}'
        return error_message

# Identifying unique records and developing plot
failed_df = data_validation()
final_failed_df = failed_df.groupby('Errors', as_index=False).agg(error_count=('Failure_Index', 'nunique'))
final_failed_df.index.name = 'SNo.'
X_values = final_failed_df['Errors'].tolist()
Y_values = final_failed_df['error_count'].tolist()
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(X_values, Y_values)
ax.bar_label(bars, fontsize=8, padding=2)
ax.tick_params(axis='x', labelsize=8)
plt.tight_layout()
plt.show()

# Calling LLM and generating output
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
                        Analyze the data in {final_failed_df} and generate insights.
                        """
        }
    ],
    temperature=0.2,
    top_p=0.2,
    max_tokens=2048,
    stream=False
)
output = response.choices[0].message.content
print(f'Output of data validation for events data:\n{output}')