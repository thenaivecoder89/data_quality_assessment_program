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
technical_dataset = os.getenv('TECHNICAL_DATA')
events_dataset = os.getenv('EVENTS_DATA')
ERROR_DATA_PATH = os.getenv('ERROR_DATA_TECHNICAL')
ERROR_DATA_PATH_TECHNICAL_EVENTS = os.getenv('ERROR_DATA_PATH_TECHNICAL_EVENTS')
model_key = os.getenv('key')

# Load data
df_technical = pd.read_excel(technical_dataset)
df_technical.index.name = 'Row_ID'
df_technical_merge = df_technical[df_technical['Player_Event_Category_ID']!='DNP']
df_events = pd.read_excel(events_dataset)
df_events_merge = df_events[df_events['Event_Category_ID']!='DNP']
merged_technical_events_df = pd.merge(df_technical_merge, df_events_merge, left_on='Player_Event_Category_ID', right_on='Event_Category_ID', how='left', validate='many_to_many')
merged_technical_events_df = merged_technical_events_df[merged_technical_events_df['Player_Contract_Signing_Date']!='DNP']
merged_technical_events_df.index.name = 'SNo.'

# Perform data validations
def merged_data_validation():
    try:
        pa_schema_technical_events_merged = DataFrameSchema(
            checks=[
                Check(lambda d: ((d['Player_Contract_Signing_Date']<=d['Event_Start_Date']) & (d['Player_Contract_End_Date']>=d['Event_End_Date'])), error='Player contract dates does not fall within the event period.')
            ]
        )
        try:
            pa_schema_technical_events_merged.validate(merged_technical_events_df, lazy=True)
            print('Technical and events merged data passed validation checks!')
            return 'Technical and events merged data passed validation checks!'
        except pa.errors.SchemaErrors as err:
            print('Technical and evants merged data failed validation checks!')
            err_df = err.failure_cases
            err_dff_merge = pd.DataFrame()
            err_dff_merge['Errors'] = err_df['check']
            err_dff_merge['Failure_Index'] = err_df['index']
            merged_technical_event_errors = pd.merge(merged_technical_events_df, err_dff_merge, left_on='SNo.', right_on='Failure_Index', how='left', validate='many_to_many')
            merged_technical_event_errors.to_excel(ERROR_DATA_PATH_TECHNICAL_EVENTS)
            return err_dff_merge
    except Exception as e:
        error_message = f'Error encountered: {e}'
        return error_message

def technical_only_data_validation():
    try:
        pa_schema_technical = DataFrameSchema(
            checks=[
                Check(lambda d: ~(d['Player_Event_Category_ID'] == 'DNP'), error='Missing event category ID'),
                Check(lambda d: ~(d['Player_Gender'] == 'DNP'), error='Missing player gender'),
                Check(lambda d: ~(d['Player_Nationality'] == 'DNP'), error='Missing player nationality'),
                Check(lambda d: ~(d['Player_Emirates_ID'] == 'DNP'), error='Missing player Emirates ID'),
                Check(lambda d: ~(d['Player_Residence'] == 'DNP'), error='Missing player residence'),
                Check(lambda d: ~(d['Player_DOB'] == 'DNP'), error='Missing player date of birth'),
                Check(lambda d: ~(d['Player_Age'] == 'DNP'), error='Missing player age'),
                Check(lambda d: ~(d['Player_Classification_AR'] == 'DNP'), error='Missing player classification (in Arabic)'),
                Check(lambda d: ~(d['Player_Classification_EN'] == 'DNP'), error='Missing player classification (in English)'),
                Check(lambda d: ~(d['Player_Sport_AR'] == 'DNP'), error='Missing player sport (in Arabic)'),
                Check(lambda d: ~(d['Player_Sport_EN'] == 'DNP'), error='Missing player sport (in English)'),
                Check(lambda d: ~(d['Player_Team'] == 'DNP'), error='Missing player team'),
                Check(lambda d: ~(d['Player_Height_cm'] == 'DNP'), error='Missing player height'),
                Check(lambda d: ~(d['Player_Weight_kg'] == 'DNP'), error='Missing player weight'),
                Check(lambda d: ~(d['Player_BMI'] == 'DNP'), error='Missing player BMI'),
                Check(lambda d: ~(d['Player_Social_Status'] == 'DNP'), error='Missing player social status'),
                Check(lambda d: ~(d['Player_Position'] == 'DNP'), error='Missing player position'),
                Check(lambda d: ~(d['Player_Contract_Signing_Date'] == 'DNP'), error='Missing player contract sign date'),
                Check(lambda d: ~(d['Player_Contract_End_Date'] == 'DNP'), error='Missing player contract end date'),
                Check(lambda d: ~(d['Player_Contract_Duration (in years)'] == 'DNP'), error='Missing player contract duration'),
                Check(lambda d: ~(d['Player_in_National_Squad'] == 'DNP'), error='Missing player in national squad'),
                Check(lambda d: ~(d['Player_Notes_AR'] == 'DNP'), error='Missing player notes (in Arabic)'),
                Check(lambda d: ~(d['Player_Notes_EN'] == 'DNP'), error='Missing player notes (in English)')
            ]
        )
        try:
            pa_schema_technical.validate(df_technical, lazy=True)
            print('Technical only data passed validation checks!')
        except pa.errors.SchemaErrors as err:
            print('Technical only data failed validation checks!')
            err_df = err.failure_cases
            err_dff = pd.DataFrame()
            err_dff['Errors'] = err_df['check']
            err_dff['Failure_Index'] = err_df['index']
            merged_technical_errors = pd.merge(df_technical, err_dff, left_on='Row_ID', right_on='Failure_Index', how='left', validate='many_to_many')
            merged_technical_errors.to_excel(ERROR_DATA_PATH)
            return err_dff
    except Exception as e:
        error_message = f'Error encountered: {e}'
        return error_message

failed_merge_df = merged_data_validation()
failed_df = technical_only_data_validation()
if failed_merge_df == 'Technical and events merged data passed validation checks!':
    final_failed_df = failed_df.groupby('Errors', as_index=False).agg(Error_Count=('Failure_Index', 'nunique'))
    print(final_failed_df)
    X_values = final_failed_df['Errors'].tolist()
    Y_values = final_failed_df['Error_Count'].tolist()
else:
    final_failed_merge_df = failed_merge_df.groupby('Errors', as_index=False).agg(Error_Count=('Failure_Index', 'nunique'))
    final_failed_df = failed_df.groupby('Errors', as_index=False).agg(Error_Count=('Failure_Index', 'nunique'))
    print(f'Failed technical and events check:\n{final_failed_merge_df}')
    print(f'Failed technical only check:\n{final_failed_df}')
    X_values = final_failed_df['Errors'].tolist()
    X_values.extend(final_failed_merge_df['Errors'].tolist())
    Y_values = final_failed_df['Error_Count'].tolist()
    Y_values.extend(final_failed_merge_df['Error_Count'].tolist())

fig, ax = plt.subplots(figsize=(25, 10), constrained_layout=True, dpi=120)
bars = ax.bar(X_values, Y_values)
ax.bar_label(bars, fontsize=8, padding=2)
ax.tick_params(axis='x', labelsize=6)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
    tick.set_ha('right')
    tick.set_rotation_mode('anchor')

plt.tight_layout()
plt.show()

# Call LLM and generate output
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
                        Analyze the data in {final_failed_df} and {failed_merge_df} and generate insights.
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