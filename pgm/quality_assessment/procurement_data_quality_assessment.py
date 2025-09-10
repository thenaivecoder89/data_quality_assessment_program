import os
os.environ['DISABLE_PANDERA_IMPORT_WARNING'] = 'True'
from dotenv import load_dotenv
import pandera.pandas as pa
from pandera import Check, DataFrameSchema, Column
import openai
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', '{,:.2f}'.format)

# Initialize environment
load_dotenv()
procurement_dataset = os.getenv('PROCUREMENT_DATA')
ERROR_DATA_PATH = os.getenv('ERROR_DATA_PROCUREMENT')
model_key = os.getenv('key')

df = pd.read_excel(procurement_dataset)
print(df)