import pandas as pd 
from tqdm import tqdm

def check_df(df):
  errores = []
  total = df.shape[0]
  for i in tqdm(range(total),desc="Checking dataframe"):
    row = df.iloc[i]
    source_path = row['source']
    target_path = row['target']
    id = row['id_class']
    try:
      source = Image.open(source_path)
      target = Image.open(target_path)
    except:
      print("\nError encontrado en fila {}".format(i))
      errores.append(row)
  if(len(errores)==0):
    print("No se encontró ningún error!")
  return errores
