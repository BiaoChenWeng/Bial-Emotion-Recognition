

import pandas as pd
import sys
name = sys.argv[1]

df = pd.read_csv(name)
df.columns = df.columns.str.strip()


df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S:%f')

df['emotion_change'] = df['emocion'] != df['emocion'].shift()

# Crear una columna para el "grupo" que indica las secuencias consecutivas de la misma emoción
df['group'] = df['emotion_change'].cumsum()
df = df[df["emocion"]!= "neutral "]
# Calcular la duración por grupo (restando el tiempo de inicio del tiempo final)
df_duration = df.groupby('group').agg(
    emocion=('emocion', 'first'),
    duracion=('time', lambda x: x.iloc[-1] - x.iloc[0]),
    time = ("time",'first')
).reset_index()
print(df_duration)
print(df)
df_duration['time'] = df_duration['time'].dt.strftime('%H:%M:%S:%f') #ponerlo en formato deseado a time


#dropea group que se usó para juntar los consecutivos
df_duration = df_duration.drop(columns=['group'])
#quitar las emociones que no se mantiene medio segundo o 15 frame
df_duration.to_csv('notNeutralData.csv', index=False)
df_duration = df_duration[df_duration['duracion'].apply(lambda x: pd.to_timedelta(x).total_seconds() >= 0.5)]

# Guardar el resultado en un nuevo archivo CSV
print(df_duration)
df_duration.to_csv('cleanData.csv', index=False)

df_duration.to_excel('cleanData.xlsx', index=False, engine='openpyxl')



