import datetime
import csv
import os
class outputManagement():
    def __init__(self,emotion):
        self.array = []
        self.array_aux = {item.lower(): 0 for item in emotion}
        self.emotions = [item.lower() for item in emotion]
        self.modo_datos = 0
        self.i = -1
        self.cont = 0 
        self.frames = []
        self.documentName = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dataPath = "datas"

    def preparar_output(self,prediccion, predict,time =None):
        predict= predict.lower()
        if time ==None: 
            time = datetime.datetime.now()
        if len(prediccion) == 1:
            prediccion = prediccion[0]
        if self.modo_datos == 0:
            self.array.append({
                "emotion": predict,
                "time": time,
                **dict(zip(self.emotions, prediccion))
            })
        elif self.modo_datos == 1:
            aux = {
                "emotion": predict,
                "time":time,
                **dict(zip(self.emotion, prediccion))
            }
            if (self.i != -1 and self.array[self.i]["emotion"] != aux["emotion"]) or self.i == -1:
                self.array.append(aux)
                self.i += 1
        elif self.modo_datos == 2:
            if self.cont == 20:
                self.array_aux[predict] += 1
                emotions_count = list(zip(self.array_aux.keys(), self.array_aux.values()))
                moreFrecuent, max_count = max(emotions_count, key=lambda x: x[1])
                
                 # Construir diccionario compatible con `sacar_output()`
                fila_csv = {
                    "predict": f"{(max_count * 100) / self.cont}% en {self.cont} predicciones",
                    "emotion": moreFrecuent,
                    "time":time,
                }

                # check if the value is a valid emotion
                fila_csv.update({emotion: self.array_aux.get(emotion, 0) for emotion in self.array_aux})

                # save on a array the result
                self.array.append(fila_csv)


                self.array_aux = dict.fromkeys(self.array_aux.keys(), 0)
                self.cont = 0
            else:
                self.array_aux[predict] += 1
                self.cont += 1

    def sacar_output(self):
        if not os.path.exists(self.dataPath):
            os.makedirs(self.dataPath, exist_ok=True)

        docPath = os.path.join(self.dataPath, f"{self.documentName}.csv")
        fieldnames = ["emotion", "time"] + self.emotions  # Columnas del CSV
        
        # Calcular el ancho máximo de cada columna
        column_widths = {field: len(field) for field in fieldnames}  # Iniciar con la longitud del nombre de la columna
        
        for row in self.array:
            for field in fieldnames:
                if field in row:
                    column_widths[field] = max(column_widths[field], len(str(row[field])))

        with open(docPath, "w", newline="", encoding="utf-8", errors="ignore") as csvfile:
            writer = csv.writer(csvfile)
            
            # Escribir encabezado alineado
            header = [field.ljust(column_widths[field]) for field in fieldnames]
            writer.writerow(header)

            # Escribir datos alineados
            for row in self.array:
                clean_row = [str(row.get(field, "")).ljust(column_widths[field]) for field in fieldnames]
                writer.writerow(clean_row)


