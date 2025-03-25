import datetime
import csv
import cv2
import os
class outputManagement():
    def __init__(self,emocion):
        self.array = []
        self.array_aux = {item.lower(): 0 for item in emocion}
        self.emociones = [item.lower() for item in emocion]
        self.modo_datos = 0
        self.i = -1
        self.cont = 0 
        self.frames = []
        self.documentName = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dataPath = "datas"

    def preparar_output(self,prediccion, predict):
        predict= predict.lower()
        if len(prediccion) == 1:
            prediccion = prediccion[0]
        if self.modo_datos == 0:
            self.array.append({
                "emocion": predict,
                "hora": datetime.datetime.now(),
                **dict(zip(self.emociones, prediccion))
            })
        elif self.modo_datos == 1:
            aux = {
                "emocion": predict,
                "hora": datetime.datetime.now(),
                **dict(zip(self.emociones, prediccion))
            }
            if (self.i != -1 and self.array[self.i]["emocion"] != aux["emocion"]) or self.i == -1:
                self.array.append(aux)
                self.i += 1
        elif self.modo_datos == 2:
            if self.cont == 20:
                self.array_aux[predict] += 1
                emociones_conteo = list(zip(self.array_aux.keys(), self.array_aux.values()))
                emocion_mas_frecuente, max_conteo = max(emociones_conteo, key=lambda x: x[1])
                
                 # Construir diccionario compatible con `sacar_output()`
                fila_csv = {
                    "predict": f"{(max_conteo * 100) / self.cont}% en {self.cont} predicciones",
                    "emocion": emocion_mas_frecuente,
                    "hora": datetime.datetime.now(),
                }

                # Asegurar que todas las emociones están presentes en el diccionario con valores por defecto (0.0)
                fila_csv.update({emocion: self.array_aux.get(emocion, 0) for emocion in self.array_aux})

                # Guardar en el array final
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
        fieldnames = ["emocion", "hora"] + self.emociones  # Columnas del CSV
        
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

    
    def saveFrame(self,frame):
        self.frames.append(frame.copy()) 

    def generateVideo(self):
        if len(self.frames) == 0 : 
            return 
        
        h, w , _ = self.frames[0].shape
        videoPath = os.path.join(self.dataPath,"{}.mp4".format(self.documentName))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
        fps = 30
        vidWriter = cv2.VideoWriter(videoPath,fourcc,fps,(w,h))
        for frame in self.frames:
            vidWriter.write(frame)
        vidWriter.release()