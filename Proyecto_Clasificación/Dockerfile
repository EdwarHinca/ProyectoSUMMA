# Imagen base con Python 3.11
FROM python:3.13

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos del proyecto al contenedor
COPY requirements.txt ./
COPY main.py ./
COPY app ./app
COPY model ./model

# Instalar las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto en el que correrá FastAPI
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
