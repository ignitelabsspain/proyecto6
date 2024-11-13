import streamlit as st
import joblib
import numpy as np

# Título de la aplicación
st.title("Predictor")
st.header("Predecir el valor de venta del vehiculo")

# Instrucciones para el usuario
st.write("Ingrese las características del coche para predecir el precio.")

# Front
year   = int(st.text_input("Año de Fabricación", value="2000"))
km     = int(st.text_input("km", value="20000"))
#fuel   = int(st.slider('Tipo de combustible (0/1)',0 ,1 ,1))
#fuel = st.selectbox("Seleccione el tipo de combustible:", ("gasolina", "diesel"))
options = ["gasolina", "diesel"]

# Selector para el tipo de combustible
fuel = st.selectbox("Seleccione el tipo de combustible:", options)

# Índice de la opción seleccionada
fuel_index = options.index(fuel)
fuel = int(fuel_index)
# Muestra el valor seleccionado
#st.write("El combustible seleccionado es:", fuel)

engine = int(st.text_input("Tamaño del motor", value="2000"))

# Backend
# Creamos el array de entrada
X_list =    [ year, km, fuel, engine ]
X = np.array(X_list, dtype=np.float64)
X = X.reshape(1,-1)

# Botón para ejecutar el modelo
if st.button("Predecir"):
    if len(X) > 0:
        # Cargar el modelo y los parámetros de normalización guardados
        scaler = joblib.load('scaler.joblib')
        model = joblib.load('model.joblib')
        
        #scaler_y = joblib.load('scaler_y.joblib')
        # Mostrar las primeras filas del DataFrame cargado
        X_scaled = scaler.transform(X)
        # Realizar predicciones con el modelo
        y = model.predict(X_scaled)
        
        # Normalizar la salida
        #y = scaler_y.transform(y)
        
        # Mostrar las predicciones
        st.write("Predicciones del precio de venta:")
        st.write(y)
        