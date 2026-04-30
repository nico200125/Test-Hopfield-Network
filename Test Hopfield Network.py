import streamlit as st
import numpy as np

# --- LÓGICA PURA DE HOPIFIELD (HEBB) ---

def calcWeights(fundmems_list):
    if not fundmems_list:
        return np.zeros((120, 120))
    
    # Regla de Hebb original: Sumatoria de (v * v.T)
    X = np.array(fundmems_list) # Matriz de M x 120
    T = X.T @ X
    
    # Tii = 0 (Diagonal a cero para estabilidad)
    np.fill_diagonal(T, 0)
    return T

def updateState(state, T, fundmems_dict):
    current_state = np.array(state, dtype=float).copy()
    
    # Máximo 100 iteraciones para evitar bucles infinitos en Hebb
    for _ in range(100):
        # Actualización síncrona: signo de (T @ estado)
        new_state = np.sign(T @ current_state)
        new_state[new_state == 0] = 1 # Forzar binario
        
        # Verificar si es una memoria conocida
        for name, pattern in fundmems_dict.items():
            if np.array_equal(new_state, pattern) or np.array_equal(new_state, -pattern):
                return new_state, name
        
        # Verificar convergencia estable
        if np.array_equal(new_state, current_state):
            break
        current_state = new_state.copy()
        
    return current_state, "Desconocido"

def generar_vector(scores):
    vec = -1 * np.ones(120)
    for j in range(12):
        limit = int(scores[j])
        for k in range(limit):
            vec[k + (j * 10)] = 1
    return vec

# --- INTERFAZ STREAMLIT ---

st.title("Red de Hopfield")

# Inicializar estados de sesión si no existen
if 'fundmems' not in st.session_state:
    st.session_state.fundmems = {}
if 'pesos_T' not in st.session_state:
    st.session_state.pesos_T = np.zeros((120, 120))

tab1, tab2 = st.tabs(["Configurar Memorias", "Probar Red"])

with tab1:
    st.header("Registro de Memorias Fundamentales")
    
    with st.form("form_memoria"):
        nombre = st.text_input("Nombre de la habitación")
        st.write("Define los 12 valores (1-10):")
        cols = st.columns(6)
        vals = []
        for i in range(12):
            with cols[i % 6]:
                vals.append(st.number_input(f"Cat {i+1}", 1, 10, 5, key=f"ins_{i}"))
        
        submit = st.form_submit_button("Guardar Memoria")
        
        if submit and nombre:
            nuevo_vector = generar_vector(vals)
            st.session_state.fundmems[nombre] = nuevo_vector
            # Recalcular pesos con la Regla de Hebb cada vez que se añade una
            st.session_state.pesos_T = calcWeights(list(st.session_state.fundmems.values()))
            st.success(f"Memoria '{nombre}' guardada y matriz T actualizada.")

    if st.session_state.fundmems:
        st.write("### Memorias actuales:", ", ".join(st.session_state.fundmems.keys()))
        if st.button("Borrar todas las memorias"):
            st.session_state.fundmems = {}
            st.session_state.pesos_T = np.zeros((120, 120))
            st.rerun()

with tab2:
    st.header("Prueba de Reconocimiento")
    
    if not st.session_state.fundmems:
        st.info("Primero debes guardar al menos una memoria en la pestaña anterior.")
    else:
        st.write("Ajusta los valores para probar la red:")
        test_vals = []
        c = st.columns(6)
        for i in range(12):
            with c[i % 6]:
                test_vals.append(st.slider(f"C{i+1}", 1, 10, 1, key=f"test_{i}"))
        
        if st.button("Ejecutar Red de Hopfield"):
            vec_test = generar_vector(test_vals)
            res_vec, res_nombre = updateState(vec_test, st.session_state.pesos_T, st.session_state.fundmems)
            
            if res_nombre != "Desconocido":
                st.balloons()
                st.success(f"¡Identificado como: **{res_nombre}**!")
            else:
                st.error("La red no pudo identificar el patrón (Mínimo local desconocido).")
            
            # Visualización rápida de los 120 bits
            st.write("Visualización de los 120 bits resultantes:")
            imagen_visual = ((res_vec.reshape(12, 10) + 1) * 127.5).astype(np.uint8)

            st.image(imagen_visual, width=400)
