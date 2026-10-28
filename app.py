import streamlit as st
import numpy as np
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Solver Lineal Pro | Álgebra Lineal",
    page_icon="📐",
    layout="wide"
)

# Estilos CSS robustos para evitar conflictos con extensiones de modo oscuro
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.title("📐 Calculadora Avanzada de Sistemas Lineales")
st.caption("Proyecto Final de Álgebra Lineal • Método de Gauss-Jordan con Análisis Numérico")

# Barra lateral para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    m_eq = st.number_input("Número de Ecuaciones (filas m)", min_value=2, max_value=8, value=3, step=1)
    n_vars = st.number_input("Número de Variables (columnas n)", min_value=2, max_value=8, value=3, step=1)
    
    st.markdown("---")
    st.info("Algoritmo de Eliminación de Gauss-Jordan con Pivoteo Parcial y NumPy.")

# Sección de entrada de datos
st.subheader(" Matriz Aumentada [A | C]")
st.write(f"Introduce los coeficientes del sistema de {m_eq} ecuaciones y {n_vars} incógnitas.")

# Inicializar estado de la matriz
if 'm' not in st.session_state or st.session_state['m'] != m_eq or st.session_state['n'] != n_vars:
    st.session_state['m'] = m_eq
    st.session_state['n'] = n_vars
    default_data = np.zeros((m_eq, n_vars + 1))
    col_names = [f"x_{j+1}" for j in range(n_vars)] + ["| C (Término Indep.)"]
    st.session_state['matrix_df'] = pd.DataFrame(default_data, columns=col_names)

edited_df = st.data_editor(st.session_state['matrix_df'], use_container_width=True)

# Algoritmo de Gauss-Jordan con Pivoteo Parcial
def gauss_jordan_con_pasos(augmented_matrix):
    mat = augmented_matrix.astype(float).copy()
    m, total_cols = mat.shape
    n = total_cols - 1
    steps = []
    
    lead = 0
    for r in range(m):
        if lead >= n:
            break
        
        i = r
        while abs(mat[i, lead]) < 1e-12:
            i += 1
            if i == m:
                i = r
                lead += 1
                if lead == n:
                    return mat, steps
        
        if i != r:
            mat[[r, i]] = mat[[i, r]]
            steps.append({
                "desc": f"Intercambio de Fila {r+1} con Fila {i+1}.",
                "matrix": mat.copy()
            })
        
        pivot = mat[r, lead]
        if abs(pivot) > 1e-12:
            mat[r] = mat[r] / pivot
            if abs(pivot - 1.0) > 1e-12:
                steps.append({
                    "desc": f"Normalizar Fila {r+1} dividiendo entre el pivote ({pivot:.4f}).",
                    "matrix": mat.copy()
                })
        
        for i in range(m):
            if i != r:
                factor = mat[i, lead]
                if abs(factor) > 1e-12:
                    mat[i] = mat[i] - factor * mat[r]
                    steps.append({
                        "desc": f"Eliminación en Fila {i+1} usando Fila {r+1}.",
                        "matrix": mat.copy()
                    })
        lead += 1
        
    return mat, steps

# Botón de ejecución
if st.button(" Resolver Sistema por Gauss-Jordan"):
    raw_matrix = edited_df.to_numpy()
    
    try:
        final_mat, steps = gauss_jordan_con_pasos(raw_matrix)
        
        st.markdown("---")
        st.subheader(" Resultados y Paso a Paso")
        
        with st.expander(" Ver proceso iterativo detallado", expanded=True):
            if not steps:
                st.info("La matriz ya se encuentra en su forma escalonada reducida.")
            for idx, step in enumerate(steps):
                st.markdown(f"**Paso {idx+1}:** {step['desc']}")
                step_df = pd.DataFrame(step['matrix'], columns=edited_df.columns)
                st.dataframe(step_df.style.format("{:.4f}"), use_container_width=True)
        
        m_rows, total_cols = final_mat.shape
        n_vars_count = total_cols - 1
        A_final = final_mat[:, :n_vars_count]
        
        rank_A = np.linalg.matrix_rank(A_final, tol=1e-10)
        rank_aug = np.linalg.matrix_rank(final_mat, tol=1e-10)
        
        st.markdown("---")
        st.subheader(" Conclusión del Sistema")
        
        if rank_A < rank_aug:
            st.error(" **Sistema Inconsistente (Sin Solución):** El sistema presenta contradicciones matemáticas.")
        elif rank_A < n_vars_count:
            st.warning(" **Sistema Compatible Indeterminado (Infinitas Soluciones):** El sistema posee variables libres.")
            st.dataframe(pd.DataFrame(final_mat, columns=edited_df.columns).style.format("{:.4f}"), use_container_width=True)
        else:
            st.success(" **Sistema Compatible Determinado (Solución Única):**")
            solutions = final_mat[:, -1]
            
            cols = st.columns(min(n_vars_count, 4))
            for j in range(n_vars_count):
                with cols[j % len(cols)]:
                    st.metric(label=f"Variable x_{j+1}", value=f"{solutions[j]:.4f}")
            
            st.code(str(np.round(solutions, 4)))
            
    except Exception as e:
        st.error(f"Ocurrió un error durante el cálculo: {str(e)}")
