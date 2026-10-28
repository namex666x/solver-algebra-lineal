import streamlit as st
import numpy as np
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Solver Lineal Pro | Álgebra Lineal",
    page_icon="📐",
    layout="wide"
)

# Estilos CSS limpios
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

st.title("📐 Calculadora Avanzada de Álgebra Lineal")
st.caption("Proyecto Final • Universidad Mariano Gálvez (UMG)")

# Barra lateral con modo de operación
with st.sidebar:
    st.header("⚙️ Modo de Operación")
    modo = st.radio(
        "Selecciona la herramienta:",
        ("Sistema de Ecuaciones [A | C]", "Matriz Inversa [A | I]")
    )
    st.markdown("---")
    st.info("Algoritmo completo con pivoteo parcial y análisis en NumPy.")

if modo == "Sistema de Ecuaciones [A | C]":
    st.subheader("📝 Sistema de Ecuaciones Lineales")
    m_eq = st.number_input("Número de Ecuaciones (filas m)", min_value=2, max_value=8, value=3, step=1)
    n_vars = st.number_input("Número de Variables (columnas n)", min_value=2, max_value=8, value=3, step=1)
    
    if 'm_sys' not in st.session_state or st.session_state['m_sys'] != m_eq or st.session_state['n_sys'] != n_vars:
        st.session_state['m_sys'] = m_eq
        st.session_state['n_sys'] = n_vars
        st.session_state['df_sys'] = pd.DataFrame(np.zeros((m_eq, n_vars + 1)), columns=[f"x_{j+1}" for j in range(n_vars)] + ["| C"])

    edited_df = st.data_editor(st.session_state['df_sys'], use_container_width=True)

    def gauss_jordan_sistema(matriz):
        mat = matriz.astype(float).copy()
        m, total_cols = mat.shape
        n = total_cols - 1
        steps = []
        lead = 0
        for r in range(m):
            if lead >= n: break
            i = r
            while abs(mat[i, lead]) < 1e-12:
                i += 1
                if i == m:
                    i = r
                    lead += 1
                    if lead == n: return mat, steps
            if i != r:
                mat[[r, i]] = mat[[i, r]]
                steps.append({"desc": f"Intercambiar Fila {r+1} con Fila {i+1}.", "matrix": mat.copy()})
            pivot = mat[r, lead]
            if abs(pivot) > 1e-12:
                mat[r] = mat[r] / pivot
                if abs(pivot - 1.0) > 1e-12:
                    steps.append({"desc": f"Normalizar Fila {r+1} dividiendo entre el pivote ({pivot:.4f}).", "matrix": mat.copy()})
            for i in range(m):
                if i != r:
                    factor = mat[i, lead]
                    if abs(factor) > 1e-12:
                        mat[i] = mat[i] - factor * mat[r]
                        steps.append({"desc": f"Eliminación en Fila {i+1} usando Fila {r+1}.", "matrix": mat.copy()})
            lead += 1
        return mat, steps

    if st.button("🚀 Resolver Sistema por Gauss-Jordan"):
        try:
            final_mat, steps = gauss_jordan_sistema(edited_df.to_numpy())
            st.markdown("---")
            st.subheader("📊 Resultados y Paso a Paso")
            with st.expander("🔍 Ver proceso iterativo detallado", expanded=True):
                for idx, step in enumerate(steps):
                    st.markdown(f"**Paso {idx+1}:** {step['desc']}")
                    st.dataframe(pd.DataFrame(step['matrix'], columns=edited_df.columns).style.format("{:.4f}"), use_container_width=True)
            
            rank_A = np.linalg.matrix_rank(final_mat[:, :-1], tol=1e-10)
            rank_aug = np.linalg.matrix_rank(final_mat, tol=1e-10)
            
            st.markdown("---")
            if rank_A < rank_aug:
                st.error("❌ Sistema Inconsistente (Sin Solución).")
            elif rank_A < (final_mat.shape[1] - 1):
                st.warning("⚠️ Sistema Compatible Indeterminado (Infinitas Soluciones).")
                st.dataframe(pd.DataFrame(final_mat, columns=edited_df.columns).style.format("{:.4f}"), use_container_width=True)
            else:
                st.success("✅ Sistema Compatible Determinado (Solución Única):")
                solutions = final_mat[:, -1]
                cols = st.columns(len(solutions))
                for j, val in enumerate(solutions):
                    with cols[j % len(cols)]:
                        st.metric(label=f"x_{j+1}", value=f"{val:.4f}")
        except Exception as e:
            st.error(f"Error en el cálculo: {str(e)}")

else:
    st.subheader("🔄 Cálculo de Matriz Inversa [A | I]")
    n_dim = st.number_input("Dimensión de la matriz cuadrada (n x n)", min_value=2, max_value=5, value=2, step=1)
    
    if 'n_inv' not in st.session_state or st.session_state['n_inv'] != n_dim:
        st.session_state['n_inv'] = n_dim
        st.session_state['df_inv'] = pd.DataFrame(np.zeros((n_dim, n_dim)), columns=[f"A_{j+1}" for j in range(n_dim)])

    st.write("Introduce los valores de la matriz cuadrada A:")
    matrix_A_df = st.data_editor(st.session_state['df_inv'], use_container_width=True)

    def gauss_jordan_inversa(A):
        n = A.shape[0]
        identity = np.eye(n)
        mat = np.hstack((A.astype(float), identity))
        steps = []
        
        for r in range(n):
            max_row = r + np.argmax(np.abs(mat[r:, r]))
            if max_row != r:
                mat[[r, max_row]] = mat[[max_row, r]]
                steps.append({"desc": f"Intercambiar Fila {r+1} con Fila {max_row+1}.", "matrix": mat.copy()})
            
            pivot = mat[r, r]
            if abs(pivot) < 1e-12:
                raise ValueError("La matriz no es invertible (determinante cero).")
            
            if abs(pivot - 1.0) > 1e-12:
                mat[r] = mat[r] / pivot
                steps.append({"desc": f"Normalizar Fila {r+1} dividiendo entre el pivote ({pivot:.4f}).", "matrix": mat.copy()})
                
            for i in range(n):
                if i != r:
                    factor = mat[i, r]
                    if abs(factor) > 1e-12:
                        mat[i] = mat[i] - factor * mat[r]
                        steps.append({"desc": f"Eliminación en Fila {i+1} usando Fila {r+1}.", "matrix": mat.copy()})
                        
        return mat, steps

    if st.button("🚀 Calcular Matriz Inversa"):
        try:
            A_input = matrix_A_df.to_numpy()
            det = np.linalg.det(A_input)
            
            if abs(det) < 1e-12:
                st.error("❌ **La matriz no es invertible:** Su determinante es cero (singular).")
            else:
                st.success(f"✅ **Matriz Invertible:** Determinante = {det:.4f}")
                final_augmented, steps_inv = gauss_jordan_inversa(A_input)
                
                st.markdown("---")
                st.subheader("📊 Paso a Paso de la Matriz Aumentada [A | I]")
                with st.expander("🔍 Ver iteraciones detalladas", expanded=True):
                    col_names = [f"A_{j+1}" for j in range(n_dim)] + [f"I_{j+1}" for j in range(n_dim)]
                    for idx, step in enumerate(steps_inv):
                        st.markdown(f"**Paso {idx+1}:** {step['desc']}")
                        st.dataframe(pd.DataFrame(step['matrix'], columns=col_names).style.format("{:.4f}"), use_container_width=True)
                
                inv_matrix = final_augmented[:, n_dim:]
                st.markdown("---")
                st.subheader("🎯 Resultado Final: Matriz Inversa A⁻¹")
                st.dataframe(pd.DataFrame(inv_matrix, columns=[f"x_{j+1}" for j in range(n_dim)]).style.format("{:.4f}"), use_container_width=True)
                st.code(str(np.round(inv_matrix, 4)))
                
        except Exception as e:
            st.error(f"Error en el cálculo de la inversa: {str(e)}")
