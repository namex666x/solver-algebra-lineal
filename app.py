import streamlit as st
import numpy as np
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Solver Lineal Pro | Álgebra Lineal",
    page_icon="👑",
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

st.title("Calculadora Avanzada de Álgebra Lineal")
st.caption("Proyecto Final • Universidad Mariano Gálvez (UMG)")#------------

# Función para formatear números de forma limpia
def format_val(val):
    if abs(val - round(val)) < 1e-4:
        return str(int(round(val)))
    return f"{val:.2f}"


def format_df(df_input, col_names):
    formatted_data = np.vectorize(format_val)(df_input)
    return pd.DataFrame(formatted_data, columns=col_names)


# Comprobación del sistema de ecuaciones
def comprobar_sistema(matriz_original, soluciones):
    A = matriz_original[:, :-1]
    C = matriz_original[:, -1]

    resultado = A @ soluciones

    correcto = np.allclose(
        resultado, C, atol=1e-8, rtol=1e-8
    )

    return resultado, C, correcto


# Comprobación de la matriz inversa
def comprobar_inversa(A, A_inv):
    resultado = A @ A_inv
    identidad = np.eye(A.shape[0])

    correcto = np.allclose(
        resultado, identidad, atol=1e-8, rtol=1e-8
    )

    return resultado, identidad, correcto


# Barra lateral con modo de operación
with st.sidebar:
    st.header(" Modo de Operación")
    modo = st.radio(
        "Selecciona la herramienta:",
        ("Sistema de Ecuaciones [A | C]", "Matriz Inversa [A | I]")
    )
    st.markdown("---")
    st.info("Algoritmo completo con pivoteo parcial y análisis en NumPy.")#----------

if modo == "Sistema de Ecuaciones [A | C]":
    st.subheader(" Sistema de Ecuaciones Lineales")
    m_eq = st.number_input("Número de Ecuaciones (filas m)", min_value=2, max_value=8, value=3, step=1)
    n_vars = st.number_input("Número de Variables (columnas n)", min_value=2, max_value=8, value=3, step=1)

    if 'm_sys' not in st.session_state or st.session_state['m_sys'] != m_eq or st.session_state['n_sys'] != n_vars:
        st.session_state['m_sys'] = m_eq
        st.session_state['n_sys'] = n_vars
        st.session_state['df_sys'] = pd.DataFrame(np.zeros((m_eq, n_vars + 1)), columns=[f"x_{j+1}" for j in range(n_vars)] + ["| C"])

    edited_df = st.data_editor(st.session_state['df_sys'], use_container_width=True)#------

    def gauss_jordan_sistema(matriz):
        mat = matriz.astype(float).copy()
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
                    "desc": f"Intercambiar Fila {r+1} con Fila {i+1}.",
                    "matrix": mat.copy()
                })

            pivot = mat[r, lead]

            if abs(pivot) > 1e-12:
                mat[r] = mat[r] / pivot
                if abs(pivot - 1.0) > 1e-12:
                    steps.append({
                        "desc": f"Normalizar Fila {r+1} dividiendo entre el pivote ({pivot:.2f}).",
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

        return mat, steps#------------------

    if st.button(" Resolver Sistema por Gauss-Jordan"):
        try:
            matriz_original = edited_df.to_numpy(dtype=float).copy()

            final_mat, steps = gauss_jordan_sistema(matriz_original)

            st.markdown("---")
            st.subheader(" Resultados y Paso a Paso")

            with st.expander(" Ver proceso iterativo detallado", expanded=True):
                for idx, step in enumerate(steps):
                    st.markdown(f"**Paso {idx+1}:** {step['desc']}")
                    st.dataframe(
                        format_df(step['matrix'], edited_df.columns),
                        use_container_width=True
                    )

            rank_A = np.linalg.matrix_rank(final_mat[:, :-1], tol=1e-10)
            rank_aug = np.linalg.matrix_rank(final_mat, tol=1e-10)

            st.markdown("---")

            if rank_A < rank_aug:
                st.error(" Sistema Inconsistente (Sin Solución).")

            elif rank_A < (final_mat.shape[1] - 1):
                st.warning(" Sistema Compatible Indeterminado (Infinitas Soluciones).")
                st.dataframe(
                    format_df(final_mat, edited_df.columns),
                    use_container_width=True
                )

            else:
                st.success(" Sistema Compatible Determinado (Solución Única):")

                # Obtener exactamente las n incógnitas
                # considerando las posiciones de los pivotes.
                soluciones = np.zeros(n_vars)

                for fila in final_mat:
                    coeficientes = fila[:-1]
                    posiciones = np.where(np.abs(coeficientes) > 1e-10)[0]

                    if len(posiciones) > 0:
                        columna = posiciones[0]
                        soluciones[columna] = fila[-1] / fila[columna]

                cols = st.columns(len(soluciones))

                for j, val in enumerate(soluciones):
                    with cols[j]:
                        st.metric(
                            label=f"x_{j+1}",
                            value=format_val(val)
                        )

                # ------------------------------------------
                # COMPROBACIÓN DE GAUSS-JORDAN
                # ------------------------------------------

                st.markdown("---")
                st.subheader(" Comprobación de Gauss-Jordan")

                st.write(
                    "Sustituimos los valores encontrados en las "
                    "ecuaciones originales para verificar que A × X = C."
                )

                resultado, C, correcto = comprobar_sistema(
                    matriz_original, soluciones
                )

                tabla_comprobacion = pd.DataFrame({
                    "Ecuación": [f"Ecuación {i+1}" for i in range(m_eq)],
                    "Resultado A × X": [format_val(v) for v in resultado],
                    "Valor original C": [format_val(v) for v in C],
                    "Estado": [
                        "Correcto" if np.isclose(resultado[i], C[i], atol=1e-8, rtol=1e-8)
                        else "Incorrecto"
                        for i in range(m_eq)
                    ]
                })

                st.dataframe(
                    tabla_comprobacion,
                    use_container_width=True,
                    hide_index=True
                )

                with st.expander(" Ver sustitución de cada ecuación"):
                    for i in range(m_eq):
                        coeficientes = matriz_original[i, :-1]

                        operacion = " + ".join(
                            f"({format_val(coeficientes[j])} × {format_val(soluciones[j])})"
                            for j in range(n_vars)
                        )

                        st.markdown(f"**Ecuación {i+1}:**")
                        st.write(operacion)
                        st.write(
                            f"Resultado: {format_val(resultado[i])} "
                            f"= {format_val(C[i])}"
                        )
                        st.markdown("---")

                if correcto:
                    st.success(
                        " Comprobación correcta: todas las ecuaciones "
                        "se cumplen con los valores encontrados."
                    )
                else:
                    st.error(
                        " La comprobación no coincide con el sistema original."
                    )

        except Exception as e:
            st.error(f"Error en el cálculo: {str(e)}")#---------------------


else:
    st.subheader("Cálculo de Matriz Inversa [A | I]")
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
                steps.append({
                    "desc": f"Intercambiar Fila {r+1} con Fila {max_row+1}.",
                    "matrix": mat.copy()
                })

            pivot = mat[r, r]

            if abs(pivot) < 1e-12:
                raise ValueError("La matriz no es invertible (determinante cero).")

            if abs(pivot - 1.0) > 1e-12:
                mat[r] = mat[r] / pivot
                steps.append({
                    "desc": f"Normalizar Fila {r+1} dividiendo entre el pivote ({pivot:.2f}).",
                    "matrix": mat.copy()
                })

            for i in range(n):
                if i != r:
                    factor = mat[i, r]
                    if abs(factor) > 1e-12:
                        mat[i] = mat[i] - factor * mat[r]
                        steps.append({
                            "desc": f"Eliminación en Fila {i+1} usando Fila {r+1}.",
                            "matrix": mat.copy()
                        })

        return mat, steps

    if st.button(" Calcular Matriz Inversa"):
        try:
            A_input = matrix_A_df.to_numpy(dtype=float)
            det = np.linalg.det(A_input)

            if abs(det) < 1e-12:
                st.error(" **La matriz no es invertible:** Su determinante es cero (singular).")

            else:
                st.success(f" **Matriz Invertible:** Determinante = {format_val(det)}")

                final_augmented, steps_inv = gauss_jordan_inversa(A_input)

                st.markdown("---")
                st.subheader(" Paso a Paso de la Matriz Aumentada [A | I]")

                with st.expander(" Ver iteraciones detalladas", expanded=True):
                    col_names = [f"A_{j+1}" for j in range(n_dim)] + [f"I_{j+1}" for j in range(n_dim)]

                    for idx, step in enumerate(steps_inv):
                        st.markdown(f"**Paso {idx+1}:** {step['desc']}")
                        st.dataframe(
                            format_df(step['matrix'], col_names),
                            use_container_width=True
                        )

                inv_matrix = final_augmented[:, n_dim:]

                st.markdown("---")
                st.subheader("Resultado Final: Matriz Inversa A⁻¹")

                inv_col_names = [f"x_{j+1}" for j in range(n_dim)]
                st.dataframe(
                    format_df(inv_matrix, inv_col_names),
                    use_container_width=True
                )

                # ------------------------------------------
                # COMPROBACIÓN DE LA MATRIZ INVERSA
                # ------------------------------------------

                st.markdown("---")
                st.subheader(" Comprobación de la Matriz Inversa")

                st.write(
                    "Multiplicamos la matriz original A por la matriz "
                    "inversa A⁻¹. El resultado debe ser la matriz identidad."
                )

                resultado, identidad, correcto = comprobar_inversa(
                    A_input, inv_matrix
                )

                st.markdown("**Matriz original A:**")
                st.dataframe(
                    format_df(A_input, [f"A_{j+1}" for j in range(n_dim)]),
                    use_container_width=True
                )

                st.markdown("**Matriz inversa A⁻¹:**")
                st.dataframe(
                    format_df(inv_matrix, [f"x_{j+1}" for j in range(n_dim)]),
                    use_container_width=True
                )

                st.markdown("**Resultado de A × A⁻¹:**")
                st.dataframe(
                    format_df(resultado, [f"C_{j+1}" for j in range(n_dim)]),
                    use_container_width=True
                )

                st.markdown("**Matriz identidad esperada:**")
                st.dataframe(
                    format_df(identidad, [f"I_{j+1}" for j in range(n_dim)]),
                    use_container_width=True
                )

                with st.expander(" Ver operaciones de multiplicación matricial"):
                    for i in range(n_dim):
                        for j in range(n_dim):

                            operacion = " + ".join(
                                f"({format_val(A_input[i, k])} × {format_val(inv_matrix[k, j])})"
                                for k in range(n_dim)
                            )

                            st.markdown(
                                f"**Posición ({i+1}, {j+1}):**"
                            )

                            st.write(operacion)

                            st.write(
                                f"Resultado: {format_val(resultado[i, j])}"
                            )

                            st.markdown("---")

                if correcto:
                    st.success(
                        " Comprobación correcta: A × A⁻¹ = I. "
                        "La matriz inversa calculada es válida."
                    )
                else:
                    st.error(
                        " La multiplicación no produjo la matriz identidad."
                    )

        except Exception as e:
            st.error(f"Error en el cálculo de la inversa: {str(e)}")
