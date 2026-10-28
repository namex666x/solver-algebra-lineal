import streamlit as st
import numpy as np
import pandas as pd

# Configuración inicial de la página
st.set_page_config(
    page_title="Solver Lineal Pro | Álgebra Lineal",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de CSS Personalizado para estética corporativa y moderna
st.markdown("""
<style>
    /* Estilos globales y tipografía */
    .main {
        background-color: #f8fafc;
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }
    
    /* Ocultar elementos predeterminados de Streamlit para look nativo */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Tarjetas contenedoras elegantes */
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }
    
    .step-card {
        background: #ffffff;
        border-left: 5px solid #3b82f6;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }

    /* Encabezados personalizados */
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 700;
    }
    
    .subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Botones profesionales */
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Título principal de la aplicación
st.markdown("<h1>Calculadora Avanzada de Sistemas Lineales</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Proyecto Final de Álgebra Lineal • Método de Gauss-Jordan con Análisis Numérico</p>", unsafe_allow_html=True)

# Barra lateral para configuración dinámica
with st.sidebar:
    st.markdown("### ⚙️ Configuración del Sistema")
    st.markdown("Define las dimensiones de tu sistema de ecuaciones lineales $A \\cdot X = C$.")
    
    m_eq = st.number_input("Número de Ecuaciones (filas $m$)", min_value=2, max_value=8, value=3, step=1)
    n_vars = st.number_input("Número de Variables (columnas $n$)", min_value=2, max_value=8, value=3, step=1)
    
    st.markdown("---")
    st.markdown("### ℹ️ Acerca de")
    st.markdown("Esta aplicación implementa el algoritmo de **Eliminación de Gauss-Jordan con Pivoteo Parcial** utilizando **NumPy** para garantizar estabilidad numérica y detección exacta de tipos de solución.")

# Contenedor principal de entrada de datos
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📝 Entrada de la Matriz Aumentada $[A \\ | \\ C]$")
st.markdown(f"Introduce los coeficientes del sistema de {m_eq} ecuaciones y {n_vars} incógnitas.")

# Inicializar estado para la matriz si cambian las dimensiones
if 'matrix_df' not in st.session_state or st.session_state.get('m') != m_eq or st.session_state.get('n') != n_vars:
    st.session_state['m'] = m_eq
    st.session_state['n'] = n_vars
    # Matriz por defecto: coeficientes en ceros y término independiente en ceros
    default_data = np.zeros((m_eq, n_vars + 1))
    col_names = [f"x_{j+1}" for j in range(n_vars)] + ["| C (Término Indep.)"]
    st.session_state['matrix_df'] = pd.DataFrame(default_data, columns=col_names)

# Editor de datos interactivo tipo hoja de cálculo
edited_df = st.data_editor(st.session_state['matrix_df'], use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Algoritmo de Gauss-Jordan con Pivoteo Parcial y Registro de Pasos
def gauss_jordan_con_pasos(augmented_matrix):
    mat = augmented_matrix.astype(float).copy()
    m, total_cols = mat.shape
    n = total_cols - 1
    steps = []
    
    lead = 0
    for r in range(m):
        if lead >= n:
            break
        
        # 1. Pivoteo parcial: buscar la fila con el mayor valor absoluto en la columna 'lead'
        i = r
        while abs(mat[i, lead]) < 1e-12:
            i += 1
            if i == m:
                i = r
                lead += 1
                if lead == n:
                    return mat, steps
        
        # Intercambiar filas si es necesario
        if i != r:
            mat[[r, i]] = mat[[i, r]]
            steps.append({
                "desc": f"Intercambio de Fila {r+1} con Fila {i+1} para pivoteo óptimo.",
                "matrix": mat.copy()
            })
        
        # 2. Normalizar la fila del pivote
        pivot = mat[r, lead]
        if abs(pivot) > 1e-12:
            mat[r] = mat[r] / pivot
            if abs(pivot - 1.0) > 1e-12:
                steps.append({
                    "desc": f"Normalizar Fila {r+1} dividiendo entre el pivote ({pivot:.4f}).",
                    "matrix": mat.copy()
                })
        
        # 3. Hacer ceros en las demás filas de la columna actual
        for i in range(m):
            if i != r:
                factor = mat[i, lead]
                if abs(factor) > 1e-12:
                    mat[i] = mat[i] - factor * mat[r]
                    steps.append({
                        "desc": f"Eliminación: Fila {i+1} $\\leftarrow$ Fila {i+1} - ({factor:.4f}) $\\times$ Fila {r+1}.",
                        "matrix": mat.copy()
                    })
        
        lead += 1
        
    return mat, steps

# Botón de ejecución
if st.button("🚀 Resolver Sistema por Gauss-Jordan", use_container_width=True):
    raw_matrix = edited_df.to_numpy()
    
    try:
        final_mat, steps = gauss_jordan_con_pasos(raw_matrix)
        
        st.markdown("---")
        st.markdown("## 📊 Resultados y Paso a Paso")
        
        # Mostrar pasos detallados en tarjetas
        with st.expander("🔍 Ver el proceso iterativo detallado paso a paso", expanded=True):
            if not steps:
                st.info("La matriz ya se encuentra en su forma escalonada reducida.")
            for idx, step in enumerate(steps):
                st.markdown(f"""
                <div class='step-card'>
                    <strong>Paso {idx+1}:</strong> {step['desc']}
                </div>
                """, unsafe_allow_html=True)
                
                # Formatear la matriz a 4 decimales para visualización limpia
                step_df = pd.DataFrame(step['matrix'], columns=edited_df.columns)
                st.dataframe(step_df.style.format("{:.4f}"), use_container_width=True)
        
        # Análisis del tipo de solución (Teorema de Rouché-Frobenius / Rango)
        m_rows, total_cols = final_mat.shape
        n_vars_count = total_cols - 1
        
        # Extraer matriz de coeficientes A y vector C
        A_final = final_mat[:, :n_vars_count]
        C_final = final_mat[:, n_vars_count]
        
        # Calcular rangos para determinar consistencia
        # Consideramos elementos muy cercanos a 0 como cero por errores numéricos
        rank_A = np.linalg.matrix_rank(A_final, tol=1e-10)
        augmented_full = final_mat
        rank_aug = np.linalg.matrix_rank(augmented_full, tol=1e-10)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Conclusión y Solución del Sistema")
        
        if rank_A < rank_aug:
            st.error("❌ **Sistema Inconsistente (Sin Solución):** El sistema presenta una contradicción (ej. una fila de ceros igual a un número distinto de cero).")
        elif rank_A < n_vars_count:
            st.warning(f"⚠️ **Sistema Compatible Indeterminado (Infinitas Soluciones):** El rango de la matriz ({rank_A}) es menor que el número de variables ({n_vars_count}). El sistema posee variables libres.")
            
            # Mostrar matriz en forma escalonada reducida final
            st.markdown("**Matriz en Forma Escalonada Reducida:**")
            st.dataframe(pd.DataFrame(final_mat, columns=edited_df.columns).style.format("{:.4f}"), use_container_width=True)
        else:
            st.success("✅ **Sistema Compatible Determinado (Solución Única Encontrada):**")
            
            # Extraer los resultados de la última columna
            solutions = final_mat[:, -1]
            
            # Mostrar soluciones en columnas estéticas
            cols = st.columns(min(n_vars_count, 4))
            for j in range(n_vars_count):
                with cols[j % len(cols)]:
                    st.metric(label=f"Variable x_{j+1}", value=f"{solutions[j]:.4f}")
            
            st.markdown("**Vector Solución Exacto ($X$):**")
            st.code(str(np.round(solutions, 4)))
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Ocurrió un error durante el cálculo numérico: {str(e)}")
