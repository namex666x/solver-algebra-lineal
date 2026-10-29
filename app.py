import streamlit as st

st.set_page_config(
    page_title="Solver Lineal Pro | UMG",
    page_icon="📐",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

st.title("📐 Calculadora Avanzada de Álgebra Lineal")
st.caption("Proyecto Final • Universidad Mariano Gálvez (UMG)")

st.markdown("---")
st.markdown("""
### ¡Bienvenido al Panel de Herramientas Numéricas!
Esta aplicación web ha sido diseñada para resolver problemas avanzados de álgebra lineal utilizando **NumPy** y algoritmos numéricos con pivoteo parcial.

Utiliza el **menú lateral izquierdo** (en la sección de páginas) para navegar entre las diferentes herramientas:
1. **📝 Sistema de Ecuaciones [A | C]:** Resuelve sistemas de $m$ ecuaciones y $n$ incógnitas paso a paso, detectando soluciones únicas, infinitas o sistemas incompatibles.
2. **🔄 Matriz Inversa [A | I]:** Calcula la inversa de matrices cuadradas mediante la matriz aumentada y validación de determinantes.
""")

st.info(" **Consejo:** Haz clic en cualquiera de las opciones del menú de la izquierda para comenzar a calcular.")
