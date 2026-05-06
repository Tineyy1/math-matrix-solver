import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp

st.set_page_config(page_title="Math & Matrix Solver", layout="wide")
st.title("Numerical Methods & Matrix Operations")

tab1, tab2 = st.tabs(["Root Finding Methods", "Matrix Operations"])

# ==========================================
# TAB 1: ROOT FINDING
# ==========================================
with tab1:
    st.header("Find Roots of an Equation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        eq_str = st.text_input("Enter equation (in terms of x)", value="x**3 - x - 2")
        method = st.selectbox("Select Method", 
                              ["Incremental Search", "Bisection Method", "False Position (Regula-Falsi)", 
                               "Newton-Raphson", "Secant Method"])
        
        if method in ["Bisection Method", "False Position (Regula-Falsi)", "Incremental Search"]:
            xl = st.number_input("Lower Bound (xl)", value=1.0)
            xu = st.number_input("Upper Bound (xu)", value=2.0)
        elif method == "Newton-Raphson":
            x0 = st.number_input("Initial Guess (x0)", value=1.0)
        elif method == "Secant Method":
            x0 = st.number_input("First Guess (x0)", value=1.0)
            x1 = st.number_input("Second Guess (x1)", value=2.0)
            
        tol = st.number_input("Tolerance", value=0.0001, format="%.5f")
        max_iter = st.number_input("Max Iterations", value=50, step=1)
        solve_btn = st.button("Solve Root")

    with col2:
        if solve_btn:
            try:
                # Setup symbolic math and functions
                x = sp.Symbol('x')
                expr = sp.sympify(eq_str)
                f = sp.lambdify(x, expr, 'numpy')
                df_expr = sp.diff(expr, x)
                df = sp.lambdify(x, df_expr, 'numpy')

                results = []
                root = None
                
                # Algorithms
                if method == "Bisection Method":
                    for i in range(max_iter):
                        xr = (xl + xu) / 2
                        err = abs(xu - xl) / 2
                        results.append({"Iter": i+1, "xl": xl, "xu": xu, "xr": xr, "f(xr)": f(xr), "Error": err})
                        if f(xr) == 0 or err < tol:
                            root = xr; break
                        if f(xl) * f(xr) < 0: xu = xr
                        else: xl = xr
                        root = xr

                elif method == "False Position (Regula-Falsi)":
                    for i in range(max_iter):
                        xr = xu - (f(xu)*(xl - xu)) / (f(xl) - f(xu))
                        err = abs(f(xr))
                        results.append({"Iter": i+1, "xl": xl, "xu": xu, "xr": xr, "f(xr)": f(xr), "Error": err})
                        if err < tol:
                            root = xr; break
                        if f(xl) * f(xr) < 0: xu = xr
                        else: xl = xr
                        root = xr

                elif method == "Newton-Raphson":
                    xr = x0
                    for i in range(max_iter):
                        fxr = f(xr)
                        dfxr = df(xr)
                        xr_new = xr - fxr/dfxr
                        err = abs(xr_new - xr)
                        results.append({"Iter": i+1, "xi": xr, "f(xi)": fxr, "f'(xi)": dfxr, "xi+1": xr_new, "Error": err})
                        xr = xr_new
                        if err < tol:
                            root = xr; break

                elif method == "Secant Method":
                    for i in range(max_iter):
                        fx1, fx0 = f(x1), f(x0)
                        x2 = x1 - (fx1 * (x0 - x1)) / (fx0 - fx1)
                        err = abs(x2 - x1)
                        results.append({"Iter": i+1, "x(i-1)": x0, "x(i)": x1, "x(i+1)": x2, "f(x(i+1))": f(x2), "Error": err})
                        x0, x1 = x1, x2
                        if err < tol:
                            root = x2; break
                
                elif method == "Incremental Search":
                    step = 0.1
                    curr_x = xl
                    for i in range(max_iter):
                        next_x = curr_x + step
                        results.append({"Iter": i+1, "x": curr_x, "f(x)": f(curr_x)})
                        if f(curr_x) * f(next_x) < 0:
                            root = (curr_x + next_x)/2
                            results.append({"Iter": i+2, "x": next_x, "f(x)": f(next_x)})
                            break
                        curr_x = next_x

                # Output
                st.success(f"Estimated Root: **{root}**")
                
                st.subheader("Iteration Table")
                st.dataframe(pd.DataFrame(results), use_container_width=True)

                st.subheader("Graph")
                fig, ax = plt.subplots()
                x_vals = np.linspace(root - 5, root + 5, 400)
                y_vals = f(x_vals)
                ax.plot(x_vals, y_vals, label=f"f(x) = {eq_str}")
                ax.axhline(0, color='black', linewidth=1)
                ax.axvline(0, color='black', linewidth=1)
                ax.scatter([root], [0], color='red', zorder=5, label=f"Root ~ {root:.4f}")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error evaluating equation: {e}")

# ==========================================
# TAB 2: MATRIX OPERATIONS
# ==========================================
with tab2:
    st.header("Matrix Operations")
    
    def parse_matrix(matrix_str):
        try:
            return np.matrix(matrix_str.replace('\n', ';'))
        except:
            return None

    op = st.selectbox("Select Operation", 
                      ["Addition", "Multiplication", "Adjoint", "Inverse", 
                       "Determinant", "Power of Matrix", "Transpose", "System of Equations (Ax = B)"])

    colA, colB = st.columns(2)
    with colA:
        st.write("Format: Space separated columns, new line for rows.")
        st.write("Example:\n1 2\n3 4")
        matA_str = st.text_area("Matrix A", "1 2\n3 4")
        A = parse_matrix(matA_str)

    if op in ["Addition", "Multiplication", "System of Equations (Ax = B)"]:
        with colB:
            if op == "System of Equations (Ax = B)":
                matB_str = st.text_area("Matrix B (Results Column)", "5\n11")
            else:
                matB_str = st.text_area("Matrix B", "5 6\n7 8")
            B = parse_matrix(matB_str)
            
    if op == "Power of Matrix":
        with colB:
            power = st.number_input("Power (n)", value=2, step=1)

    if st.button("Calculate Matrix"):
        if A is None:
            st.error("Invalid Matrix A")
        else:
            try:
                st.subheader("Result:")
                if op == "Addition":
                    st.write(A + B)
                elif op == "Multiplication":
                    st.write(A * B)
                elif op == "Transpose":
                    st.write(A.T)
                elif op == "Determinant":
                    st.write(np.linalg.det(A))
                elif op == "Inverse":
                    st.write(np.linalg.inv(A))
                elif op == "Adjoint":
                    # Adjoint is Inverse * Determinant
                    inv = np.linalg.inv(A)
                    det = np.linalg.det(A)
                    st.write(np.round(inv * det))
                elif op == "Power of Matrix":
                    st.write(np.linalg.matrix_power(A, power))
                elif op == "System of Equations (Ax = B)":
                    x = np.linalg.solve(A, B)
                    st.write("Solutions (x):")
                    st.write(x)
            except np.linalg.LinAlgError as e:
                st.error(f"Linear Algebra Error: {e}")
            except Exception as e:
                st.error(f"Error: {e}")