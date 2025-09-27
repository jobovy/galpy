import sympy as sp
from sympy.utilities.codegen import codegen

# Symbols and expression
x, y = sp.symbols('x y')
expr = sp.sin(x) * sp.exp(y) + sp.sqrt(x**2 + y**2)

# Generate C code
[(c_name, c_code), (h_name, h_code)] = codegen(
    name_expr=("myfunc", expr),
    language="C",
    project="example",
    to_files=True
)

print(f"C file: {c_name}, header: {h_name}")
