# Guarda este código como convert_notebook.py
import json
import os

def convert_ipynb_to_py(ipynb_file, py_file):
    try:
        # Leer el archivo .ipynb
        with open(ipynb_file, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Abrir archivo .py para escribir
        with open(py_file, 'w', encoding='utf-8') as f:
            # Procesar cada celda
            for cell in notebook['cells']:
                # Si es una celda de código
                if cell['cell_type'] == 'code':
                    # Escribir el código
                    source = ''.join(cell['source'])
                    if source.strip():  # Evitar celdas vacías
                        f.write(source)
                        # Asegurar que hay un salto de línea al final
                        if not source.endswith('\n'):
                            f.write('\n')
                        f.write('\n')  # Línea en blanco entre celdas
                # Si es una celda markdown
                elif cell['cell_type'] == 'markdown':
                    # Escribir el markdown como comentario
                    source = ''.join(cell['source'])
                    if source.strip():  # Evitar celdas vacías
                        # Convertir cada línea a comentario
                        commented = '\n'.join(['# ' + line for line in source.split('\n')])
                        f.write(commented)
                        f.write('\n\n')  # Línea en blanco entre celdas
        
        print(f"Archivo convertido exitosamente: {py_file}")
        return True
    except Exception as e:
        print(f"Error al convertir el archivo: {str(e)}")
        return False

# Ruta de archivos
notebook_file = "Preprocess.ipynb"
python_file = "preprocess_convertido.py"

# Realizar la conversión
convert_ipynb_to_py(notebook_file, python_file)