import os
import sys

def check_project_structure():
    """Verificar la estructura del proyecto y los permisos"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    required_dirs = [
        'src',
        'static',
        'static/css',
        'static/js',
        'templates',
        'uploads',
        'instance'
    ]
    
    required_files = [
        'templates/index.html',
        'static/css/styles.css',
        'static/js/voiceProcessing.js',
        'src/app.py'
    ]
    
    print("Verificando estructura del proyecto...")
    
    # Verificar directorios
    for dir_path in required_dirs:
        full_path = os.path.join(base_dir, dir_path)
        if not os.path.exists(full_path):
            print(f"ERROR: Directorio faltante: {dir_path}")
            os.makedirs(full_path)
            print(f"Creado directorio: {dir_path}")
        if not os.access(full_path, os.W_OK):
            print(f"ERROR: Sin permisos de escritura en: {dir_path}")
    
    # Verificar archivos
    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        if not os.path.exists(full_path):
            print(f"ERROR: Archivo faltante: {file_path}")
        elif not os.access(full_path, os.R_OK):
            print(f"ERROR: Sin permisos de lectura en: {file_path}")
    
    print("Verificación completa.")

if __name__ == "__main__":
    check_project_structure()