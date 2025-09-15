import os
import shutil
import atexit

UPLOAD_FOLDER = 'uploads'

def clear_uploads():
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Greška pri brisanju: {e}")
                
def register_cleanup():
    atexit.register(clear_uploads)