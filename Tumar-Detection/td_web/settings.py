import os
from base64 import b64encode

random_bytes = os.urandom(64)
SECRET_KEY = b64encode(random_bytes).decode('utf-8')
DEBUG = True
