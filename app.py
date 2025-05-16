import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from app.home import show

if __name__ == "__main__":
    show()
