import os 
import dotenv

dotenv.load_dotenv()



password= os.getenv('password')
username = os.getenv('pgusername')
port=os.getenv('port')
hostname=os.getenv('hostname')
NomicModel='nomic-embed-text'
qwen = 'qwen2.5-coder:3b'