from huggingface_hub import hf_hub_download
import zipfile

path = hf_hub_download(repo_id='smartywu/big-lama', filename='big-lama.zip')
print('Downloaded:', path)
with zipfile.ZipFile(path) as z:
    for name in z.namelist():
        print(' ', name)
