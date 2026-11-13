"""Installe le bundle NiiVue figé et vérifie son empreinte."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'vendor_manifest.json').read_text())
    target=root/'vendor/niivue.es.js'
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==manifest['sha256']:
        print('NiiVue vérifié :',manifest['version']);return
    with urlopen(manifest['url'],timeout=60) as response:data=response.read()
    if hashlib.sha256(data).hexdigest()!=manifest['sha256']:raise ValueError('Empreinte NiiVue incorrecte')
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.tmp');temporary.write_bytes(data);temporary.replace(target)
    print('NiiVue installé :',manifest['version'])


if __name__=='__main__':main()
