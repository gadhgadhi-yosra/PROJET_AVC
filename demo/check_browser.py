"""Test réel du navigateur et captures de secours sur un phantom synthétique."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'docs/yosra/demo_verification';output.mkdir(exist_ok=True)
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
            headless=True,args=['--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto('http://127.0.0.1:7860/',wait_until='networkidle',timeout=120000)
        page.screenshot(path=str(output/'01_interface.png'),full_page=True)
        inputs=page.locator('input[type=file]')
        inputs.nth(1).set_input_files(str(ROOT/'data/demo/synthetic/mask.nii.gz'))
        inputs.nth(0).set_input_files(str(ROOT/'data/demo/synthetic/t1.nii.gz'))
        page.get_by_role('button',name='Analyser',exact=True).click()
        page.wait_for_function("document.body.innerText.includes('2.091')",timeout=120000)
        page.screenshot(path=str(output/'02_rapport_synthetique.png'),full_page=True)
        page.goto('http://127.0.0.1:7860/viewer/',wait_until='networkidle',timeout=120000)
        page.wait_for_function("document.getElementById('status').textContent.startsWith('Prêt')",timeout=120000)
        page.locator('#image').set_input_files(str(ROOT/'data/demo/synthetic/t1.nii.gz'))
        page.locator('#mask').set_input_files(str(ROOT/'data/demo/synthetic/mask.nii.gz'))
        page.locator('#load').click()
        page.wait_for_function("document.getElementById('status').textContent.startsWith('Chargé')",timeout=120000)
        page.screenshot(path=str(output/'03_niivue_coupes.png'),full_page=True)
        page.locator('#mode').click();page.wait_for_timeout(1500)
        page.screenshot(path=str(output/'04_niivue_3d.png'),full_page=True)
        report=dict(browser='Chrome headless / WebGL SwiftShader',gradio_report_ml=2.091,
                    niivue_loaded=True,render_3d_clicked=True,page_errors=errors,
                    scope='Phantom synthétique et masque fourni ; aucune inférence de modèle')
        (output/'browser_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
        browser.close()
        if errors:raise RuntimeError(errors)
        print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
