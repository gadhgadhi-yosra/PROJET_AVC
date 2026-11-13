"""Démo locale Gradio : IRM, segmentation fournie ou modèle raccordé, rapport."""
import json
import os
from pathlib import Path
import tempfile
import gradio as gr
from core import load_volume, aligned, predict, summarize, plot_slices, ood_distance


def run(t1, supplied_mask, uncertainty, atlas, brain, position):
    if not t1: raise gr.Error('Chargez une IRM NIfTI.')
    try:
        image, x = load_volume(t1)
        provenance = 'Masque fourni — aucune inférence exécutée'
        if not supplied_mask:
            adapter = os.environ.get('AVC_PREDICT_ADAPTER')
            if not adapter: raise ValueError('Modèle non configuré : chargez un masque existant ou branchez le modèle de l’équipe.')
            result = predict(t1, adapter)
            supplied_mask, uncertainty = result['mask_path'], result.get('uncertainty_path')
            provenance = f'Prédiction par {adapter}'
        mask = aligned(supplied_mask, image, binary=True)
        unc = aligned(uncertainty, image) if uncertainty else None
        regions = aligned(atlas, image) if atlas else None
        labels_path = os.environ.get('AVC_ATLAS_LABELS')
        labels = json.loads(Path(labels_path).read_text()) if labels_path else None
        report = summarize(image, mask, unc, regions, labels)
        report['provenance'] = provenance
        report['ood'] = {'status':'Référence et masque cérébral prétraité requis', 'alert':None}
        reference = os.environ.get('AVC_REFERENCE_HISTOGRAMS')
        if brain and reference: report['ood'] = ood_distance(image, x, brain, reference)
        output = Path(tempfile.mkdtemp(prefix='avc-report-'))/'rapport.json'
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return plot_slices(image,x,mask,position), report, str(output)
    except (ValueError, OSError, KeyError) as error:
        raise gr.Error(str(error)) from error


def build_app():
    with gr.Blocks(title='AVC · Domain shift') as app:
        gr.Markdown('# AVC · Explorer une segmentation\nDémo de recherche de Yosra. Chargez une IRM et un masque, ou utilisez le modèle configuré par l’équipe.')
        with gr.Row():
            with gr.Column(scale=1):
                t1 = gr.File(label='IRM T1 (.nii / .nii.gz)', type='filepath')
                mask = gr.File(label='Masque lésionnel existant (facultatif si modèle connecté)', type='filepath')
                with gr.Accordion('Incertitude, atlas et distance de domaine', open=False):
                    uncertainty = gr.File(label='Carte d’incertitude', type='filepath')
                    atlas = gr.File(label='Atlas déjà recalé sur cette IRM', type='filepath')
                    brain = gr.File(label='Masque cérébral du prétraitement (OOD)', type='filepath')
                position = gr.Slider(0,1,value=.5,label='Position relative des coupes')
                button = gr.Button('Analyser', variant='primary')
            with gr.Column(scale=2):
                figure = gr.Plot(label='Coupes et lésion')
                report = gr.JSON(label='Rapport de recherche')
                download = gr.File(label='Télécharger le rapport JSON')
        button.click(run, [t1,mask,uncertainty,atlas,brain,position], [figure,report,download])
        gr.Markdown('**[Ouvrir la vue 3D NiiVue](/viewer/)** · Les régions exigent un atlas recalé et vérifié. Une distance élevée seule ne prouve pas un échec de segmentation.')
    return app


def create_server():
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    root=Path(__file__).resolve().parent
    server=FastAPI()
    @server.get('/viewer',include_in_schema=False)
    def redirect_viewer():return RedirectResponse('/viewer/')
    @server.get('/viewer/',include_in_schema=False)
    def viewer():return FileResponse(root/'viewer.html')
    server.mount('/viewer/vendor',StaticFiles(directory=root/'vendor'),name='niivue_assets')
    return gr.mount_gradio_app(server,build_app(),path='/',run_history=False)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(create_server(),host=os.environ.get('GRADIO_SERVER_NAME','127.0.0.1'),
                port=int(os.environ.get('GRADIO_SERVER_PORT','7860')))
