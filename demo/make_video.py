"""Vidéo MP4 de secours à partir des captures du test navigateur réel."""
from pathlib import Path
import subprocess
import tempfile
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont


def main():
    root=Path(__file__).resolve().parents[1];images=root/'docs/yosra/demo_verification'
    shots=[('01_interface.png','01 · Démo locale Gradio'),
           ('02_rapport_synthetique.png','02 · Coupes et rapport : volume synthétique 2,091 mL'),
           ('03_niivue_coupes.png','03 · Visualisation multiplanaire NiiVue'),
           ('04_niivue_3d.png','04 · Rendu 3D NiiVue')]
    output=root/'docs/yosra/demo_secours.mp4'
    try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',26)
    except OSError:font=ImageFont.load_default()
    with tempfile.TemporaryDirectory(prefix='avc-video-') as tmp:
        for index,(name,title) in enumerate(shots):
            source=Image.open(images/name).convert('RGB');source.thumbnail((1400,780))
            frame=Image.new('RGB',(1440,900),'#102827');frame.paste(source,((1440-source.width)//2,75))
            draw=ImageDraw.Draw(frame);draw.text((30,20),title,fill='white',font=font)
            draw.text((30,855),'PHANTOM SYNTHÉTIQUE · masque fourni · aucune inférence de modèle',fill='#ffd29d',font=font)
            frame.save(Path(tmp)/f'{index:02}.png')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-framerate','1/5','-i',str(Path(tmp)/'%02d.png'),
            '-c:v','libx264','-pix_fmt','yuv420p','-r','25','-movflags','+faststart',str(output)],check=True)
    print(output)


if __name__=='__main__':main()
