"""Génère huit diapositives éditables à partir des livrables locaux."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BG='F6F5F0';INK='123B37';ACCENT='B84A2B'


def text(slide, value, x, y, w, h, size=22, color=INK):
    box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    frame=box.text_frame;frame.word_wrap=True
    for i,line in enumerate(value.split('\n')):
        p=frame.paragraphs[0] if i==0 else frame.add_paragraph()
        p.text=line;p.font.size=Pt(size);p.font.color.rgb=RGBColor.from_string(color)
        p.font.name='Aptos';p.space_after=Pt(15)
    return box


def main():
    deck=Presentation();deck.slide_width=Inches(13.333);deck.slide_height=Inches(7.5)
    slides=[
        ('Préparer, mesurer, expliquer', 'Segmentation des lésions d’AVC sur IRM T1\nDonnées et analyses · données, Spark, domain shift et démonstration\nÉtat local au 12 novembre 2026',None,
         'Mon rôle est de fournir des données traçables et de mesurer le décalage entre centres. Les modèles et l’adaptation sont les contributions de Fayfa et Anas.'),
        ('Un dataset contrôlé et versionné', '1 453 paires · 55 centres\n1 415 volumes prétraités\n37 cas R032 exclus + 1 quarantaine QC\nSplit v2 : 1 415 cas validés et exportés\nAnnotations brutes conservées',None,
         'La version 1 du split est archivée. La version 2 retire un cas ; les autres affectations sont conservées. Relecture collective de la quarantaine nécessaire. La tolérance de recadrage existante est de 1 %.'),
        ('Spark : accélération mesurée', 'RAS → 1 mm → z-score → recadrage → QC\nPilote : 8 volumes, un essai\nMeilleur temps : 27,63 s / 4 workers\nAccélération : ×2,05', 'preprocessing/reports/benchmark_speedup.png',
         'Le démarrage Spark est exclu du temps de calcul. Les workers ne sont pas des cœurs réservés. Ne pas extrapoler ces huit petits volumes au lot complet.'),
        ('Tester la généralisation par centre', '916 cas train / validation\n231 cas test ID · 268 cas test OOD\n5 plis séparés par centre\n6 centres entièrement OOD\nSeuils d’alerte réglés sur validation',None,
         'Pour chaque pli, le test ID ne retient que les centres présents dans son entraînement. Les centres OOD ne servent jamais au réglage des seuils.'),
        ('Le z-score aligne les échelles', 'Les distributions gardent des différences de forme.\nComparer les centres reste nécessaire.\nCes mesures ne suffisent pas à conclure à une baisse de Dice.', 'figures/s5_avant_apres_par_centre.png',
         'Les statistiques et figures proviennent des rapports locaux courants. La distance est descriptive. La sélection OOD utilisait déjà une distance, ce qui limite l’interprétation de son contraste ID/OOD.'),
        ('Empreinte du centre : quelle incertitude ?', 'Complément exploratoire S6\n912 train / 228 réservés · 25 centres\nForêt : 70,1 % avant / 51,1 % après\nÉcart : −19,0 points\nIC conditionnel : [−25,5 ; −12,6]', 'analysis/reports/s6_uncertainty/confidence_intervals.png',
         '10 000 bootstrap appariés stratifiés par centre. Comparaison 32 classes avant / 30 après. Modèles et partition fixés : pas de variabilité de réentraînement. La cohorte avait déjà été explorée. Ces chiffres complètent les moyennes de cinq plis initiales, sans les remplacer. Une moindre empreinte ne démontre pas une amélioration du Dice.'),
        ('Délai et Dice : première analyse historique', 'Baseline CNN · protocole historique\n167 ID / 257 OOD : cas complets\nOOD : gain de R² du délai = 0,0381\nAvec volume lésionnel : 0,0002\nExploratoire · distinct de C1 v2', 'analysis/reports/s8_historical/time_dice.png',
         '999 permutations intra-centre : p ID = 0,911, OOD = 0,001, non corrigées. L’association OOD s’atténue après prise en compte du volume. Ce résultat n’établit pas de causalité. ID servait à choisir le modèle. Les analyses S7 et réseau sur C1 v2 restent à produire.'),
        ('Démo locale et prochaine étape', 'Gradio : IRM, masque, volume, rapport JSON\nNiiVue : visualisation 3D locale\nAtlas MNI disponible ; incertitude selon modèle\nModèle final : raccordement préparé\nSecours synthétique clairement étiqueté',None,
         'Démontrer avec le phantom et préciser qu’il s’agit d’un masque fourni. Pour une vraie inférence, intégrer le modèle C8 puis enregistrer une vidéo. Image Docker construite et testée. Navigateur Chrome : coupes et 3D vérifiées. Vidéo MP4 de secours synthétique disponible.'),
    ]
    for index,(title,body,figure,notes) in enumerate(slides,1):
        slide=deck.slides.add_slide(deck.slide_layouts[6])
        slide.background.fill.solid();slide.background.fill.fore_color.rgb=RGBColor.from_string(BG)
        text(slide,'PROJET AVC / DONNÉES ET ANALYSES',.55,.25,10,.4,12,ACCENT)
        text(slide,title,.55,.85,12.2,.9,32)
        if figure and (ROOT/figure).exists():
            text(slide,body,.55,2.0,4.2,4.5,21)
            from PIL import Image
            with Image.open(ROOT/figure) as im: ratio=im.width/im.height
            width=min(7.5,4.5*ratio);height=width/ratio
            slide.shapes.add_picture(str(ROOT/figure),Inches(5.15+(7.5-width)/2),Inches(2+(4.5-height)/2),width=Inches(width),height=Inches(height))
        else:
            text(slide,body,.65,2.0,11.8,4.5,27)
        text(slide,f'RECHERCHE · résultats provisoires / voir rapports locaux                                      {index:02d} / 08',.55,7.0,12.2,.3,10)
        slide.notes_slide.notes_text_frame.text=notes
    path=OUT/'Soutenance_Projet_AVC.pptx';deck.save(path)
    check=Presentation(path)
    assert len(check.slides)==8
    print(path)


if __name__=='__main__':main()
