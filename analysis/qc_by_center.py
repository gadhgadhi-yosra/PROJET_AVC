"""Synthèse descriptive du QC par centre, sans modification des sujets ni des exclusions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/avc-mpl')
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def summarize(frame):
    required={'case_id','site','status','empty_mask','days_post_stroke','chronicity','inventory_warnings','error'}
    if not required<=set(frame):raise ValueError('Colonnes QC manquantes')
    if frame.case_id.isna().any() or frame.case_id.duplicated().any():raise ValueError('Identifiants absents ou dupliqués')
    if frame.site.isna().any() or frame.site.astype(str).str.strip().eq('').any():raise ValueError('Centre absent')
    if not frame.status.isin(['ok','excluded','error']).all():raise ValueError('Statut QC inconnu')
    rows=[]
    for site,g in frame.groupby('site',sort=True):
        ok=g[g.status=='ok'];days=pd.to_numeric(ok.days_post_stroke,errors='coerce')
        valid=np.isfinite(days)&days.ge(0)
        masks=ok.empty_mask
        if not masks.dropna().isin([True,False]).all():raise ValueError('Indicateur masque vide invalide')
        warnings=ok.inventory_warnings.fillna('').astype(str).str.strip().ne('')
        row=dict(site=site,inventoried=len(g),retained=len(ok),excluded=int(g.status.eq('excluded').sum()),
                 errors=int(g.status.eq('error').sum()),empty_masks=int(masks.eq(True).sum()),
                 nonempty_masks=int(masks.eq(False).sum()),mask_status_unknown=int(masks.isna().sum()),
                 delay_available=int(valid.sum()),delay_missing_or_invalid=int((~valid).sum()),
                 chronicity_flag_present=int(ok.chronicity.eq(True).sum()),
                 chronicity_unspecified=int(ok.chronicity.isna().sum()),
                 metadata_warning_cases=int(warnings.sum()))
        row['delay_missing_or_invalid_percent']=100*row['delay_missing_or_invalid']/len(ok) if len(ok) else np.nan
        rows.append(row)
    table=pd.DataFrame(rows).set_index('site')
    assert (table.inventoried==table.retained+table.excluded+table.errors).all()
    assert (table.retained==table.empty_masks+table.nonempty_masks+table.mask_status_unknown).all()
    assert (table.retained==table.delay_available+table.delay_missing_or_invalid).all()
    return table


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subjects',type=Path,default=ROOT/'data/processed/subjects.parquet')
    parser.add_argument('--output',type=Path,default=ROOT/'analysis/reports/qc_by_center')
    args=parser.parse_args();frame=pd.read_parquet(args.subjects);table=summarize(frame)
    args.output.mkdir(parents=True,exist_ok=True)
    table.to_csv(args.output/'qc_by_center.csv')
    excluded=frame[frame.status!='ok'].groupby(['site','status','error'],dropna=False).size().rename('n').reset_index()
    excluded.to_csv(args.output/'exclusion_reasons.csv',index=False)
    warning_rows=[]
    for r in frame.loc[frame.status=='ok',['site','inventory_warnings']].itertuples():
        for code in sorted(set(str(r.inventory_warnings).split(';'))) if pd.notna(r.inventory_warnings) else []:
            code=code.strip()
            if code:warning_rows.append({'site':r.site,'warning':code})
    warning_table=pd.DataFrame(warning_rows,columns=['site','warning']).groupby(['site','warning']).size().rename('n').reset_index()
    warning_table.to_csv(args.output/'metadata_warnings.csv',index=False)
    counts={c:int(table[c].sum()) for c in table.columns if not c.endswith('_percent')}
    summary={'date':'2026-11-13','source':str(args.subjects.relative_to(ROOT)) if args.subjects.is_relative_to(ROOT) else args.subjects.name,
             'source_sha256':hashlib.sha256(args.subjects.read_bytes()).hexdigest(),
             'centers_inventoried':len(table),'centers_retained':int(table.retained.gt(0).sum()),
             'centers_without_valid_delay_among_retained':int(((table.retained>0)&(table.delay_available==0)).sum()),
             'counts':counts,'denominators':{'statuses':'Tous les sujets inventoriés dans la table de prétraitement.',
             'masks_and_metadata':'Sujets status=ok uniquement ; les exclusions ne sont pas comptées comme métadonnées manquantes.',
             'delay_rate':'Nombre de délais absents, non numériques, non finis ou négatifs / sujets retenus du centre ; non défini si aucun retenu.'},
             'limits':['CHRONICITY non renseigné ne signifie pas aigu, chronique ou absence de maladie.',
             'Un masque vide est conservé comme cas négatif ; il ne constitue pas automatiquement une anomalie.',
             'Un centre sans sujet retenu a des taux non définis, pas des taux nuls.',
             'Le tableau résume les contrôles existants ; aucune nouvelle exclusion ni validation anatomique.']}
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# Contrôle qualité par centre','',
           f"État au 13 novembre 2026 : **{counts['inventoried']} sujets inventoriés**, **{counts['retained']} retenus**, **{counts['excluded']} exclus**, **{counts['errors']} erreurs actives**.",'',
           f"{summary['centers_inventoried']} centres inventoriés ; {summary['centers_retained']} représentés dans la livraison.",'',
           f"Parmi les sujets retenus : {counts['empty_masks']} masques vides, {counts['mask_status_unknown']} statuts de masque inconnus, {counts['delay_available']} délais disponibles et {counts['delay_missing_or_invalid']} manquants ou invalides.",'',
           'Les effectifs QC couvrent tout l’inventaire. Les masques et métadonnées ci-dessous sont comptés uniquement parmi les sujets retenus.','',
           '| Centre | Inventoriés | Retenus | Exclus | Erreurs | Masques vides | Délai manquant/invalide |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for site,r in table.iterrows():
        missing=f"{int(r.delay_missing_or_invalid)} / {int(r.retained)} ({r.delay_missing_or_invalid_percent:.1f} %)" if r.retained else 'Non défini'
        empty=str(int(r.empty_masks)) if r.retained else 'Non défini'
        lines.append(f'| {site} | {int(r.inventoried)} | {int(r.retained)} | {int(r.excluded)} | {int(r.errors)} | {empty} | {missing} |')
    lines+=['','## Définitions et limites','']+['- '+v for v in summary['limits']]
    lines+=['','Le CSV détaillé distingue masques vides, non vides et statut inconnu, et conserve les avertissements de métadonnées. Les codes d’avertissement sont détaillés dans `metadata_warnings.csv` ; un sujet peut avoir plusieurs codes. Les motifs d’exclusion sont dans `exclusion_reasons.csv`.','',
            '## Reproduction','','```sh','.venv-spark/bin/python analysis/qc_by_center.py','```','',
            'Figure : `qc_by_center.png`. Les fichiers sources, annotations, partitions et décisions QC restent inchangés.','']
    (args.output/'README.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,max(9,len(table)*.27)),sharey=True)
    positions=np.arange(len(table));colors=['#0f766e','#c2410c','#991b1b'];bottom=np.zeros(len(table))
    for column,color,label in zip(['retained','excluded','errors'],colors,['Retenus','Exclus','Erreurs']):
        axes[0].barh(positions,table[column],left=bottom,color=color,label=label);bottom+=table[column].to_numpy()
    axes[0].set_yticks(positions,table.index,fontsize=8);axes[0].invert_yaxis();axes[0].legend(fontsize=8);axes[0].set_xlabel('Sujets inventoriés')
    axes[1].barh(positions,table.delay_missing_or_invalid_percent.fillna(0),color='#c2410c');axes[1].set(xlim=(0,105),xlabel='Délai absent / invalide (% des retenus)')
    axes[2].barh(positions,table.empty_masks,color='#0f766e');axes[2].set_xlabel('Masques vides parmi les retenus');axes[2].set_xlim(0,max(2,table.empty_masks.max()+1))
    for i,(_,r) in enumerate(table.iterrows()):
        if not r.retained:
            for ax in axes[1:]:ax.text(.3,i,'N/A',va='center',fontsize=8,color='gray')
    for ax in axes:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('QC par centre · livraison C1 v2 · aucune nouvelle exclusion')
    fig.tight_layout();fig.savefig(args.output/'qc_by_center.png',dpi=150);plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
