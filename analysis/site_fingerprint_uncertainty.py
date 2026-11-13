"""S6 complémentaire : lot réservé stratifié et bootstrap apparié, modèles fixés."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/avc-mpl')
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from site_fingerprint import ROOT, load_features, models


def stratified_paired_bootstrap(correct, sites, repeats=10000, seed=2026):
    """Mêmes indices pour toutes les colonnes ; poids égal par centre, effectifs conservés."""
    correct=np.asarray(correct,float);sites=np.asarray(sites)
    if correct.ndim!=2 or len(correct)!=len(sites) or len(sites)==0:
        raise ValueError('Matrices non alignées ou vides')
    if not np.isfinite(correct).all() or not np.isin(correct,[0,1]).all() or repeats<2:
        raise ValueError('Réussites binaires et au moins deux répétitions requises')
    rng=np.random.default_rng(seed);labels=np.unique(sites)
    draws=np.zeros((repeats,correct.shape[1]));point=np.zeros(correct.shape[1])
    for site in labels:
        values=correct[sites==site]
        point+=values.mean(axis=0)/len(labels)
        # Lots limités pour éviter un temporaire excessif.
        for start in range(0,repeats,250):
            count=min(250,repeats-start)
            indices=rng.integers(len(values),size=(count,len(values)))
            draws[start:start+count]+=values[indices].mean(axis=1)/len(labels)
    return point,draws


def interval(values):
    return np.quantile(values,[.025,.975]).tolist()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=10000)
    parser.add_argument('--output',type=Path,default=ROOT/'analysis/reports/s6_uncertainty')
    args=parser.parse_args()
    if args.repeats<100:raise ValueError('Au moins 100 répétitions pour le rapport')
    data,features=load_features(ROOT/'data/processed',ROOT/'data/qc_s4/subjects.parquet')
    if data.case_id.duplicated().any():raise ValueError('Sujets dupliqués')
    # Vérifier aussi l’absence de sessions multiples d’un même sujet.
    subjects=data.case_id.str.replace(r'^.*?_sub-','sub-',regex=True).str.replace(r'_ses-.*$','',regex=True)
    if subjects.duplicated().any():raise ValueError('Plusieurs sessions par sujet : partition groupée requise')
    y=data.site.to_numpy();seed=2026
    train,test=next(StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(np.zeros(len(y)),y))
    configs=[('avant_32',('avant','Forme de l’histogramme, 32 classes')),
             ('apres_30',('après','Forme de l’histogramme, 30 classes')),
             ('apres_150',('après','Forme de l’histogramme, 150 classes'))]
    columns=[];predictions=[];records=[]
    for model_name,estimator in models().items():
        if model_name=='random forest':estimator.set_params(n_jobs=2)
        for tag,key in configs:
            X=features[key]
            fitted=clone(estimator).fit(X[train],y[train])
            pred=fitted.predict(X[test]);name=f'{model_name}_{tag}'
            columns.append(name);predictions.append(pred)
            records.append({'name':name,'model':model_name,'representation':tag,
                            'balanced_accuracy':float(balanced_accuracy_score(y[test],pred))})
            print(name,records[-1]['balanced_accuracy'],flush=True)
    control=models()['random forest'].set_params(n_jobs=2)
    shuffled=np.random.default_rng(seed).permutation(y[train])
    control.fit(features[configs[-1][1]][train],shuffled)
    pred=control.predict(features[configs[-1][1]][test])
    columns.append('controle_labels_train_melanges');predictions.append(pred)
    records.append({'name':columns[-1],'model':'random forest','representation':'controle',
                    'balanced_accuracy':float(balanced_accuracy_score(y[test],pred))})
    predicted=np.column_stack(predictions)
    points,draws=stratified_paired_bootstrap(predicted==y[test,None],y[test],args.repeats,seed)
    for i,record in enumerate(records):
        assert np.isclose(record['balanced_accuracy'],points[i])
        record['ci_low'],record['ci_high']=interval(draws[:,i])
    comparisons=[]
    for model in models():
        before=columns.index(f'{model}_avant_32')
        after=columns.index(f'{model}_apres_30')
        lo,hi=interval(draws[:,after]-draws[:,before])
        comparisons.append(dict(model=model,comparison='apres_30_moins_avant_32',
                                delta=float(points[after]-points[before]),ci_low=lo,ci_high=hi))
    private=ROOT/'data/analysis/s6_uncertainty';private.mkdir(parents=True,exist_ok=True)
    cohort=data[['case_id','site']].copy();cohort['subset']='train';cohort.loc[test,'subset']='test'
    cohort.to_csv(private/'partition.csv',index=False)
    output=data.iloc[test][['case_id','site']].copy()
    for i,name in enumerate(columns):output[name]=predicted[:,i]
    output.to_csv(private/'predictions.csv',index=False)
    args.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(args.output/'intervals.csv',index=False)
    pd.DataFrame(comparisons).to_csv(args.output/'paired_differences.csv',index=False)
    cohort.groupby(['site','subset']).size().unstack(fill_value=0).to_csv(args.output/'counts_by_center.csv')
    summary={'seed':seed,'scans':len(data),'centers':len(set(y)),'train':len(train),'test':len(test),
             'bootstrap_repeats':args.repeats,'confidence_level':.95,'chance':1/len(set(y)),
             'partition_sha256':hashlib.sha256((private/'partition.csv').read_bytes()).hexdigest(),
             'method':'Bootstrap percentile apparié des sujets du test, stratifié par centre ; classifieurs entraînés une seule fois sur les 80 % train.',
             'scope':'Complément exploratoire à S6 ; ce lot réservé ne constitue pas un nouvel échantillon externe.',
             'limitations':['Intervalles conditionnels aux modèles et à la partition fixés ; variabilité de réentraînement non incluse.',
                            'Même cohorte déjà explorée en S6 ; aucune validation confirmatoire indépendante.',
                            'Centres présents dans train et test : reconnaissance du centre, pas généralisation à un centre inconnu.',
                            'Comparaison 32 classes avant / 30 après : différence de granularité restante.',
                            'Bootstrap suppose des sujets indépendants au sein du centre ; couverture non garantie sur ce petit échantillon.',
                            'Intervalles marginaux à 95 %, sans correction pour comparaisons multiples ; aucun Dice mesuré.',
                            'Le contrôle utilise une seule permutation des labels du train ; son intervalle ne constitue ni une distribution nulle ni un test contre le hasard.'],
             'scores':records,'paired_differences':comparisons}
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# S6 — incertitude de la reconnaissance du centre','',summary['scope'],'',
           f"Lot réservé : {len(test)} sujets ; entraînement : {len(train)} ; {len(set(y))} centres. {args.repeats:,} rééchantillonnages appariés.",'',
           '| Classifieur | Représentation | Exactitude équilibrée | IC conditionnel 95 % |','|---|---|---:|---:|']
    for r in records:lines.append(f"| {r['model']} | {r['representation']} | {r['balanced_accuracy']:.1%} | [{r['ci_low']:.1%} ; {r['ci_high']:.1%}] |")
    lines+=['','## Différence après − avant (30 / 32 classes)','']
    for r in comparisons:lines.append(f"- {r['model']} : {100*r['delta']:.1f} points, intervalle [{100*r['ci_low']:.1f} ; {100*r['ci_high']:.1f}] points.")
    lines+=['','## Interprétation et limites','','Ces résultats complètent les moyennes de cinq plis S6 existantes ; ils ne les remplacent pas. Un intervalle de différence s’obtient par rééchantillonnage apparié, pas par soustraction des bornes des deux intervalles.','']+['- '+v for v in summary['limitations']]
    lines+=['','## Reproduction','','```sh','.venv-spark/bin/python analysis/site_fingerprint_uncertainty.py','```','',
            'Sources méthodologiques : [bootstrap apparié, SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) ; [précautions sur les scores de validation croisée, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html).',
            'Prédictions individuelles et partition conservées localement dans `data/analysis/s6_uncertainty/`, ignorées par Git.','']
    (args.output/'README.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    for i,r in enumerate(records):
        value=r['balanced_accuracy'];ax.errorbar(value,i,xerr=[[value-r['ci_low']],[r['ci_high']-value]],fmt='o',capsize=4,color='#0f766e')
    ax.set_yticks(range(len(records)),[r['name'].replace('_',' ') for r in records])
    ax.axvline(summary['chance'],ls='--',color='gray',label=f"Hasard : {summary['chance']:.0%}")
    ax.set(xlim=(0,1),xlabel='Exactitude équilibrée · IC bootstrap conditionnels à 95 %',
           title='S6 complémentaire · lot réservé · modèles et partition fixés')
    ax.legend();fig.tight_layout();fig.savefig(args.output/'confidence_intervals.png',dpi=170);plt.close(fig)
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
