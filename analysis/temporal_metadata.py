"""S8 : disponibilité et distributions du délai, sans inventer de scores Dice."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/avc-mpl')
import numpy as np
import pandas as pd


def main():
    root=Path(__file__).resolve().parents[1]
    data=pd.read_parquet(root/'data/processed/subjects.parquet')
    data=data[data.status=='ok'].copy()
    days=pd.to_numeric(data.days_post_stroke,errors='coerce')
    valid=np.isfinite(days)&(days>=0)
    data['delay_available']=valid
    data['valid_days']=days.where(valid)
    table=data.groupby('site').agg(n=('case_id','size'),delay_available=('delay_available','sum'),
        median_days=('valid_days','median'),min_days=('valid_days','min'),max_days=('valid_days','max'))
    table['missing_or_invalid']=table.n-table.delay_available
    out=root/'analysis/reports/s8_metadata';out.mkdir(exist_ok=True)
    table.to_csv(out/'delay_by_center.csv')
    log=np.log1p(days[valid]);sites=data.loc[valid,'site']
    grouped=pd.DataFrame({'log_days':log,'site':sites}).groupby('site').log_days
    total=float(((log-log.mean())**2).sum())
    eta=float((grouped.size()*(grouped.mean()-log.mean())**2).sum()/total) if total else None
    summary=dict(subjects=len(data),valid_delay=int(valid.sum()),missing_or_invalid=int((~valid).sum()),
        centers=int(data.site.nunique()),centers_without_delay=int((table.delay_available==0).sum()),
        median_days=float(days[valid].median()),eta_squared_log_delay_by_center=eta,
        interpretation='Description des métadonnées ; aucune relation avec le Dice mesurée. CHRONICITY ne définit pas un stade.')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,7))
    ordered=table.sort_values('median_days').index
    axes[0].bar(ordered,table.loc[ordered,'delay_available'],color='#0f766e',label='Délai disponible')
    axes[0].bar(ordered,table.loc[ordered,'missing_or_invalid'],bottom=table.loc[ordered,'delay_available'],color='#c2410c',label='Manquant / invalide')
    axes[0].legend();axes[0].set_ylabel('Sujets');axes[0].tick_params(axis='x',rotation=90,labelsize=7)
    available=table[table.delay_available>0].sort_values('median_days').index
    axes[1].boxplot([data.loc[(data.site==s)&valid,'valid_days'] for s in available],showfliers=False)
    axes[1].set_xticks(range(1,len(available)+1),available,rotation=90,fontsize=7)
    axes[1].set_yscale('symlog');axes[1].set_ylabel('Délai post-AVC (jours)')
    fig.suptitle('Délai post-AVC par centre · cohorte QC v2 · analyse descriptive')
    fig.tight_layout();fig.savefig(root/'figures/s8_delai_metadonnees.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
