import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from integration.run_remaining import validate_experiment


class RemainingRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.split=self.root/'splits.json'
        data={'cases':[{'case_id':c,'site':c.upper()} for c in ('train','val','id','ood')],
              'ood_test':['ood'],'folds':[{'train':['train'],'val':['val'],'id_test':['id']}]}
        self.split.write_text(json.dumps(data))
        self.item={'model':'cnn','technique':'baseline','fold':0,'results':'results.csv',
                   'split_sha256':hashlib.sha256(self.split.read_bytes()).hexdigest()}

    def write_scores(self,cases):
        pd.DataFrame([dict(model='cnn',technique='baseline',fold=0,case_id=c,dice=.6) for c in cases]).to_csv(self.root/'results.csv',index=False)

    def test_missing_delivery_reports_missing(self):
        self.assertEqual(validate_experiment(self.item,self.split,self.root)['status'],'missing')

    def test_wrong_split_rejected_even_if_scores_exist(self):
        self.write_scores(['val','id','ood']);self.item['split_sha256']='old-protocol'
        with self.assertRaises(ValueError):validate_experiment(self.item,self.split,self.root)

    def test_partial_delivery_and_training_cases_rejected(self):
        for cases in (['val','id'],['val','id','ood','train']):
            self.write_scores(cases)
            with self.assertRaises(ValueError):validate_experiment(self.item,self.split,self.root)

    def test_complete_delivery_is_ready(self):
        self.write_scores(['val','id','ood'])
        result=validate_experiment(self.item,self.split,self.root)
        self.assertEqual(result['status'],'ready');self.assertEqual(result['scores'],3)


if __name__=='__main__':unittest.main()
