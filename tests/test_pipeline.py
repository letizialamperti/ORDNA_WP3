import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ordna_training.cli import main
from ordna_training.data import read_split,sample_triplet,TOKEN_MAP
from ordna_training.model import TripletEncoder,reference_objective


class PipelineTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_encoder_permutation_and_gradients(self):
        torch.manual_seed(1)
        model=TripletEncoder(token_emb_dim=2,seq_len=8,sample_repr_dim=16,sample_emb_dim=4)
        x=torch.randint(0,17,(2,4,2,8))
        a,r=model(x);b,_=model(x[:,[3,0,2,1]])
        torch.testing.assert_close(a,b,atol=2e-5,rtol=1e-4)
        self.assertEqual(tuple(r.shape),(2,16))
        loss,_=reference_objective(a,b,model(torch.flip(x,[0]))[0])
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_disjoint_triplet_reads(self):
        arrays=[np.arange(24).reshape(12,2,1)+100*i for i in range(3)]
        for seed in range(10):
            a,p,n,ai,ni=sample_triplet(arrays,4,np.random.default_rng(seed))
            self.assertNotEqual(ai,ni)
            self.assertFalse(set(a[:,0,0])&set(p[:,0,0]))

    def test_provided_splits(self):
        root=Path(__file__).resolve().parents[1]
        splits=[read_split(p) for p in sorted((root/'splits/strict_5km_3fold').glob('*.csv'))]
        self.assertEqual(len(splits),3)
        self.assertEqual([s['set'].eq('train').sum() for s in splits],[290,286,288])
        self.assertTrue(all(set(s.spygen_code)==set(splits[0].spygen_code) for s in splits))
        held=[set(s.loc[s['set'].eq('validation'),'spygen_code']) for s in splits]
        self.assertEqual(len(set.union(*held)),432)
        self.assertEqual(sum(map(len,held)),432)

    def test_train_isolation_and_frozen_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);raw=root/'raw';raw.mkdir();split=root/'split.csv'
            pd.DataFrame({'spygen_code':['a','b','short','zero','v'],'set':['train','train','train','train','validation']}).to_csv(split,index=False)
            for sample in ['a','b']:
                pd.DataFrame({'Forward':['ACGT','NNNN','acgu','gggg']*2,'Reverse':['tgca']*8}).to_csv(raw/f'{sample}.csv',index=False)
            # This cannot be tokenized: successful training proves validation was not opened.
            (raw/'v.csv').write_text('not,a,valid,raw,file\n')
            pd.DataFrame({'Forward':['acgt'],'Reverse':['tgca']}).to_csv(raw/'short.csv',index=False)
            pd.DataFrame({'Forward':['invalid!'],'Reverse':['tgca']}).to_csv(raw/'zero.csv',index=False)
            common=['--samples-dir',str(raw),'--split',str(split),'--cache-dir',str(root/'cache'),'--device','cpu']
            main(['train',*common,'--output',str(root/'train'),'--sequence-length','8','--subset-size','2','--token-emb-dim','2','--representation-dim','16','--embedding-dim','4','--batch-size','1','--steps-per-epoch','2','--epochs','1'])
            qc=pd.read_csv(root/'train/training_qc.csv')
            self.assertEqual(set(qc.loc[qc.included_training,'Sample']),{'a','b'})
            exclusions=pd.read_csv(root/'train/excluded/excluded_samples.csv')
            self.assertEqual(set(exclusions.Sample),{'short','zero'})
            self.assertEqual(dict(zip(exclusions.Sample,exclusions.valid_pairs)),{'short':1,'zero':0})
            self.assertEqual(pd.read_csv(root/'train/sample_roles.csv').spygen_code.tolist(),['a','b','v'])
            self.assertEqual(len(pd.read_csv(root/'train/sample_roles_original.csv')),5)
            self.assertTrue((root/'train/checkpoints/best_train_loss.pt').exists())
            history=json.loads((root/'train/history.json').read_text())
            self.assertEqual(set(history[0]['training_sample_draw_counts']),{'a','b'})
            pd.DataFrame({'Forward':['ACGT','nnnn','acgu'],'Reverse':['tgca']*3}).to_csv(raw/'v.csv',index=False)
            main(['export',*common,'--checkpoint',str(root/'train/checkpoints/last.pt'),'--output',str(root/'export')])
            for dim in [4,16]:
                path=root/'export'/('embeddings_projector_4D.csv' if dim==4 else 'embeddings_preprojector_16D.csv')
                exported=pd.read_csv(path)
                self.assertEqual(exported.Sample.tolist(),['a','b','v'])
                self.assertEqual(exported.role.tolist(),['train','train','outer_eval'])
                self.assertEqual(len([c for c in exported if c.startswith('Dim')]),dim)
                self.assertTrue(np.isfinite(exported.filter(regex='^Dim').to_numpy()).all())
            self.assertTrue(json.loads((root/'export/export_config.json').read_text())['model_frozen'])
            exported_before=pd.read_csv(root/'export/embeddings_projector_4D.csv')
            # Completed exports must skip all forwards when resumed.
            with patch.object(TripletEncoder,'forward',side_effect=AssertionError('Unexpected recomputation')):
                main(['export',*common,'--checkpoint',str(root/'train/checkpoints/last.pt'),'--output',str(root/'export'),'--resume'])
            pd.testing.assert_frame_equal(exported_before,pd.read_csv(root/'export/embeddings_projector_4D.csv'))
            with self.assertRaisesRegex(ValueError,'configuration mismatch'):
                main(['export',*common,'--checkpoint',str(root/'train/checkpoints/last.pt'),'--output',str(root/'export'),'--resume','--max-subsets','1'])
            # Simulate interruption on sample two: sample one must be reusable.
            original_forward=TripletEncoder.forward
            counter=[0]
            def interrupted(model,*args,**kwargs):
                counter[0]+=1
                if counter[0]==2:raise RuntimeError('simulated interruption')
                return original_forward(model,*args,**kwargs)
            partial=['export',*common,'--checkpoint',str(root/'train/checkpoints/last.pt'),'--output',str(root/'partial_export'),'--max-subsets','1','--resume']
            with patch.object(TripletEncoder,'forward',interrupted):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):main(partial)
            self.assertEqual(len(list((root/'partial_export/samples').glob('*.json'))),1)
            main(partial)
            self.assertEqual(json.loads((root/'partial_export/progress.json').read_text())['state'],'complete')
            self.assertEqual(len(pd.read_csv(root/'partial_export/embeddings_projector_4D.csv')),3)
            changed=root/'changed.csv'
            pd.DataFrame({'spygen_code':['a','b','v'],'set':['train','validation','train']}).to_csv(changed,index=False)
            with self.assertRaisesRegex(ValueError,'Split differs'):
                main(['export',*common,'--split',str(changed),'--checkpoint',str(root/'train/checkpoints/last.pt'),'--output',str(root/'forbidden')])


if __name__=='__main__':unittest.main()
