"""Scientific boundaries: PAD invariance, cell weighting, accumulation and resume."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from input_pipeline import JointMLMCollator, RandomLengthBatchSampler
from model import ILC2Transformer, ModelConfig, MODEL_INPUTS, mlm_objective
from train import (make_optimizer, make_scheduler, save_checkpoint, load_training_checkpoint,
                   train, evaluate, seed_all)


NORM=dict(mask_probability=.5,pad_token_id=0,mask_token_id=1,first_gene_token_id=2)
INPUT=dict(mask_seed=0,mask_replacement_fraction=.8,random_replacement_fraction=.1)


class TinyCells(Dataset):
    def __init__(self):
        self.lengths=np.array([5,12,8,7,4,15,9])
        self.frame=pd.DataFrame(dict(study=["s1"]*4+["s2"]*3,tissue=["lung"]*7))

    def __len__(self):return len(self.lengths)

    def __getitem__(self,key):
        row,epoch=key if isinstance(key,tuple) else (key,0)
        n=self.lengths[row]
        return dict(gene_ids=np.arange(n,dtype=np.int32)+2,values=np.linspace(.2,2,n,dtype=np.float32),
                    cell_id=f"c{row}",source_row=row,sample_index=row,epoch=epoch,split="train")


def loader():
    data=TinyCells()
    return DataLoader(data,batch_sampler=RandomLengthBatchSampler(data.lengths,2,4),
                      collate_fn=JointMLMCollator(NORM,INPUT,20,True),num_workers=0,
                      generator=torch.Generator().manual_seed(0))


def small_model(dropout=0):
    return ILC2Transformer(ModelConfig(n_genes=20,max_length=64,hidden_size=24,num_heads=3,
                                      intermediate_size=48,num_layers=2,dropout=dropout))


class ModelTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):seed_all(17)

    def test_full_encoder_padding_invariance_and_mean_pool(self):
        model=small_model().eval()
        batch=next(iter(loader()))
        inputs={k:batch[k] for k in MODEL_INPUTS}
        a=model(**inputs)
        padded={k:torch.nn.functional.pad(v,(0,9),value=False if k=="attention_mask" else 0) for k,v in inputs.items()}
        b=model(**padded)
        torch.testing.assert_close(a['gene_embeddings'],b['gene_embeddings'][:,:inputs['input_ids'].shape[1]],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(a['cell_embedding'],b['cell_embedding'],atol=1e-6,rtol=1e-5)
        self.assertTrue((b['gene_embeddings'][~padded['attention_mask']]==0).all())

    def test_targets_only_tied_real_gene_head_and_cell_macro(self):
        model=small_model()
        batch=next(iter(loader()))
        out=mlm_objective(model,batch)
        mask=batch['target_mask']
        logits,values=model.predict_targets(model(**{k:batch[k] for k in MODEL_INPUTS})['gene_embeddings'],mask)
        self.assertEqual(logits.shape,(int(mask.sum()),20))
        logits.sum().backward()
        self.assertTrue(model.gene_embedding.weight.grad[2:].abs().sum()>0)
        manual=[]
        for cell in range(len(mask)):
            selected=out['target_cells']==cell
            if selected.any():manual.append(out['id_losses'][selected].mean())
        torch.testing.assert_close(out['id_sum']/out['valid_cells'],torch.stack(manual).mean())
        with self.assertRaises(ValueError):
            model(**{**{k:batch[k] for k in MODEL_INPUTS},'input_ids':torch.zeros(1,65,dtype=torch.long),
                     'values':torch.zeros(1,65),'attention_mask':torch.ones(1,65,dtype=torch.bool),
                     'position_ids':torch.arange(65)[None]})

    def test_unequal_microbatches_match_complete_batch_gradients(self):
        data=TinyCells()
        collate=JointMLMCollator(NORM,INPUT,20,True)
        samples=[data[i] for i in range(7)]
        full=collate(samples)
        a=small_model();b=copy.deepcopy(a)
        result=mlm_objective(a,full)
        ((.2*result['id_sum']+.8*result['value_sum'])/result['valid_cells']).backward()
        n=int(full['loss_cell_mask'].sum())
        for start in range(0,7,2):
            part=mlm_objective(b,collate(samples[start:start+2]))
            ((.2*part['id_sum']+.8*part['value_sum'])/n).backward()
        for p,q in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-6)

    def test_empty_targets_and_idf_pooling(self):
        model=small_model()
        collate=JointMLMCollator({**NORM,'mask_probability':0},INPUT,20,True)
        batch=collate([TinyCells()[0]])
        result=mlm_objective(model,batch)
        self.assertEqual(int(result['valid_cells']),0)
        self.assertEqual(float(result['id_sum']+result['value_sum']),0)
        encoded=model(**{k:batch[k] for k in MODEL_INPUTS})
        pooled=model.pool_idf(encoded['gene_embeddings'],batch['input_ids'],batch['attention_mask'],torch.ones(20))
        torch.testing.assert_close(pooled,encoded['cell_embedding'])

    def test_checkpoint_restores_optimizer_scheduler_rng_and_next_update(self):
        cfg=dict(learning_rate=3e-4,min_learning_rate=1e-6,weight_decay=.01,adam_betas=[.9,.999],adam_eps=1e-8,warmup_fraction=.1)
        model=small_model(dropout=.1)
        optimizer=make_optimizer(model,cfg);scheduler=make_scheduler(optimizer,cfg,20)
        batches=list(loader())

        def step(batch):
            optimizer.zero_grad(set_to_none=True)
            out=mlm_objective(model,batch)
            ((.2*out['id_sum']+.8*out['value_sum'])/out['valid_cells']).backward()
            optimizer.step();scheduler.step()

        step(batches[0])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'last.pt'
            state=dict(sampler=loader().batch_sampler.state_dict(1),global_step=1)
            save_checkpoint(path,model,optimizer,scheduler,state,cfg,INPUT,{'vocab':'fixed'},{'vocab':b'example'})
            step(batches[1]);expected=copy.deepcopy(model.state_dict());expected_lr=scheduler.get_last_lr()
            restored=load_training_checkpoint(path,model,optimizer,scheduler,cfg,INPUT,{'vocab':'fixed'})
            self.assertEqual(restored['sampler']['start_batch'],1)
            step(batches[1])
            for k,v in model.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
            self.assertEqual(scheduler.get_last_lr(),expected_lr)
            with self.assertRaises(ValueError):
                load_training_checkpoint(path,model,optimizer,scheduler,cfg,INPUT,{'vocab':'changed'})

    def test_complete_tiny_training_validation_and_finished_resume(self):
        cfg_path=Path(__file__).resolve().parents[1]/'configs/training_e1b_v03.json'
        import json
        cfg=json.loads(cfg_path.read_text())
        with tempfile.TemporaryDirectory() as folder:
            cfg.update(output=folder,epochs=2,accumulation_steps=3,checkpoint_every_steps=1,log_every_steps=20,precision='fp32')
            model=small_model()
            vocab=pd.DataFrame(dict(gene_id=np.arange(20),gene_symbol=np.arange(20),training_positive_count=1,is_functional67=False))
            train(model,loader(),loader(),vocab,cfg,INPUT,torch.device('cpu'),{'tiny':'test'},{})
            checkpoint=torch.load(Path(folder)/'last.pt',weights_only=False)
            self.assertEqual(checkpoint['state']['global_step'],4)  # Two full and two tail accumulation windows.
            self.assertTrue(checkpoint['state']['finished'])
            self.assertTrue((Path(folder)/'best.pt').exists())
            self.assertEqual(len(pd.read_csv(Path(folder)/'metrics.csv')),4)
            self.assertTrue(np.isfinite(evaluate(model,loader(),torch.device('cpu'),cfg).summary(cfg)['loss']))
            train(model,loader(),loader(),vocab,cfg,INPUT,torch.device('cpu'),{'tiny':'test'},{},resume=Path(folder)/'last.pt')

    def test_partial_epoch_resume_matches_uninterrupted_training(self):
        import json
        cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/training_e1b_v03.json').read_text())
        cfg.update(epochs=2,accumulation_steps=3,checkpoint_every_steps=1,log_every_steps=20,precision='fp32')
        vocab=pd.DataFrame(dict(gene_id=np.arange(20),gene_symbol=np.arange(20),training_positive_count=1,is_functional67=False))
        with tempfile.TemporaryDirectory() as folder:
            cfg['output']=str(Path(folder)/'reference')
            seed_all(77);reference=small_model(dropout=.1)
            train(reference,loader(),loader(),vocab,cfg,INPUT,torch.device('cpu'),{'tiny':'test'},{})
            expected=copy.deepcopy(reference.state_dict())
            cfg['output']=str(Path(folder)/'resumed')
            seed_all(77);interrupted=small_model(dropout=.1)

            def save_then_interrupt(*args,**kwargs):
                save_checkpoint(*args,**kwargs)
                raise RuntimeError('simulated process interruption')

            with patch('train.save_checkpoint',side_effect=save_then_interrupt):
                with self.assertRaisesRegex(RuntimeError,'simulated process interruption'):
                    train(interrupted,loader(),loader(),vocab,cfg,INPUT,torch.device('cpu'),{'tiny':'test'},{})
            checkpoint=Path(cfg['output'])/'last.pt'
            saved=torch.load(checkpoint,weights_only=False)
            self.assertEqual(saved['state']['sampler']['start_batch'],3)
            self.assertIsInstance(saved['state']['train_metrics'],dict)
            resumed=small_model(dropout=.1)
            train(resumed,loader(),loader(),vocab,cfg,INPUT,torch.device('cpu'),{'tiny':'test'},{},resume=checkpoint)
            for key,value in resumed.state_dict().items():
                torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
            actual=pd.read_csv(Path(cfg['output'])/'metrics.csv')
            wanted=pd.read_csv(Path(folder)/'reference/metrics.csv')
            pd.testing.assert_frame_equal(actual,wanted)


if __name__=='__main__':unittest.main()
