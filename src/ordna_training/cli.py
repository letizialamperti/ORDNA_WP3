"""Training uses split train IDs only; validation IDs enter only frozen inference."""
import argparse
import csv
import hashlib
import json
import platform
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import read_split,locate_files,tokenize_file,sample_triplet,InsufficientReadsError
from .model import TripletEncoder,reference_objective


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def device_for(name):
    if name=='auto':name='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable')
    if name=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS requested but unavailable')
    return torch.device(name)


def model_config(args):
    return {'token_emb_dim':args.token_emb_dim,'seq_len':args.sequence_length,'sample_repr_dim':args.representation_dim,'sample_emb_dim':args.embedding_dim}


def train(args):
    split=read_split(args.split)
    files=locate_files(args.samples_dir,split.spygen_code)
    if args.output.exists():raise ValueError(f'Output directory already exists: {args.output}; choose a fresh run directory')
    if args.steps_per_epoch<1 or args.epochs<1 or args.batch_size<1 or args.subset_size<1:
        raise ValueError('Positive epochs, steps, batch size and subset size required')
    original_train_ids=split.loc[split['set'].eq('train'),'spygen_code'].tolist()
    if len(original_train_ids)<2:raise ValueError('At least two training samples required')
    args.output.mkdir(parents=True)
    excluded_dir=args.output/'excluded';excluded_dir.mkdir()
    split.to_csv(args.output/'sample_roles_original.csv',index=False)
    arrays=[];qc=[];train_ids=[];excluded=[]
    # Only training files are opened/tokenized here. Validation files are located, not read.
    for sample in original_train_ids:
        try:
            arr,info=tokenize_file(files[sample],args.cache_dir,args.sequence_length)
        except InsufficientReadsError as error:
            arr=None;info=error.info
        enough=info['valid_pairs']>=2*args.subset_size
        qc.append({'Sample':sample,'role':'train' if enough else 'excluded','included_training':enough,**info})
        if not enough:
            excluded.append({'Sample':sample,'original_role':'train','reason':'insufficient_valid_paired_reads','valid_pairs':info['valid_pairs'],'required_pairs':2*args.subset_size,'source':str(files[sample])})
            print(f'EXCLUDED {sample}: {info["valid_pairs"]} valid pairs; need {2*args.subset_size}',flush=True)
            continue
        arrays.append(arr);train_ids.append(sample)
    effective_split=split.loc[split['set'].eq('validation')|split.spygen_code.isin(train_ids)].copy()
    effective_split.to_csv(args.output/'sample_roles.csv',index=False)
    pd.DataFrame(excluded,columns=['Sample','original_role','reason','valid_pairs','required_pairs','source']).to_csv(excluded_dir/'excluded_samples.csv',index=False)
    pd.DataFrame(qc).to_csv(args.output/'training_qc.csv',index=False)
    (excluded_dir/'summary.json').write_text(json.dumps({'original_n_train':len(original_train_ids),'effective_n_train':len(train_ids),'n_excluded':len(excluded),'validation_changed':False,'minimum_valid_pairs':2*args.subset_size},indent=2))
    if len(train_ids)<2:raise ValueError('Fewer than two training samples remain after exclusions; exclusion audit saved')
    checkpoints=args.output/'checkpoints';checkpoints.mkdir()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=device_for(args.device)
    model=TripletEncoder(**model_config(args)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[2,8],gamma=.3)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(model=model_config(args),split_sha256=sha(args.split),n_train=len(train_ids),n_validation=int(split['set'].eq('validation').sum()),device_used=str(device),python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__,sample_selection='uniform training sample pairs; non-overlapping random anchor/positive reads',checkpoint_selection='last epoch of prespecified budget; no validation labels or features used',regularizer='upstream batch Frobenius norm sum, dynamic weight',source_model_sha256=sha(Path(__file__).with_name('model.py')))
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    history=[];best_training_loss=float('inf')
    for epoch in range(args.epochs):
        model.train();losses=[];triplets=[];norms=[];counts=np.zeros(len(arrays),dtype=int)
        for step in range(args.steps_per_epoch):
            rng=np.random.default_rng(np.random.SeedSequence([args.seed,epoch,step]))
            batches=[[],[],[]]
            for _ in range(args.batch_size):
                a,p,n,ai,ni=sample_triplet(arrays,args.subset_size,rng)
                for group,value in zip(batches,[a,p,n]):group.append(value)
                counts[ai]+=2;counts[ni]+=1
            optimizer.zero_grad(set_to_none=True)
            embedded=[model(torch.tensor(np.stack(group),device=device))[0] for group in batches]
            loss,details=reference_objective(*embedded,margin=args.margin,l2_max_weight=args.norm_weight)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();optimizer.step()
            losses.append(float(loss.detach().cpu()));triplets.append(float(details['triplet'].cpu()));norms.append(float(details['norm'].cpu()))
        lr=optimizer.param_groups[0]['lr'];scheduler.step()
        record={'epoch':epoch+1,'loss':float(np.mean(losses)),'triplet_loss':float(np.mean(triplets)),'norm':float(np.mean(norms)),'learning_rate':lr,'training_sample_draw_counts':dict(zip(train_ids,counts.tolist()))}
        history.append(record)
        checkpoint={'model_config':model_config(args),'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'training_config':config,'completed_epochs':epoch+1,'split_sha256':config['split_sha256'],'train_ids':train_ids,'original_train_ids':original_train_ids,'excluded_train_ids':[r['Sample'] for r in excluded]}
        temporary=checkpoints/'last.partial.pt';torch.save(checkpoint,temporary);temporary.replace(checkpoints/'last.pt')
        if record['loss']<best_training_loss:
            best_training_loss=record['loss']
            temporary=checkpoints/'best_train_loss.partial.pt';torch.save(checkpoint,temporary);temporary.replace(checkpoints/'best_train_loss.pt')
        (args.output/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps({k:v for k,v in record.items() if k!='training_sample_draw_counts'}),flush=True)
    print(f'Training complete: {checkpoints / "last.pt"}',flush=True)


def export(args):
    from .exporting import export_embeddings
    return export_embeddings(args,sha,device_for)


def audit(args):
    split=read_split(args.split);files=locate_files(args.samples_dir,split.spygen_code)
    report={'split_sha256':sha(args.split),'counts':split['set'].value_counts().to_dict(),'files_found':len(files),'labels_used':False,'training_started':False}
    print(json.dumps(report,indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ['audit','train','export']:
        p=sub.add_parser(command)
        p.add_argument('--samples-dir',type=Path,required=True)
        p.add_argument('--split',type=Path,required=True)
        if command=='audit':continue
        p.add_argument('--output',type=Path,required=True)
        p.add_argument('--cache-dir',type=Path,required=True)
        p.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto')
        p.add_argument('--seed',type=int,default=0)
        if command=='export':
            p.add_argument('--checkpoint',type=Path,required=True)
            p.add_argument('--resume',action='store_true',help='Resume only matching export configuration; skip completed samples')
            p.add_argument('--log-every-subsets',type=int,default=50)
            p.add_argument('--max-subsets',type=int,default=None)
            continue
        p.add_argument('--sequence-length',type=int,required=True)
        p.add_argument('--subset-size',type=int,default=1000)
        p.add_argument('--embedding-dim',type=int,default=64)
        p.add_argument('--representation-dim',type=int,default=256)
        p.add_argument('--token-emb-dim',type=int,default=8)
        p.add_argument('--batch-size',type=int,default=8)
        p.add_argument('--steps-per-epoch',type=int,required=True)
        p.add_argument('--epochs',type=int,required=True)
        p.add_argument('--learning-rate',type=float,default=1e-4)
        p.add_argument('--weight-decay',type=float,default=.01)
        p.add_argument('--margin',type=float,default=.3)
        p.add_argument('--norm-weight',type=float,default=.1)
    args=parser.parse_args(argv)
    if getattr(args,'log_every_subsets',1)<1:parser.error('--log-every-subsets must be positive')
    if getattr(args,'max_subsets',None) is not None and args.max_subsets<1:parser.error('--max-subsets must be positive')
    {'audit':audit,'train':train,'export':export}[args.command](args)


if __name__=='__main__':main()
