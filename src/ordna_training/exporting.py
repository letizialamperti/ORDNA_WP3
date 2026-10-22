"""Frozen inference with atomic per-sample results and checked restart support."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import read_split,locate_files,tokenize_file
from .model import TripletEncoder


def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def file_signature(path):
    stat=path.stat()
    return {'source':str(path.resolve()),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def export_embeddings(args,sha,device_for):
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    split=read_split(args.split)
    if checkpoint['split_sha256']!=sha(args.split):raise ValueError('Split differs from training checkpoint')
    original_ids=split.loc[split['set'].eq('train'),'spygen_code'].tolist()
    if original_ids!=checkpoint.get('original_train_ids',checkpoint['train_ids']):raise ValueError('Training IDs differ from checkpoint')
    train_ids=checkpoint['train_ids'];excluded=checkpoint.get('excluded_train_ids',[])
    if set(train_ids)&set(excluded) or set(train_ids)|set(excluded)!=set(original_ids):
        raise ValueError('Checkpoint exclusion manifest is inconsistent')
    split=split.loc[split['set'].eq('validation')|split.spygen_code.isin(train_ids)].copy()
    files=locate_files(args.samples_dir,split.spygen_code)
    identity={'checkpoint_sha256':sha(args.checkpoint),'split_sha256':sha(args.split),'export_seed':args.seed,'max_subsets':args.max_subsets,'aggregation':'equal subset means including final partial subset; population std ddof0','n_train':len(train_ids),'n_validation':len(split)-len(train_ids),'model_frozen':True,'checkpoint_training_config':checkpoint['training_config'],'sample_sources':{s:file_signature(files[s]) for s in split.spygen_code},'export_format_version':2}
    config_path=args.output/'export_config.json'
    if args.output.exists():
        if not args.resume:raise ValueError(f'Export output already exists: {args.output}; use --resume for a matching resumable export')
        if not config_path.exists():raise ValueError('Existing directory is not a resumable export; choose a new output directory')
        if json.loads(config_path.read_text())!=identity:
            raise ValueError('Resume configuration mismatch (checkpoint, split, seed, subset policy or raw files changed); use a new output directory')
    else:
        args.output.mkdir(parents=True)
        atomic_json(config_path,identity)
    records_dir=args.output/'samples';records_dir.mkdir(exist_ok=True)
    split.to_csv(args.output/'sample_roles.csv',index=False)
    device=device_for(args.device)
    model=TripletEncoder(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'],strict=True);model.to(device).eval()
    size=checkpoint['training_config']['subset_size'];rows=[[],[]];qc=[]
    started=time.monotonic();total=len(split)
    print(f'EXPORT START: {total} samples; device={device}; max_subsets={args.max_subsets}; resume={args.resume}',flush=True)
    atomic_json(args.output/'progress.json',{'state':'running','completed':0,'total':total})
    with torch.inference_mode():
        for number,(sample,role) in enumerate(split[['spygen_code','set']].itertuples(index=False,name=None),1):
            path=records_dir/(hashlib.sha256(sample.encode()).hexdigest()+'.json')
            if path.exists():
                record=json.loads(path.read_text())
                if record['Sample']!=sample or record['source_signature']!=identity['sample_sources'][sample]:
                    raise ValueError(f'Invalid saved sample record: {sample}')
                print(f'[{number}/{total}] SKIP {sample}: already completed',flush=True)
            else:
                print(f'[{number}/{total}] START {sample} ({role}): loading/tokenizing reads',flush=True)
                sample_started=time.monotonic()
                arr,info=tokenize_file(files[sample],args.cache_dir,checkpoint['model_config']['seq_len'])
                if file_signature(files[sample])!=identity['sample_sources'][sample]:raise ValueError(f'{sample}: raw file changed during export')
                seed=int.from_bytes(hashlib.sha256(f'{args.seed}:{sample}'.encode()).digest()[:8],'little')
                indices=np.random.default_rng(seed).permutation(len(arr))
                use=len(arr) if args.max_subsets is None else min(len(arr),size*args.max_subsets)
                n_subsets=(use+size-1)//size;stats=[[],[]]
                print(f'[{number}/{total}] {sample}: {len(arr)} valid pairs; {n_subsets} subsets',flush=True)
                for subset,start in enumerate(range(0,use,size),1):
                    batch=np.asarray(arr[indices[start:min(start+size,use)]],dtype='int64')
                    embeddings=model(torch.tensor(batch[None],device=device))
                    for block,value in zip(stats,embeddings):block.append(value[0].cpu().numpy())
                    if subset%args.log_every_subsets==0 or subset==n_subsets:
                        print(f'[{number}/{total}] {sample}: subset {subset}/{n_subsets}',flush=True)
                sample_rows=[]
                for block in stats:
                    values=np.stack(block)
                    mean=values.mean(0);std=values.std(0,ddof=0)
                    sample_rows.append({'Sample':sample,'role':'train' if role=='train' else 'outer_eval',**{f'Dim{i+1}':float(v) for i,v in enumerate(mean)},**{f'Standard_dev_{i+1}':float(v) for i,v in enumerate(std)}})
                sample_qc={'Sample':sample,'role':role,'valid_pairs':len(arr),'used_pairs':use,'subsets':n_subsets,'leftover_subset_size':use%size,'discarded_invalid_pairs':info['discarded_pairs'],'source':info['source'],'normalized_valid_pairs_sha256':info['normalized_valid_pairs_sha256']}
                record={'Sample':sample,'source_signature':identity['sample_sources'][sample],'rows':sample_rows,'qc':sample_qc}
                atomic_json(path,record)
                print(f'[{number}/{total}] SAVED {sample}: {time.monotonic()-sample_started:.1f}s',flush=True)
            for target,row in zip(rows,record['rows']):target.append(row)
            qc.append(record['qc'])
            atomic_json(args.output/'progress.json',{'state':'running','completed':number,'total':total,'last_sample':sample,'elapsed_this_job_seconds':time.monotonic()-started})
    outputs=[args.output/f'embeddings_projector_{checkpoint["model_config"]["sample_emb_dim"]}D.csv',args.output/f'embeddings_preprojector_{checkpoint["model_config"]["sample_repr_dim"]}D.csv',args.output/'embedding_qc.csv']
    for values,path in zip([*rows,qc],outputs):
        temp=path.with_suffix('.csv.partial');pd.DataFrame(values).to_csv(temp,index=False);temp.replace(path)
    atomic_json(args.output/'progress.json',{'state':'complete','completed':total,'total':total,'elapsed_this_job_seconds':time.monotonic()-started})
    print(f'Export complete: {args.output}',flush=True)
