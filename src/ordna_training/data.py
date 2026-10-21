"""Label-free roles and deterministic raw-read token caches (no learned transform)."""
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Same IUPAC token order used by upstream Bio.Data.IUPACData, then U; padding=0.
TOKEN_MAP={base:i+1 for i,base in enumerate('gatcrywsmkhbvdnu')}


class InsufficientReadsError(ValueError):
    def __init__(self,path,info):
        super().__init__(f'{path}: no valid paired reads')
        self.info=info


def read_split(path):
    d=pd.read_csv(path,dtype=str)
    if set(d.columns)!={'spygen_code','set'}:
        raise ValueError('Split must have exactly spygen_code,set columns; labels are not accepted')
    if d.isna().any().any() or not d.spygen_code.is_unique:
        raise ValueError('Missing or duplicate sample IDs/roles')
    if set(d['set'])!={'train','validation'}:
        raise ValueError('Expected both train and validation roles')
    return d


def locate_files(root,ids):
    expected=set(ids);files={}
    for path in sorted(Path(root).rglob('*.csv')):
        if path.stem not in expected:continue
        if path.stem in files:raise ValueError(f'Duplicate raw sample file: {path.stem}')
        files[path.stem]=path.resolve()
    missing=sorted(expected-set(files))
    if missing:raise ValueError(f'Missing raw samples ({len(missing)}): {missing}')
    return files


def valid_pairs(path,seq_len,stats=None):
    with open(path,newline='') as handle:
        reader=csv.DictReader(handle)
        if reader.fieldnames is None or not {'Forward','Reverse'}<=set(reader.fieldnames):
            raise ValueError(f'{path}: required columns Forward,Reverse')
        for row in reader:
            if stats is not None:stats['original_rows']+=1
            pair=tuple((row.get(k) or '').strip().lower() for k in ['Forward','Reverse'])
            if not all(s and set(s)<=set(TOKEN_MAP) for s in pair):
                if stats is not None:stats['discarded_pairs']+=1
                continue
            if max(map(len,pair))>seq_len:
                raise ValueError(f'{path}: read exceeds sequence_length={seq_len}; no silent truncation')
            yield pair


def tokenize_file(path,cache_dir,seq_len):
    path=Path(path).resolve();cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    stat=path.stat()
    provenance={'source':str(path),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sequence_length':seq_len,'tokens':TOKEN_MAP,'cache_version':1}
    key=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    target=cache_dir/f'{key}.npy';meta=cache_dir/f'{key}.json'
    if target.exists() and meta.exists():return np.load(target,mmap_mode='r'),json.loads(meta.read_text())
    stats={'original_rows':0,'discarded_pairs':0}
    count=sum(1 for _ in valid_pairs(path,seq_len,stats))
    if not count:raise InsufficientReadsError(path,{**provenance,**stats,'valid_pairs':0})
    temp=cache_dir/f'{key}.{os.getpid()}.partial.npy'
    arr=np.lib.format.open_memmap(temp,mode='w+',dtype='uint8',shape=(count,2,seq_len))
    arr[:]=0
    digest=hashlib.sha256()
    for i,pair in enumerate(valid_pairs(path,seq_len)):
        for direction,s in enumerate(pair):arr[i,direction,:len(s)]=[TOKEN_MAP[c] for c in s]
        digest.update((pair[0]+'\t'+pair[1]+'\n').encode())
    arr.flush();del arr
    current=path.stat()
    if current.st_size!=stat.st_size or current.st_mtime_ns!=stat.st_mtime_ns:
        temp.unlink();raise ValueError(f'{path}: changed while caching')
    temp.replace(target)
    info={**provenance,**stats,'valid_pairs':count,'normalized_valid_pairs_sha256':digest.hexdigest()}
    meta.write_text(json.dumps(info,indent=2))
    return np.load(target,mmap_mode='r'),info


def sample_triplet(arrays,subset_size,rng):
    if len(arrays)<2:raise ValueError('Need at least two training samples')
    a=int(rng.integers(len(arrays)))
    n=int(rng.integers(len(arrays)-1));n+=n>=a
    indices=rng.choice(len(arrays[a]),2*subset_size,replace=False)
    neg=rng.choice(len(arrays[n]),subset_size,replace=False)
    return (np.asarray(arrays[a][indices[:subset_size]],dtype='int64'),
            np.asarray(arrays[a][indices[subset_size:]],dtype='int64'),
            np.asarray(arrays[n][neg],dtype='int64'),a,n)
