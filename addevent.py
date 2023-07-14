import os
import pandas as pd
import numpy as np
import pickle
import vitaldb
import time
from tqdm import tqdm
from datetime import datetime, timedelta
import ray

vpath = '/workspace/delirium_jmkim'
savepath = '/workspace/delirium_addevent'
os.makedirs(savepath, exist_ok=True)
vlist = os.listdir(vpath)
info = [v for v in vlist if '.vital' not in v][0]
info = pd.read_excel(os.path.join(vpath, info))
vlist = [v for v in vlist if '.vital' in v]
for item in ('ane.start', 'op.start', 'op.end', 'ane.end'):
    info[item] = pd.to_datetime(info[item]) - timedelta(hours=9)
    info[f'{item}.ts'] = info[item].apply(lambda x: time.mktime(x.timetuple()))


ane_trks = ['Orchestra/PPF20_CE', 'Primus/EXP_SEVO', 'Primus/EXP_DES']
bsr_trks = ['Intellivue/EEG_RATIO_SUPPRN', 'BIS/SR', 'ROOT/SR', 'Bx50/BIS_BSR']

@ray.remote
def trackcontrol(n, chunk):
    FILES, BIS, ANES, BSR = [], [], [], []
    NPTRACK = []
    for v in tqdm(chunk):
        vf = vitaldb.VitalFile(os.path.join(vpath, v))
        trks = vf.trks.keys()

        event_recs = []            
        for a in ane_trks:
            if a in trks:
                # Add Event From ane_trks
                record = vf.to_pandas([a], 1, return_datetime=True)
                times = record[record[a] >= 0.2]['Time'] - timedelta(hours=9)
                if len(times) < 2: continue

                start, end = time.mktime(times.iloc[0].timetuple()), time.mktime(times.iloc[-1].timetuple())

                event_recs.append({'dt': start, 'val': 'ane.start'})
                event_recs.append({'dt': end, 'val': 'ane.end'})          
                for item in ('op.start', 'op.end'):
                    event_recs.append({'dt': info[info['fileid'] == v].iloc[0][f'{item}.ts'], 'val': item})  
                break
        
        # Add Event From Excel
        if len(event_recs) == 0:
            # Add Event From Excel
            for item in ('ane.start', 'op.start', 'op.end', 'ane.end'):
                event_recs.append({'dt': info[info['fileid'] == v].iloc[0][f'{item}.ts'], 'val': item})
                
        vf.add_track('EVENT', event_recs, mindisp=0, maxdisp=10)
        vf.to_vital(os.path.join(savepath, v))

        for b in bsr_trks:
            if b in trks:
                event = vf.to_pandas(['EVENT'], 1)
                record = vf.to_pandas(['BIS/BIS', b], 1, return_datetime=True)
                if 'ane.start' in pd.unique(event.dropna()['EVENT']):
                    start = event[event['EVENT'] == 'ane.start'].index[0]
                else:
                    start = 0
                if 'ane.end' in pd.unique(event.dropna()['EVENT']):
                    end = event[event['EVENT'] == 'ane.end'].index[0]
                else:
                    end = event.shape[0] - 1
                
                record = record.loc[start:end][['BIS/BIS', b]]
                record['caseNo'] = info[info['fileid'] == v]['caseNo'].iloc[0]
                NPTRACK.append(record.to_numpy())
                break
        
        FILES.append(v)
        BIS.append('BIS/BIS' in trks)
        ANES.append([a for a in ane_trks if a in trks])
        BSR.append([b for b in bsr_trks if b in trks])
        
    return (FILES, BIS, ANES, BSR, NPTRACK)

ray.init(num_cpus=112)
chunks = np.array_split(vlist, 112)
rayget = ray.get([trackcontrol.remote(n, chunk) for n, chunk in enumerate(chunks)])
ray.shutdown()

chunk, BIS, ANES, BSR, NPTRACK = [], [], [], [], []
for r in rayget:
    chunk += list(r[0])
    BIS += list(r[1])
    ANES += list(r[2])
    BSR += list(r[3])
    NPTRACK += list(r[4])

pd.DataFrame(np.transpose([chunk, BIS, ANES, BSR]), columns=['fileid', 'BIS', 'ANES', 'BSR']).to_csv('res.csv', index=None)
with open('nparray.npy', 'wb') as f:
    np.save(f, np.concatenate(NPTRACK))

