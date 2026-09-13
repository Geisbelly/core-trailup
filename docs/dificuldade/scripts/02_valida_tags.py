"""O que sobrou no residuo corresponde a ASSUNTO?

Se questoes que covariam depois de remover habilidade compartilham tags, a
estrutura e real e interpretavel. Se nao, e ruido correlacionado.
"""
import os, numpy as np, pandas as pd
from sklearn.cluster import HDBSCAN, SpectralClustering
from sklearn.metrics import adjusted_mutual_info_score
HERE=os.path.dirname(os.path.abspath(__file__))
C=np.load(f'{HERE}/C.npy'); qs=np.load(f'{HERE}/qids.npy')
meta=pd.read_csv('/Users/user/Downloads/Inicio/EdNet-Contents/contents/questions.csv')
meta['qidx']=meta.question_id.str[1:].astype(np.int32)
meta=meta.set_index('qidx').loc[qs]
part=meta.part.values
tags=[set(str(t).split(';'))-{'-1',''} for t in meta.tags]
print(f'{len(qs)} questoes | {len(set(part))} parts | '
      f'{len(set(t for s in tags for t in s))} tags distintas\n')

print('=== questoes que covariam compartilham tag/part? ===')
n=len(C); iu=np.triu_indices(n,1)
r=C[iu]
mesma_part=np.array([part[i]==part[j] for i,j in zip(*iu)])
jac=np.array([len(tags[i]&tags[j])/max(len(tags[i]|tags[j]),1) for i,j in zip(*iu)])
print(f'  correlacao media, MESMA part : {r[mesma_part].mean():+.4f}  ({mesma_part.sum():,} pares)')
print(f'  correlacao media, OUTRA part : {r[~mesma_part].mean():+.4f}  ({(~mesma_part).sum():,} pares)')
print(f'  diferenca: {r[mesma_part].mean()-r[~mesma_part].mean():+.4f}')
for lo,hi,lab in [(0,0.01,'nenhuma tag'),(0.01,0.34,'ate 1/3'),(0.34,1.01,'mais de 1/3')]:
    m=(jac>=lo)&(jac<hi)
    if m.sum()<100: continue
    print(f'  correlacao media, tags em comum {lab:>12}: {r[m].mean():+.4f} ({m.sum():>7,} pares)')
print(f'  correlacao de r com sobreposicao de tags: {np.corrcoef(r,jac)[0,1]:+.4f}')

print('\n=== agrupando pelo padrao, os grupos batem com as tags? ===')
D=np.clip(1-C,0,2)
for nome,lab in [('HDBSCAN (descobre k)',HDBSCAN(min_cluster_size=20,metric='precomputed').fit(D.astype(np.float64)).labels_)]:
    k=len(set(lab))-(1 if -1 in lab else 0)
    m=lab!=-1
    ami_p=adjusted_mutual_info_score(part[m],lab[m]) if k>1 else float('nan')
    print(f'  {nome}: {k} grupos, ruido {(lab==-1).mean():.0%} | AMI com part {ami_p:+.3f}')
for k in (5,10,20):
    sc=SpectralClustering(n_clusters=k,affinity='precomputed',random_state=0,assign_labels='kmeans')
    A=np.clip(C,0,1)
    lab=sc.fit_predict(A)
    ami_p=adjusted_mutual_info_score(part,lab)
    # AMI com a tag dominante
    td=[max(t,key=lambda x:0) if t else '-' for t in tags]
    td=[sorted(t)[0] if t else '-' for t in tags]
    ami_t=adjusted_mutual_info_score(td,lab)
    print(f'  espectral k={k:>2}: AMI com part {ami_p:+.3f} | AMI com tag principal {ami_t:+.3f}')
print('\n  (AMI 0 = nenhuma relacao; 1 = identicos. Referencia: AMI entre part e tag principal =',
      f'{adjusted_mutual_info_score(part,[sorted(t)[0] if t else "-" for t in tags]):+.3f})')
