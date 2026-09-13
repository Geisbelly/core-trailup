"""Avaliar resposta discursiva por GRAFO de relacao entre resposta e gabarito.

Cada texto vira um grafo: conceitos (palavras de conteudo) sao nos, e uma aresta
liga dois conceitos que aparecem proximos. Comparar resposta com gabarito deixa
de ser "tem as mesmas palavras" e passa a ser "liga as mesmas coisas".

Camada semantica: LSA treinada no proprio corpus casa conceitos que nao sao a
mesma palavra ("Zentralbank" x "Notenbank"), sem baixar modelo.
"""
import os, re, numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
HERE=os.path.dirname(os.path.abspath(__file__))
d=pd.read_excel('/Users/user/Downloads/Inicio/19467052/GPT4FeedbackIncreasesStudentActivation.xlsx',sheet_name='Data')
d=d.rename(columns={'Sample Solution':'gab','Student Answer':'resp','Question':'q',
                    'Content_Mean_Run1_2_3':'nota','Text Length':'len'})
d['aluno']=d['Color (Pseudonym)'].astype(str)+'_'+d['Animal (Pseudonym)'].astype(str)
d=d.dropna(subset=['resp','gab','nota'])
print(f'{len(d):,} respostas | {d.aluno.nunique()} alunos | {d.Task.nunique()} tarefas')
print(f'nota: media {d.nota.mean():.2f} | distribuicao {d.nota.round().value_counts().sort_index().to_dict()}\n')

TOK=re.compile(r'[A-Za-zÄÖÜäöüß]{3,}')
def toks(t): return [w.lower() for w in TOK.findall(str(t))]
# stopwords do proprio corpus (sem depender de lista de idioma)
todos=Counter(w for t in pd.concat([d.resp,d.gab,d.q]) for w in toks(t))
STOP={w for w,_ in todos.most_common(60)}
print(f'{len(todos):,} tokens distintos | stopwords = 60 mais frequentes (ex: {sorted(list(STOP))[:8]})')

def grafo(t,jan=4):
    """nos = conceitos; arestas = conceitos que coocorrem numa janela de `jan`."""
    ws=[w for w in toks(t) if w not in STOP]
    nos=Counter(ws); ar=Counter()
    for i,a in enumerate(ws):
        for b in ws[i+1:i+jan]:
            if a!=b: ar[(min(a,b),max(a,b))]+=1
    return nos,ar

# --- camada semantica: LSA no corpus
corpus=list(pd.concat([d.resp,d.gab,d.q]).astype(str))
tf=TfidfVectorizer(analyzer='word',token_pattern=r'[A-Za-zÄÖÜäöüß]{3,}',min_df=3,sublinear_tf=True)
M=tf.fit_transform(corpus)
svd=TruncatedSVD(n_components=150,random_state=0).fit(M)
termos=np.array(tf.get_feature_names_out())
EMB=normalize(svd.components_.T)          # vetor por termo
IDX={t:i for i,t in enumerate(termos)}
print(f'LSA: {M.shape[1]:,} termos -> 150 dimensoes')

def sem_cov(nos_a,nos_b):
    """cobertura semantica: cada no do gabarito acha um parecido na resposta?"""
    ia=[IDX[w] for w in nos_a if w in IDX]; ib=[IDX[w] for w in nos_b if w in IDX]
    if not ia or not ib: return 0.0,0.0
    S=EMB[ib]@EMB[ia].T
    return float(S.max(axis=1).mean()), float((S.max(axis=1)>0.5).mean())

def feats(r,g,q):
    nr,ar=grafo(r); ng,ag=grafo(g); nq,_=grafo(q)
    sr,sg=set(nr),set(ng)
    er,eg=set(ar),set(ag)
    inter_n=sr&sg; inter_e=er&eg
    # peso: conceitos do gabarito ponderados por frequencia nele
    peso=sum(ng[w] for w in inter_n)/max(sum(ng.values()),1)
    # aresta ponderada
    pe=sum(min(ar[e],ag[e]) for e in inter_e)/max(sum(ag.values()),1)
    # centralidade: grau no grafo do gabarito
    grau=defaultdict(int)
    for (a,b),w in ag.items(): grau[a]+=w; grau[b]+=w
    top=[w for w,_ in sorted(grau.items(),key=lambda x:-x[1])[:10]]
    cob_top=np.mean([w in sr for w in top]) if top else 0.0
    sc_med,sc_frac=sem_cov(sr,sg)
    # o que a resposta tem que NAO esta no gabarito nem na pergunta (divagacao)
    fora=sr-sg-set(nq)
    return {
        'no_jaccard':len(inter_n)/max(len(sr|sg),1),
        'no_cobertura':len(inter_n)/max(len(sg),1),
        'no_peso':peso,
        'aresta_jaccard':len(inter_e)/max(len(er|eg),1),
        'aresta_cobertura':len(inter_e)/max(len(eg),1),
        'aresta_peso':pe,
        'cob_conceitos_centrais':cob_top,
        'sem_media':sc_med,'sem_frac':sc_frac,
        'divagacao':len(fora)/max(len(sr),1),
        'razao_tam':len(sr)/max(len(sg),1),
        'n_nos':len(sr),'n_arestas':len(er),
        'densidade':len(er)/max(len(sr),1),
    }
print('extraindo features de grafo...')
X=pd.DataFrame([feats(r,g,q) for r,g,q in zip(d.resp,d.gab,d.q)],index=d.index)
X['log_len']=np.log1p(d['len'].fillna(d.resp.astype(str).str.len()))
d=pd.concat([d.drop(columns=['len']),X],axis=1)
d.to_parquet(f'{HERE}/classex_grafo.parquet')
print(f'-> {X.shape[1]} features\n')
print('=== correlacao de cada feature com a nota (Spearman) ===')
for c in X.columns:
    print(f'  {c:>24}: {d[c].corr(d.nota,method="spearman"):+.3f}')
