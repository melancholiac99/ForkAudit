import os,sys
from common import ROOT,WORK
from rres_chinese import OUT
from rres_summary import scores
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from launch import train,evaluate
assert (OUT/'residual_token_kl.json').is_file() and (OUT/'han.json').is_file()
train(ROOT,{**os.environ,'OPSD_RRES_PROJECTOR':str(WORK/'projector.pt')},'Rres')
for bench in ('aime24','aime25','hmmt25'):
    evaluate(ROOT,os.environ.copy(),'Rres@100',bench,1001)
scores()
