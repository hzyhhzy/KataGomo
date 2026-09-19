"""Small real-CUDA match regression cases; SGFs verify exact starts and colors."""
import collections
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--engine',required=True)
parser.add_argument('--config',required=True,help='Two-model match config; model paths relative to its directory')
args=parser.parse_args()
CONFIG=Path(args.config).resolve()
ROOT=CONFIG.parent
BIN=Path(args.engine).resolve()
BASE=CONFIG.read_text()
BASE_VALUES={}
for line in BASE.splitlines():
    line=line.split('#',1)[0].strip()
    if '=' in line:
        key,value=line.split('=',1)
        BASE_VALUES[key.strip()]=value.strip()
BOT_NAMES=[BASE_VALUES[f'botName{i}'] for i in range(2)]
MOVES = ['H4 J3 L5 J5 H5', 'A6 B6 A8 A4 A7']

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def config(overrides):
    values = dict(BASE_VALUES)
    values.update(numGameThreads='3',maxPlayouts0='1',maxPlayouts1='1',
                  numNNServerThreadsPerModel='1',nnCacheSizePowerOfTwo='12',
                  nnMutexPoolSizePowerOfTwo='10',maxMovesPerGame='1',
                  numGamesTotal='5',logGamesEvery='1')
    for i in range(2):
        values[f'nnModelFile{i}'] = str(ROOT / values[f'nnModelFile{i}'])
    values['matchOpeningFile'] = str(ROOT / values['matchOpeningFile'])
    values.update(overrides)
    return '\n'.join(k+' = '+str(v) for k,v in values.items() if v is not None)+'\n'

work = Path(tempfile.mkdtemp(prefix='match-integration-',dir=ROOT))
library = work/'two.txt'
library.write_text('\n'.join('15 '+m for m in MOVES)+'\n')
tests = []

def run(name, overrides, expect=None):
    directory = work/name
    directory.mkdir()
    cfg = directory/'test.cfg'
    cfg.write_text(config(overrides))
    result = subprocess.run([str(BIN),'match','-config',str(cfg),'-log-file',str(directory/'engine.log'),
                             '-sgf-output-dir',str(directory/'sgfs')], cwd=directory,
                            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
    (directory/'stdout.log').write_text(result.stdout)
    if expect is not None:
        require(result.returncode != 0,name+' did not fail')
        require(expect in result.stdout,name+' missing expected error: '+result.stdout[-1500:])
        require('Loaded neural net' not in result.stdout,name+' loaded models before failing')
    else:
        require(result.returncode == 0,name+' failed: '+result.stdout[-2000:])
    tests.append(name)
    print('PASS',name,flush=True)
    return directory

for name,text,error in [
    ('empty_file','# only comments\n','contains no openings'),
    ('bad_coordinate','15 A1 A1\n','occupied coordinate'),
    ('forbidden','15 G8 A1 J8 C1 H7 E1 H9 G1 H8\n','illegal move'),
    ('finished','15 A1 A2 B1 B2 C1 C2 D1 D2 E1\n','finished game'),
    ('wrong_size','13 A1 B1\n','board size is not in bSizes'),
]:
    bad=work/(name+'.txt'); bad.write_text(text)
    run(name,{'matchOpeningFile':str(bad)},error)
run('missing_file',{'matchOpeningFile':str(work/'absent.txt')},'Could not open matchOpeningFile')
run('three_bots',{'numBots':'3','matchOpeningFile':str(library)},'exactly two bots')
run('excluded_bot',{'includeBots':'0','matchOpeningFile':str(library)},'requires both bots')
run('conflicting_pairing',{'blackPriority0':'1','matchOpeningFile':str(library)},'cannot be combined')

def sgf_sequence(moves):
    columns='ABCDEFGHJKLMNOPQRST'
    return [(('B' if i%2==0 else 'W'),chr(97+columns.index(m[0]))+chr(97+15-int(m[1:])))
            for i,m in enumerate(moves.split())]

for n in (1,2,3,5,7):
    overrides={'matchOpeningFile':str(library),'numGamesTotal':str(n)}
    if n==5:
        overrides.update(initGamesWithPolicy='true',policyInitAvgMoveNum='10')
    directory=run('paired_'+str(n),overrides)
    observed=collections.Counter()
    for path in (directory/'sgfs').glob('*.sgfs'):
        for sgf in path.read_text().splitlines():
            seq=re.findall(r';([BW])\[([a-z]*)\]',sgf)
            candidates=[i for i,m in enumerate(MOVES) if seq[:5]==sgf_sequence(m)]
            require(len(candidates)==1,'SGF opening modified: '+sgf[:500])
            require('startTurnIdx=5,' in sgf and 'usedInitialPosition=1' in sgf,'missing exact start marker')
            require(len(seq)==6,'unexpected number of generated moves')
            b=re.search(r'PB\[([^]]+)\]',sgf).group(1)
            w=re.search(r'PW\[([^]]+)\]',sgf).group(1)
            require({b,w}==set(BOT_NAMES),'unexpected bot colors')
            observed[candidates[0],BOT_NAMES.index(b)]+=1
    expected=collections.Counter(((i//2)%2,i%2) for i in range(n))
    require(observed==expected,f'{n}: expected {expected}, got {observed}')
    report=json.loads(next((directory/'matchresult').glob('*.json')).read_text())
    require(report['total']==n,'wrong game count')

# Omitted/explicitly empty keeps the old random-opening path.
for name,value in [('empty_setting',''),('quoted_empty_setting','""'),('omitted_setting',None)]:
    directory=run(name,{'matchOpeningFile':value,'numGamesTotal':'1'})
    for path in (directory/'sgfs').glob('*.sgfs'):
        require('usedInitialPosition=1' not in path.read_text(),'empty setting used fixed opening')
# Both bots may share a model: result JSON must not index past deduplicated models.
run('shared_model',{'matchOpeningFile':str(library),'numGamesTotal':'2',
                   'nnModelFile1':str(ROOT/BASE_VALUES['nnModelFile0'])})
(work/'summary.json').write_text(json.dumps({'status':'PASS','tests':tests},indent=2))
print('ALL INTEGRATION TESTS PASS',work,flush=True)
