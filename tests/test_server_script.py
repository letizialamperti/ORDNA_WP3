"""Check actual Bash dispatch without requiring OAR, CUDA or a real Conda install."""
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ServerScriptTests(unittest.TestCase):
    def test_config_conda_and_two_stage_dispatch(self):
        root=Path(__file__).resolve().parents[1]
        script=root/'scripts/run_fold.sh'
        subprocess.run(['bash','-n',str(script)],check=True)
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);raw=work/'raw reads';raw.mkdir()
            calls=work/'calls.jsonl';activation=work/'conda activation'
            mock=work/'python mock'
            mock.write_text('#!'+sys.executable+'\nimport json,os,sys\nwith open(os.environ["CALLS"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
            mock.chmod(0o755)
            conda=work/'conda.sh'
            conda.write_text('conda() { printf "%s\\n" "$*" >> "$ACTIVATION"; }\n')
            cfg=work/'server.env'
            settings={'FOLD':'02','RAW_DIR':str(raw),'OUTPUT_ROOT':str(work/'results root'),'CACHE_DIR':str(work/'cache'),'SEQUENCE_LENGTH':'150','STEPS_PER_EPOCH':'2','EPOCHS':'1','CONDA_SH':str(conda),'CONDA_ENV':'ordna-training','PYTHON_BIN':str(mock),'DEVICE':'cuda','SEED':'2','EXPORT_SEED':'0','MAX_EXPORT_SUBSETS':'3'}
            cfg.write_text('\n'.join(k+'='+shlex.quote(v) for k,v in settings.items())+'\n')
            env={**os.environ,'RUN_CONFIG':str(cfg),'CALLS':str(calls),'ACTIVATION':str(activation),'OAR_JOB_ID':'test-job'}
            result=subprocess.run(['bash',str(script)],env=env,cwd=work,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            commands=[json.loads(line) for line in calls.read_text().splitlines()]
            self.assertEqual(len(commands),2)
            self.assertEqual(commands[0][:3],['-m','ordna_training.cli','train'])
            self.assertEqual(commands[1][:3],['-m','ordna_training.cli','export'])
            self.assertEqual(activation.read_text().strip(),'activate ordna-training')
            for command in commands:
                self.assertEqual(command[command.index('--samples-dir')+1],str(raw))
                self.assertTrue(command[command.index('--split')+1].endswith('_fold_02.csv'))
                self.assertEqual(command[command.index('--device')+1],'cuda')
            self.assertTrue(commands[1][commands[1].index('--checkpoint')+1].endswith('/fold_02/training/checkpoints/last.pt'))
            self.assertEqual(commands[1][commands[1].index('--max-subsets')+1],'3')
            # A training error must stop the shell before export.
            calls.unlink()
            mock.write_text('#!'+sys.executable+'\nimport sys\nsys.exit(9)\n')
            failure=subprocess.run(['bash',str(script)],env=env,cwd=work,capture_output=True,text=True)
            self.assertEqual(failure.returncode,9)
            self.assertFalse(calls.exists())


if __name__=='__main__':unittest.main()
