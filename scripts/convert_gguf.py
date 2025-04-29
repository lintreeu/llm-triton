#!/usr/bin/env python
from pathlib import Path
import json, torch, argparse
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('json_out')
    args=ap.parse_args()
    cfg={'n_layers':1,'n_heads':2,'n_kv_heads':2,'hidden_size':32,'intermediate_size':64,
         'vocab_size':32,'max_position_embeddings':64}
    Path(args.json_out).write_text(json.dumps(cfg,indent=2))
if __name__=='__main__':
    main()
