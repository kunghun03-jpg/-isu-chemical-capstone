"""Build audit-only MAT case list; never fabricate RF references or read model weights."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.analyze_fault_signatures import metadata
from scripts.validate_v2_integration import write_csv


def discover(roots):
    selected=[]; excluded=[]; seen=set()
    for root in roots:
        root=Path(root).expanduser().resolve()
        if not root.is_dir(): raise ValueError(f'MAT root missing: {root}')
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix.lower()!='.mat': continue
            if not any(p.startswith('Looseness') for p in path.parts): continue
            resolved=path.resolve()
            if resolved in seen: continue
            seen.add(resolved)
            try:
                m=metadata(path)
                if m['family']!='Looseness': raise ValueError('not_looseness')
                selected.append({'path':str(resolved),'rpm':m['rpm'],'condition':m['condition'],'date':m['date'],'raw_prediction':''})
            except ValueError as error:
                excluded.append({'path':str(path),'reason':str(error)})
    return selected,excluded


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mat-root',action='append',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output).expanduser().resolve()
    if out==ROOT or ROOT in out.parents: raise ValueError('output must be outside repository')
    rows,bad=discover(a.mat_root);out.mkdir(parents=True)
    write_csv(out/'looseness_cases.csv',rows);write_csv(out/'excluded_files.csv',bad)
    print(f'Listed {len(rows)} files; excluded {len(bad)}. No MAT inference performed.')

if __name__=='__main__': main()
