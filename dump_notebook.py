import json

with open('wdbc_boosting_research.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('Total cells:', len(nb['cells']))
with open('notebook_dump.txt', 'w', encoding='utf-8') as out_f:
    for i, cell in enumerate(nb['cells']):
        out_f.write(f"=== Cell {i}: {cell.get('cell_type')} ===\n")
        source = ''.join(cell.get('source', []))
        out_f.write(source + '\n\n')
        out_f.write('--- Outputs: ---\n')
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'stream':
                out_f.write('STREAM:\n' + ''.join(out.get('text', [])) + '\n')
            elif out.get('output_type') in ['execute_result', 'display_data']:
                data = out.get('data', {})
                for k in data.keys():
                    if k == 'text/plain':
                        out_f.write(f"PLAIN ({out.get('output_type')}):\n" + ''.join(data[k]) + '\n')
                    elif k == 'image/png':
                        out_f.write(f"IMAGE ({out.get('output_type')})\n")
        out_f.write('===============================================\n\n')

print('Dumped to notebook_dump.txt successfully')
