#!/usr/bin/env python3
import os, sys, json, pathlib

def main(root, file='result_printout.txt', n=4*6):
    # Get last N subfolders by name (desc)
    subs = sorted((d for d in os.listdir(root) if os.path.isdir(p := os.path.join(root, d))), reverse=True)[:n]
    subs = sorted(subs)
    out = {}
    
    # print(subs)

    for sub in subs:
        path = os.path.join(root, sub, file)
        if os.path.isfile(path):
            with open(path) as f:
                values = [line for line in f.read().strip().split('\n') if ':' in line or '|' in line]
                # for val in values:
                #     print(val)
                #     print(val.split(':', 1))
                data = dict(line.split(':', 1) for line in values if ':' in line)
                out_format = [line for line in values if '|' in line]
                out_format = out_format[0].split('|')
            out[sub] = dict([(val[0].strip(), val[1].strip()) for val in data.items()])
            out[sub]['format'] = [line.strip() for line in out_format]
    
    # Save compact output
    # pathlib.Path('parsed.json').write_text(json.dumps(out, separators=(',', ':')))
    # print(json.dumps(out, separators=(',', ':')))
    # print(out.items()['Accuracy'])
    # print(out)

    def format_print(value, index):
        # description = ['CD', 'C', 'GD', 'G', 'D', 'P']
        prepend = ''
        if 'unique' in value['format']:
            prepend += 'C'
        elif 'major' in value['format']:
            prepend += 'G'
        else:
            prepend += 'P'

        if 'description' in value['format']:
            prepend += 'D'

        # if 'shuffle_0' in value['format']:
        #     prepend += '0'
        # if 'shuffle_1' in value['format']:
        #     prepend += '1'
        # if 'shuffle_2' in value['format']:
        #     prepend += '2'
        # if 'shuffle_3' in value['format']:
        #     prepend += '3'
        return f"{prepend} & {float(value['Accuracy']):.3f} & {float(value['Precision']):.3f} & {float(value['Recall']):.3f} & {float(value['F1 Macro']):.3f} & {float(value['F1 Weighted']):.3f} \\\\"

    for index, value in enumerate(out):
        print(format_print(out[value], index // 4))
        # print(out[value]['Accuracy'])

    print(f"Done. Parsed {len(out)} folders → parsed.json")

if __name__ == '__main__':
    main('./mangles/')