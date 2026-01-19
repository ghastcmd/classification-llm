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

    global previous_current_pattern, previous_stack, current_pattern
    previous_stack = {'CD': [], 'C': [], 'GD': [], 'G': [], 'PD': [], 'P': []}
    previous_current_pattern = ''
    current_pattern = ''

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
        global current_pattern, previous_stack
        current_pattern = prepend
        previous_stack[prepend].append([float(value['Accuracy']), float(value['Precision']), float(value['Recall']), float(value['F1 Macro']), float(value['F1 Weighted'])])
        return f"{prepend} & {float(value['Accuracy']):.3f} & {float(value['Precision']):.3f} & {float(value['Recall']):.3f} & {float(value['F1 Macro']):.3f} & {float(value['F1 Weighted']):.3f} \\\\"

    def get_mean(current_list: list):
        max_size = len(current_list)
        return_value = [0, 0, 0, 0, 0]
        for line in current_list:
            for index, val in enumerate(line):
                return_value[index] += val
        
        for val in return_value:
            val /= max_size
        
        hand = current_list[0]

        global current_pattern
        # print(hand)
        return f'Média & {hand[0]:.3f} & {hand[1]:.3f} & {hand[2]:.3f} & {hand[3]:.3f} & {hand[4]:.3f} \\\\'

    previous_index = 0

    for index, value in enumerate(out):
        print(format_print(out[value], index // 4))
        if previous_index != (index + 1) // 4:
            print(get_mean(previous_stack[current_pattern]))
            previous_index = (index + 1) // 4
        previous_current_pattern = current_pattern

    # print(previous_stack)

    print(f"Done. Parsed {len(out)} folders → parsed.json")

if __name__ == '__main__':
    main('./mangles/')