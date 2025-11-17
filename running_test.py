import os

def main():
    values = [
        '--overall_suggestion --without_few_shot --whole_test --description --unique', # characteristic, description
        '--overall_suggestion --without_few_shot --whole_test --unique', # characteristic
        '--overall_suggestion --without_few_shot --whole_test --description --major', # general, description
        '--overall_suggestion --without_few_shot --whole_test --major', # general
        '--overall_suggestion --without_few_shot --whole_test --description', # description
        '--overall_suggestion --without_few_shot --whole_test', # default
    ]

    for value in values:
        # os.system('py test.py')
        for val in ['', '--shuffle_1', '--shuffle_2', '--shuffle_3']:
            os.system(f'py main.py {value} ' + val)
            os.system('py mangler.py')


if __name__ == '__main__':
    main()
