import os

def main():
    values = [
        # '--overall_suggestion --without_few_shot --whole_test --description --major --unique',
        # '--overall_suggestion --without_few_shot --whole_test --description --unique',
        # '--overall_suggestion --without_few_shot --whole_test --major --unique',
        # '--overall_suggestion --without_few_shot --whole_test --unique',
        '--overall_suggestion --without_few_shot --whole_test --description --major',
        '--overall_suggestion --without_few_shot --whole_test --description',
        '--overall_suggestion --without_few_shot --whole_test --major',
        '--overall_suggestion --without_few_shot --whole_test',
        # '--overall_suggestion --without_few_shot --whole_test --cleaned --description --major --unique',
        # '--overall_suggestion --without_few_shot --whole_test --cleaned --description --unique',
        # '--overall_suggestion --without_few_shot --whole_test --cleaned --major --unique',
        # '--overall_suggestion --without_few_shot --whole_test --cleaned --unique',
    ]

    for value in values:
        # os.system('py test.py')
        for val in ['', '--shuffle_1', '--shuffle_2', '--shuffle_3']:
            os.system(f'py main.py {value} ' + val)
            os.system('py mangler.py')


if __name__ == '__main__':
    main()
