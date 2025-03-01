from datetime import datetime
import os
import shutil

def make_folder(name):
    os.makedirs(name, exist_ok=True)

def copy2folder(foldername, filename):
    filename = os.path.basename(filename)
    shutil.copy2(filename, os.path.join(foldername, filename))

def main():
    date_now = datetime.now()
    date_now = ''.join(f'{date_now}'.split(':'))
    foldername = f'mangle.{date_now}'
    foldername = os.path.join('mangles', foldername)
    make_folder(foldername)
    
    copy2folder(foldername, './cached.txt')
    copy2folder(foldername, './prompt.txt')
    copy2folder(foldername, './results.txt')
    copy2folder(foldername, './version.txt')
    copy2folder(foldername, './result_printout.txt')
    
if __name__ == '__main__':
    main()