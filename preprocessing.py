from os import path, walk, makedirs, chdir
import re

PUNCTUATION = {'?', '!', '..', '...'}


def replace(match):
    if match[0] in PUNCTUATION:
        return '.'
    return ' '


def process(src_dir):
    corpus_path = path.join(src_dir, 'corpus')
    processed_corpus_path = path.join(src_dir, 'processed_corpus')

    if not path.exists(processed_corpus_path):
        makedirs(processed_corpus_path)

    for (dir_path, dir_names, file_names) in walk(corpus_path):
        if dir_names:
            for dir_name in dir_names:
                author_dir_path = path.join(processed_corpus_path, dir_name)
                if not path.exists(author_dir_path):
                    makedirs(author_dir_path)

        if file_names:
            for file_name in file_names:
                chdir(dir_path)
                with open(file_name, 'r', encoding='cp1251') as file_handler:
                    book = file_handler.read()
                book = re.sub(r'[XIV\r\t\n_"\']| -|- |[?!]|\.\.\.|\.\.', repl=replace, string=book).lower()

                author_dir_name = path.basename(dir_path)
                author_processed_dir_path = path.join(processed_corpus_path, author_dir_name)
                chdir(author_processed_dir_path)
                with open(file_name, 'w') as file_handler:
                    file_handler.write(book)


if __name__ == '__main__':
    SRC_DIR = path.dirname(path.realpath(__file__))
    process(SRC_DIR)

