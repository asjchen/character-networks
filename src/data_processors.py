# Processing Movie Lines Data

import ast
import argparse
import csv
import os
import re
import snap

movies_filename = 'movie_titles_metadata.tsv'
characters_filename = 'movie_characters_metadata.tsv'
conversations_filename = 'movie_conversations.tsv'
lines_filename = 'movie_lines.tsv'

def filter_quote_marks(row):
    return [re.sub('\"', '', entry) for entry in row]

class Character:
    def __init__(self, raw_row):
        self.id = raw_row[0]
        assert self.id[0] == 'u'
        self.name = raw_row[1]
        self.movie_id = raw_row[2]
        self.gender = raw_row[4]
        self.credits_position = int(raw_row[5]) if raw_row[5] != '?' else None

class Line:
    def __init__(self, raw_row, characters):
        self.id = raw_row[0]
        assert self.id[0] == 'L'
        self.character = characters[raw_row[1]]
        self.movie_id = raw_row[2]
        self.text = raw_row[4]

class Conversation:
    def __init__(self, raw_row, characters, lines):
        self.characters = set([characters[raw_row[0]], characters[raw_row[1]]])
        self.movie_id = raw_row[2]
        self.lines = [lines[line_id] for line_id in ast.literal_eval(re.sub('\' \'', '\',\'', raw_row[3]))]
        # print [line.text for line in self.lines]

class Movie:
    def __init__(self, raw_row, movie_to_characters, movie_to_conversations):
        self.id = raw_row[0]
        assert self.id[0] == 'm'
        self.name = raw_row[1]
        self.year = int(re.sub(r'[^0-9]', '', raw_row[2]))
        self.imdb_rating = float(raw_row[3])
        self.imdb_votes = int(raw_row[4])
        self.genres = ast.literal_eval(raw_row[5])
        self.characters = movie_to_characters[self.id]
        self.conversations = movie_to_characters[self.id]


def main():
    parser = argparse.ArgumentParser(
        description='Reads and processes the movie dialog data in a given directory')
    parser.add_argument('data_dir', help='Directory containing the dialog data')
    args = parser.parse_args()

    movie_to_characters = {}
    id_to_character = {}
    with open(os.path.join(args.data_dir, characters_filename)) as char_file:
        for raw_row in char_file.readlines():
            row = filter_quote_marks(raw_row.strip().split('\t'))
            if len(row) < 6:
                continue
            if len(row) > 6:
                row = [row[0], '\t'.join(row[1: -4])] + row[-4: ]
            character = Character(row)
            if character.movie_id not in movie_to_characters:
                movie_to_characters[character.movie_id] = []
            movie_to_characters[character.movie_id].append(character)
            id_to_character[character.id] = character

    lines = {}
    with open(os.path.join(args.data_dir, lines_filename)) as lines_file:
        for raw_row in lines_file.readlines():
            row = filter_quote_marks(raw_row.strip().split('\t'))
            if len(row) < 5:
                row += [''] * (5 - len(row))
            line = Line(row, id_to_character)
            lines[line.id] = line

    movie_to_conversations = {}
    with open(os.path.join(args.data_dir, conversations_filename)) as conv_file:
        for raw_row in conv_file.readlines():
            row = filter_quote_marks(raw_row.strip().split('\t'))
            if len(row) < 4:
                continue
            conv = Conversation(row, id_to_character, lines)
            if conv.movie_id not in movie_to_conversations:
                movie_to_conversations[conv.movie_id] = []
            movie_to_conversations[conv.movie_id].append(conv)

    movies = {}
    with open(os.path.join(args.data_dir, movies_filename)) as movies_file:
        for raw_row in movies_file.readlines():
            row = filter_quote_marks(raw_row.strip().split('\t'))
            if len(row) < 6:
                continue
            mov = Movie(row, movie_to_characters, movie_to_conversations)
            movies[mov.id] = mov


if __name__ == '__main__':
    main()
