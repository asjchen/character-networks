# Classes and functions for processing the movie dialogue data from the Kaggle
# data set, provided by Cornell University (URL to Kaggle source: 
# https://www.kaggle.com/Cornell-University/movie-dialog-corpus)
# These functions help convert the data into a character network for each movie

# Currently, the directed character networks are based on whether character B
# has ever talked more than character A in a conversation. The multigraph 
# chracter networks represent individual conversations, with one edge between
# characters A and B for each conversation they have had

import ast
import os
import re
import snap

import graph_generators as gg

# Names of the files in the data directory
movies_filename = 'movie_titles_metadata.tsv'
characters_filename = 'movie_characters_metadata.tsv'
conversations_filename = 'movie_conversations.tsv'
lines_filename = 'movie_lines.tsv'

# Filters the quotation marks in the CSV entries
def filter_quote_marks(row):
    return [re.sub('\"', '', entry) for entry in row]

# Represents a character from a certain movie
# The input raw_row is a list of string entries, representing a row in the CSV
# In the object, id, movie_id, and credits_position are integers,
# and the name and gender are strings
class Character:
    def __init__(self, raw_row):
        assert raw_row[0][0] == 'u'
        self.id = int(raw_row[0][1:])
        self.name = raw_row[1]
        self.movie_id = int(raw_row[2][1:])
        self.gender = raw_row[4]
        self.credits_position = int(raw_row[5]) if raw_row[5] != '?' else None

# Represents the line of dialog spoken by a certain character
# The input raw_row is a list of string entries, representing a row in the CSV,
# and characters is a dictionary mapping character ID to Character object
# In the object, id and movie_id are integers, character is a Character object,
# and text is a string
class Line:
    def __init__(self, raw_row, characters):
        assert raw_row[0][0] == 'L'
        self.id = int(raw_row[0][1:])
        self.character = characters[int(raw_row[1][1:])]
        self.movie_id = int(raw_row[2][1:])
        self.text = raw_row[4]

# Represents a conversation between two characters
# The input raw_row is a list of string entries, representing a row in the CSV,
# characters is a dictionary mapping ID to Character object, and lines is a 
# dictionary mapping line ID to Line object
# In the object, characters is a set of two Character objects, 
# movie_id is an integer, and lines is a list of Line objects
class Conversation:
    def __init__(self, raw_row, characters, lines):
        self.characters = set([characters[int(raw_row[0][1:])], 
            characters[int(raw_row[1][1:])]])
        self.movie_id = int(raw_row[2][1:])
        self.lines = [lines[int(line_id[1:])] for line_id \
            in ast.literal_eval(re.sub('\' \'', '\',\'', raw_row[3]))]

# Represents a movie
# The input raw_row is a list of string entries, representing a row in the CSV,
# movie_to_characters is a dictionary mapping movie ID to list of Character 
# objects, movie_to_conversations is a dictionary mapping mapping movie ID to
# list of Conversation objects
# In the object, id, year, and imdb_votes are integers, name is a 
# string, imdb_rating is a float, genres is a list of strings, characters is a
# list of Character objects, and conversations is a list of Conversation 
# objects
class Movie:
    def __init__(self, raw_row, movie_to_characters, movie_to_conversations):
        assert raw_row[0][0] == 'm'
        self.id = int(raw_row[0][1:])
        self.name = raw_row[1]
        self.year = int(re.sub(r'[^0-9]', '', raw_row[2]))
        self.imdb_rating = float(raw_row[3])
        self.imdb_votes = int(raw_row[4])
        self.genres = ast.literal_eval(re.sub('\' \'', '\',\'', raw_row[5]))
        self.characters = movie_to_characters[self.id]
        self.conversations = movie_to_conversations[self.id]

# Converts a Movie object into a directed graph (TNGraph) character network.
# It creates an edge from character A to character B if there is a conversation
# between them in which B speaks more words than A
# (TODO: find a better asymmetric metric for measuring conversation dynamics)
def graph_talks_more_words(movie):
    graph = snap.PNGraph.New()
    for ch in movie.characters:
        graph.AddNode(ch.id)
    for conv in movie.conversations:
        word_counts = {ch.id: 0 for ch in conv.characters}
        for line in conv.lines:
            word_counts[line.character.id] += len(line.text.split())
        sorted_char_ids = sorted(word_counts.keys(), 
            key=lambda ch_id: word_counts[ch_id])
        graph.AddEdge(sorted_char_ids[0], sorted_char_ids[1])
    return graph

# Converts a Movie object into an undirected multigraph (TNEANet with every 
# edge reciprocated) character network. It creates an edge between characters
# A and B for every conversation that they have with each other (so multiple 
# edges can exist between two characters)
def graph_conversations_undirected(movie):
    graph = snap.TNEANet.New()
    for ch in movie.characters:
        graph.AddNode(ch.id)
    for conv in movie.conversations:
        chars = list(conv.characters)
        graph.AddEdge(chars[0].id, chars[1].id)
        graph.AddEdge(chars[1].id, chars[0].id)
    return graph

# Takes the path of the data directory and a graph class [DirectedGraphModel 
# and UndirectedMultiGraphModel] and produces two dictionaries: movies maps
# ID to Movie object and movie_networks maps ID to graph_class object
def get_movie_networks(data_dir, graph_class):
    movie_to_characters = {}
    id_to_character = {}

    # Note: most entries are separated by the tab character
    with open(os.path.join(data_dir, characters_filename)) as char_file:
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
    with open(os.path.join(data_dir, lines_filename)) as lines_file:
        for raw_row in lines_file.readlines():
            row = filter_quote_marks(raw_row.strip().split('\t'))
            if len(row) < 5:
                row += [''] * (5 - len(row))
            line = Line(row, id_to_character)
            lines[line.id] = line

    movie_to_conversations = {}
    with open(os.path.join(data_dir, conversations_filename)) as conv_file:
        for raw_row in conv_file.readlines():
            row = filter_quote_marks(raw_row.strip().split('\t'))
            if len(row) < 4:
                continue
            conv = Conversation(row, id_to_character, lines)
            if conv.movie_id not in movie_to_conversations:
                movie_to_conversations[conv.movie_id] = []
            movie_to_conversations[conv.movie_id].append(conv)

    movies = {}
    with open(os.path.join(data_dir, movies_filename)) as movies_file:
        for raw_row in movies_file.readlines():
            row = filter_quote_marks(raw_row.strip().split('\t'))
            if len(row) < 6:
                continue
            mov = Movie(row, movie_to_characters, movie_to_conversations)
            movies[mov.id] = mov

    movie_networks = {}
    if graph_class == gg.DirectedGraphModel:
        for movie_id in movies:
            movie_networks[movie_id] = graph_talks_more_words(movies[movie_id])
    elif graph_class == gg.UndirectedMultiGraphModel:
        for movie_id in movies:
            movie_networks[movie_id] = graph_conversations_undirected(
                movies[movie_id])
    else:
        raise Exception(('graph_class parameter must be either '
            'DirectedGraphModel or UndirectedMultiGraphModel'))

    return movies, movie_networks


