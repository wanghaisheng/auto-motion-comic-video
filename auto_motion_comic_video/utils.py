from .constants import Character
import random
from collections import Counter
import os
import requests
import zipfile
import textwrap

from PIL import Image, ImageDraw, ImageFont , ImageFile


def ensure_assets_are_available():
    if not os.path.exists('./assets'):
        print('Assets not present. Downloading them')
        response = requests.get('https://drive.google.com/file/d/1hQ5MTPxjom_E6mqyJO_ppNrhFp_c4PYx/view?usp=sharing')
        with open('assets.zip', 'wb') as file:
            file.write(response.content)
        with zipfile.ZipFile('assets.zip', 'r') as zip_ref:
            zip_ref.extractall('assets')
        os.remove('assets.zip')

def get_characters(common: Counter):
    users_to_characters = {}
    most_common =  [t[0] for t in common.most_common()]
    # print('most_common',type(most_common),most_common)
    all_rnd_characters = [
        Character.GODOT,
        Character.FRANZISKA,
        Character.JUDGE,
        Character.LARRY,
        Character.MAYA,
        Character.KARMA,
        Character.PAYNE,
        Character.MAGGEY,
        Character.PEARL,
        Character.LOTTA,
        Character.GUMSHOE,
        Character.GROSSBERG,
        Character.APOLLO,
        Character.KLAVIER,
        Character.MIA,
        Character.WILL,
        Character.OLDBAG,
        Character.REDD,
    ]
    rnd_characters = []
    if len(most_common) > 0:
        users_to_characters[most_common[0]] = Character.PHOENIX
        if len(most_common) > 1:
            users_to_characters[most_common[1]] = Character.EDGEWORTH
            for character in most_common[2:]:
                if len(rnd_characters) == 0:
                    rnd_characters = all_rnd_characters.copy()
                rnd_character = random.choice(
                    rnd_characters
                )
                rnd_characters.remove(rnd_character)
                users_to_characters[character] = rnd_character
    return users_to_characters

def get_all_music_available():
    ensure_assets_are_available()
    available_music = os.listdir('./assets/music')
    available_music.append('rnd')
    return available_music

def is_music_available(music: str) -> bool:
    music = music.lower()
    ensure_assets_are_available()
    available_music = os.listdir('./assets/music')
    available_music.append('rnd')
    return music in available_music
def url_ok(url):


    try:
        response = requests.head(url)
    except Exception as e:
        # print(f"NOT OK: {str(e)}")
        return False
    else:
        if response.status_code == 200:
            # print("OK")
            return True
        else:
            print(f"NOT OK: HTTP response code {response.status_code}")

            return False   

abbreviations = {'dr.': 'doctor', 'mr.': 'mister', 'bro.': 'brother', 'bro': 'brother', 'mrs.': 'mistress', 'ms.': 'miss', 'jr.': 'junior', 'sr.': 'senior',
                 'i.e.': 'for example', 'e.g.': 'for example', 'vs.': 'versus'}
terminators = ['.', '!', '?']
wrappers = ['"', "'", ')', ']', '}']


def find_sentences(paragraph):
    end = True
    sentences = []
    while end > -1:
        end = find_sentence_end(paragraph)
        if end > -1:
            sentences.append(paragraph[end:].strip())
            paragraph = paragraph[:end]
    sentences.append(paragraph)
    sentences.reverse()
    return sentences


def find_sentence_end(paragraph):
    [possible_endings, contraction_locations] = [[], []]
    contractions = abbreviations.keys()
    sentence_terminators = terminators + \
        [terminator + wrapper for wrapper in wrappers for terminator in terminators]
    for sentence_terminator in sentence_terminators:
        t_indices = list(find_all(paragraph, sentence_terminator))
        possible_endings.extend(([] if not len(t_indices) else [
                                [i, len(sentence_terminator)] for i in t_indices]))
    for contraction in contractions:
        c_indices = list(find_all(paragraph, contraction))
        contraction_locations.extend(
            ([] if not len(c_indices) else [i + len(contraction) for i in c_indices]))
    possible_endings = [
        pe for pe in possible_endings if pe[0] + pe[1] not in contraction_locations]
    if len(paragraph) in [pe[0] + pe[1] for pe in possible_endings]:
        max_end_start = max([pe[0] for pe in possible_endings])
        possible_endings = [
            pe for pe in possible_endings if pe[0] != max_end_start]
    possible_endings = [pe[0] + pe[1] for pe in possible_endings if sum(pe) > len(
        paragraph) or (sum(pe) < len(paragraph) and paragraph[sum(pe)] == ' ')]
    end = (-1 if not len(possible_endings) else max(possible_endings))
    return end


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)


# partition list a into k partitions
def partition_list_from_sentence_charactersum(a, k, result):
    # check degenerate conditions
    if k <= 1:
        return [result]
    if k >= len(a):
        return [[x] for x in result]
    # create a list of indexes to partition between, using the index on the
    # left of the partition to indicate where to partition
    # to start, roughly partition the array into equal groups of len(a)/k (note
    # that the last group may be a different size)
    partition_between = []
    for i in range(k-1):
        partition_between.append(int((i+1)*len(a)/k))
    # the ideal size for all partitions is the total height of the list divided
    # by the number of paritions
    print(a)
    average_height = float(sum(a))/k
    best_score = None
    best_partitions = None
    count = 0
    no_improvements_count = 0
    # loop over possible partitionings
    while True:
        # partition the list
        partitions = []
        result_partitions = []

        index = 0
        for div in partition_between:
            # create partitions based on partition_between
            # print('!!!', index, div)
            partitions.append(a[index:div])
            result_partitions.append(result[index:div])
            index = div
        # append the last partition, which runs from the last partition divider
        # to the end of the list
        partitions.append(a[index:])
        result_partitions.append(result[index:])
        # evaluate the partitioning
        worst_height_diff = 0
        worst_partition_index = -1
        for p in partitions:
            # compare the partition height to the ideal partition height
            height_diff = average_height - sum(p)
            # if it's the worst partition we've seen, update the variables that
            # track that
            if abs(height_diff) > abs(worst_height_diff):
                worst_height_diff = height_diff
                worst_partition_index = partitions.index(p)
        # if the worst partition from this run is still better than anything
        # we saw in previous iterations, update our best-ever variables
        if best_score is None or abs(worst_height_diff) < best_score:
            best_score = abs(worst_height_diff)
            best_partitions = result_partitions
            no_improvements_count = 0
        else:
            no_improvements_count += 1
        # decide if we're done: if all our partition heights are ideal, or if
        # we haven't seen improvement in >5 iterations, or we've tried 100
        # different partitionings
        # the criteria to exit are important for getting a good result with
        # complex data, and changing them is a good way to experiment with getting
        # improved results
        if worst_height_diff == 0 or no_improvements_count > 5 or count > 100:
            return best_partitions
        count += 1
        # adjust the partitioning of the worst partition to move it closer to the
        # ideal size. the overall goal is to take the worst partition and adjust
        # its size to try and make its height closer to the ideal. generally, if
        # the worst partition is too big, we want to shrink the worst partition
        # by moving one of its ends into the smaller of the two neighboring
        # partitions. if the worst partition is too small, we want to grow the
        # partition by expanding the partition towards the larger of the two
        # neighboring partitions
        if worst_partition_index == 0:  # the worst partition is the first one
            if worst_height_diff < 0:
                # partition too big, so make it smaller
                partition_between[0] -= 1
            else:
                # partition too small, so make it bigger
                partition_between[0] += 1
        # the worst partition is the last one
        elif worst_partition_index == len(partitions)-1:
            if worst_height_diff < 0:
                # partition too small, so make it bigger
                partition_between[-1] += 1
            else:
                # partition too big, so make it smaller
                partition_between[-1] -= 1
        else:  # the worst partition is in the middle somewhere
            left_bound = worst_partition_index - 1  # the divider before the partition
            right_bound = worst_partition_index  # the divider after the partition
            if worst_height_diff < 0:  # partition too big, so make it smaller
                # the partition on the left is bigger than the one on the right, so make the one on the right bigger
                if sum(partitions[worst_partition_index-1]) > sum(partitions[worst_partition_index+1]):
                    partition_between[right_bound] -= 1
                else:  # the partition on the left is smaller than the one on the right, so make the one on the left bigger
                    partition_between[left_bound] += 1
            else:  # partition too small, make it bigger
                # the partition on the left is bigger than the one on the right, so make the one on the left smaller
                if sum(partitions[worst_partition_index-1]) > sum(partitions[worst_partition_index+1]):
                    partition_between[left_bound] -= 1
                else:  # the partition on the left is smaller than the one on the right, so make the one on the right smaller
                    partition_between[right_bound] += 1


# partition list a into k partitions
def partition_list(a, k):
    # check degenerate conditions
    if k <= 1:
        return [a]
    if k >= len(a):
        return [[x] for x in a]
    # create a list of indexes to partition between, using the index on the
    # left of the partition to indicate where to partition
    # to start, roughly partition the array into equal groups of len(a)/k (note
    # that the last group may be a different size)
    partition_between = []
    for i in range(k-1):
        partition_between.append((i+1)*len(a)/k)
    # the ideal size for all partitions is the total height of the list divided
    # by the number of paritions
    average_height = float(sum(a))/k
    best_score = None
    best_partitions = None
    count = 0
    no_improvements_count = 0
    # loop over possible partitionings
    while True:
        # partition the list
        partitions = []
        index = 0
        for div in partition_between:
            # create partitions based on partition_between
            partitions.append(a[index:div])
            index = div
        # append the last partition, which runs from the last partition divider
        # to the end of the list
        partitions.append(a[index:])
        # evaluate the partitioning
        worst_height_diff = 0
        worst_partition_index = -1
        for p in partitions:
            # compare the partition height to the ideal partition height
            height_diff = average_height - sum(p)
            # if it's the worst partition we've seen, update the variables that
            # track that
            if abs(height_diff) > abs(worst_height_diff):
                worst_height_diff = height_diff
                worst_partition_index = partitions.index(p)
        # if the worst partition from this run is still better than anything
        # we saw in previous iterations, update our best-ever variables
        if best_score is None or abs(worst_height_diff) < best_score:
            best_score = abs(worst_height_diff)
            best_partitions = partitions
            no_improvements_count = 0
        else:
            no_improvements_count += 1
        # decide if we're done: if all our partition heights are ideal, or if
        # we haven't seen improvement in >5 iterations, or we've tried 100
        # different partitionings
        # the criteria to exit are important for getting a good result with
        # complex data, and changing them is a good way to experiment with getting
        # improved results
        if worst_height_diff == 0 or no_improvements_count > 5 or count > 100:
            return best_partitions
        count += 1
        # adjust the partitioning of the worst partition to move it closer to the
        # ideal size. the overall goal is to take the worst partition and adjust
        # its size to try and make its height closer to the ideal. generally, if
        # the worst partition is too big, we want to shrink the worst partition
        # by moving one of its ends into the smaller of the two neighboring
        # partitions. if the worst partition is too small, we want to grow the
        # partition by expanding the partition towards the larger of the two
        # neighboring partitions
        if worst_partition_index == 0:  # the worst partition is the first one
            if worst_height_diff < 0:
                # partition too big, so make it smaller
                partition_between[0] -= 1
            else:
                # partition too small, so make it bigger
                partition_between[0] += 1
        # the worst partition is the last one
        elif worst_partition_index == len(partitions)-1:
            if worst_height_diff < 0:
                # partition too small, so make it bigger
                partition_between[-1] += 1
            else:
                # partition too big, so make it smaller
                partition_between[-1] -= 1
        else:  # the worst partition is in the middle somewhere
            left_bound = worst_partition_index - 1  # the divider before the partition
            right_bound = worst_partition_index  # the divider after the partition
            if worst_height_diff < 0:  # partition too big, so make it smaller
                # the partition on the left is bigger than the one on the right, so make the one on the right bigger
                if sum(partitions[worst_partition_index-1]) > sum(partitions[worst_partition_index+1]):
                    partition_between[right_bound] -= 1
                else:  # the partition on the left is smaller than the one on the right, so make the one on the left bigger
                    partition_between[left_bound] += 1
            else:  # partition too small, make it bigger
                # the partition on the left is bigger than the one on the right, so make the one on the left smaller
                if sum(partitions[worst_partition_index-1]) > sum(partitions[worst_partition_index+1]):
                    partition_between[left_bound] -= 1
                else:  # the partition on the left is smaller than the one on the right, so make the one on the right smaller
                    partition_between[right_bound] += 1


def print_best_partition(a, k):
    # simple function to partition a list and print info
    print('    Partitioning {0} into {1} partitions'.format(a, k))
    p = partition_list(a, k)
    print('    The best partitioning is {0}\n    With heights {1}\n'.format(
        p, map(sum, p)))


def  split_comment2scene(comment,scene_character_limit):

    stences_list = find_sentences(comment)
    # print('--',stences_list)
    amount = [len(i) for i in stences_list]
    # print('---',amount)
    # scene_no=sum(amount) /250+1
    chunks = int(sum(amount)/scene_character_limit) + 1
    # print('---0', chunks)
    # print('---0', chunks)

    sentence_chunks = partition_list_from_sentence_charactersum(
        amount, chunks, stences_list)
    for scene in sentence_chunks:
        scene_content = ' '.join(scene)
        # the while loop will leave a trailing space,
        scene_content = scene_content.strip()
        # so the trailing whitespace must be dealt with
        # before or after the while loop
        while '  ' in scene_content:
            scene_content = scene_content.replace('  ', ' ')
    return sentence_chunks
def format_line(text,text_len):
    # font = ImageFont.truetype('assets/igiari/Galmuri11.ttf', 15)
    # image_size = font.getsize(text=text)
    # print(image_size)
    # text_len=80
    if(len(text) > text_len):

        wrapper = textwrap.TextWrapper(
            width=text_len, break_long_words=False, replace_whitespace=False)
        text = wrapper.wrap(text=text)
    else:
        text = [text]
    return text