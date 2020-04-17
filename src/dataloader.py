import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

'''
Transforms alphabetical input to numerical input, replace each letter by its corresponding 
index from letter_list
<sos> and <eos> will be added
'''
def transform_letter_to_index(transcript, letter2index):
    '''
    :param transcript :(N, ) Transcripts are the text input
    :param letter2index: mapping
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    '''
    indices_list = []

    for label in transcript:
        text = " ".join([letter.decode('utf-8') for letter in label])
        indices_list.append([letter2index['<sos>']] +
                        text2index(text, letter2index) +
                        [letter2index['<eos>']])

    return indices_list

# will remove <eos>, <pad> if any
def transform_index_to_letter(indices, index2letter, stop_indices):
    clipped_indices = []
    for row in indices:
        clipped_row = []
        for idx in row:
            if idx in stop_indices:
                break
            clipped_row.append(idx)
        clipped_indices.append(clipped_row)

    texts_list = index2text(clipped_indices, index2letter, is_batch=True)
    return texts_list

def text2index(text, letter2index, is_batch=False):
    if is_batch:
        return [index2text(row, letter2index) for row in text]
    else:
        return [letter2index[letter] for letter in text]

# input is numpy.ndarray
def index2text(index, index2letter, is_batch=False):
    if is_batch:
        return [index2text(row, index2letter) for row in index]
    else:
        return "".join(index2letter[i] for i in index)

'''
Optional, create dictionaries for letter2index and index2letter transformations
'''
def create_dictionaries(letter_list):
    letter2index = dict()
    index2letter = dict()

    for idx, letter in enumerate(letter_list):
        letter2index[letter] = idx
        index2letter[idx] = letter

    return letter2index, index2letter


class Speech2TextDataset(Dataset):
    '''
    Dataset class for the speech to text data, this may need some tweaking in the
    getitem method as your implementation in the collate function may be different from
    ours. 
    '''
    def __init__(self, speech, text=None, isTrain=True):
        self.speech = speech
        self.isTrain = isTrain
        if (text is not None):
            self.text = text

    def __len__(self):
        return self.speech.shape[0]

    def __getitem__(self, index):
        if (self.isTrain == True):
            return torch.tensor(self.speech[index].astype(np.float32)), torch.tensor(self.text[index])
        else:
            return torch.tensor(self.speech[index].astype(np.float32))

# customize pinned memory for fast host-gpu copies
class CustomBatch:
    def __init__(self, batch_data, is_test=False):
        # reference: https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
        # outputs and targets are shifted by 1
        # TODO: sort seq in a batch
        if not is_test:
            speeches, texts = zip(*batch_data)
            outputs_text = [text[:-1] for text in texts]
            targets_text = [text[1:] for text in texts]

            self.inputs_lens = [len(x) for x in speeches]
            self.outputs_lens = [len(y) for y in outputs_text]
            self.targets_lens = [len(y) for y in targets_text]

            self.inputs = pad_sequence(speeches, batch_first=True)
            self.outputs = pad_sequence(outputs_text, batch_first=True)
            self.targets = pad_sequence(targets_text, batch_first=True)
        else:
            speeches = batch_data
            self.inputs_lens = [len(x) for x in speeches]
            self.inputs = pad_sequence(speeches, batch_first=True)
            self.outputs = None
            self.targets = None

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inputs = self.inputs.pin_memory()
        if self.outputs is not None:
            self.outputs = self.outputs.pin_memory()
        if self.targets is not None:
            self.targets = self.targets.pin_memory()
        return self

def collate(batch_data):
    ### Return the padded speech and text data, and the length of utterance and transcript ###
    return CustomBatch(batch_data)


def collate_test(batch_data):
    ### Return padded speech and length of utterance ###
    return CustomBatch(batch_data, is_test=True)