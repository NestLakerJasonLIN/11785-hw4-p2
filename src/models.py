import torch
import torch.nn as nn
import torch.nn.utils as utils
import random

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Attention(nn.Module):
    '''
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    '''
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, lens):
        '''
        :param query :(N, context_size) Query is the output of LSTMCell from Decoder
        :param key: (N, key_size) Key Projection from Encoder per time step
        :param value: (N, value_size) Value Projection from Encoder per time step
        :return output: Attended Context
        :return attention_mask: Attention mask that can be plotted  
        '''


class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    The length of utterance (speech input) can be hundereds to thousands of frames long.
    The Paper reports that a direct LSTM implementation as Encoder resulted in slow convergence,
    and inferior results even after extensive training.
    The major reason is inability of AttendAndSpell operation to extract relevant information
    from a large number of input steps.
    '''
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)

    def forward(self, x):
        '''
        :param x :(N, T) input to the pBLSTM
        :return output: (N, T, H) encoded sequence from pyramidal Bi-LSTM 
        '''


class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key and value.
    Key and value are nothing but simple projections of the output from pBLSTM network.
    '''
    def __init__(self, input_dim, hidden_dim, value_size=128,key_size=128):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)
        
        ### Add code to define the blocks of pBLSTMs! ###

        self.key_network = nn.Linear(hidden_dim*2, value_size)
        self.value_network = nn.Linear(hidden_dim*2, key_size)

    def forward(self, x, lens):
        rnn_inp = utils.rnn.pack_padded_sequence(x, lengths=lens, batch_first=False, enforce_sorted=False)
        outputs, _ = self.lstm(rnn_inp)

        ### Use the outputs and pass it through the pBLSTM blocks! ###

        linear_input, _ = utils.rnn.pad_packed_sequence(outputs)
        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)

        return keys, value


class Decoder(nn.Module):
    '''
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step, 
    thus we use LSTMCell instead of LSLTM here.
    The output from the second LSTMCell can be used as query here for attention module.
    In place of value that we get from the attention, this can be replace by context we get from the attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    '''
    def __init__(self, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=False, isLM=False):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=hidden_dim + value_size, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size)

        self.isAttended = isAttended
        self.isLM = isLM
        self.vocab_size, self.hidden_dim, self.value_size, self.key_size = \
            vocab_size, hidden_dim, value_size, key_size
        if (isAttended == True):
            self.attention = Attention()

        self.character_prob = nn.Linear(key_size + value_size, vocab_size)

        self.init_weights()

    def forward(self, key, values, text=None, isTrain=True, batch_size=None, gumbel_noise=True):
        '''
        :param key :(T, N, key_size) Output of the Encoder Key projection layer
        :param values: (T, N, value_size) Output of the Encoder Value projection layer
        :param text: (N, text_len) Batch input of text with text_length
        :param isTrain: Train or eval mode
        :return predictions: Returns the character perdiction probability 
        '''
        if (isTrain == True):
            max_len =  text.shape[1]
            embeddings = self.embedding(text)
            batch_size = text.shape[0]
        else:
            max_len = 250

        predictions = []
        hidden_states = [None, None]
        prediction = torch.zeros(batch_size,1).to(DEVICE)

        for i in range(max_len):
            # * Implement Gumble noise and teacher forcing techniques 
            # * When attention is True, replace values[i,:,:] with the context you get from attention.
            # * If you haven't implemented attention yet, then you may want to check the index and break 
            #   out of the loop so you do you do not get index out of range errors. 

            # add gumbel noise in generation mode. TODO: is this really correct?
            if (isTrain):
                # Teacher forcing
                teacher_forcing_prob = 0.1
                if (random.random() <= teacher_forcing_prob):
                    if gumbel_noise:
                        prediction = get_gumbel_prediction(prediction)
                    char_embed = self.embedding(prediction.argmax(dim=-1))
                else:
                    char_embed = embeddings[:,i,:]
            else:
                if gumbel_noise:
                    prediction = get_gumbel_prediction(prediction)
                char_embed = self.embedding(prediction.argmax(dim=-1))
            # char_embed.shape: [batch_size, hidden_dim]

            if (self.isAttended):
                # attention-based encoder-decoder
                raise NotImplementedError
            elif (not self.isLM):
                # no attention encoder-decoder
                context = values[i,:,:]
            else:
                # pure decoder language model
                context = torch.zeros(batch_size, self.value_size).to(DEVICE) # TODO: random

            # context.shape: [batch_size, value_size]

            inp = torch.cat([char_embed, context], dim=1)
            # inp.shape: [batch_size, hidden_dim+value_size]

            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            ### Compute attention from the output of the second LSTM Cell ###
            output = hidden_states[1][0]
            # output.shape: [batch_size, key_size]

            prediction = self.character_prob(torch.cat([output, context], dim=1))
            # prediction.shape: [batch_size, vocab_size]

            predictions.append(prediction.unsqueeze(1))

        return torch.cat(predictions, dim=1)

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.lstm1.weight_hh.data.uniform_(-initrange, initrange)
        self.lstm1.weight_ih.data.uniform_(-initrange, initrange)
        self.lstm2.weight_hh.data.uniform_(-initrange, initrange)
        self.lstm2.weight_ih.data.uniform_(-initrange, initrange)


class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''
    def __init__(self, input_dim, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=False):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, hidden_dim)

    def forward(self, speech_input, speech_len, text_input=None, isTrain=True):
        key, value = self.encoder(speech_input, speech_len)
        if (isTrain == True):
            predictions = self.decoder(key, value, text_input)
        else:
            predictions = self.decoder(key, value, text=None, isTrain=False)
        return predictions

def get_gumbel_prediction(prediction):
    U = torch.rand(prediction.shape[1]).to(DEVICE)
    G = -torch.log(-torch.log(U))
    prediction = torch.log(torch.nn.functional.softmax(prediction, dim=-1)) + G.repeat(prediction.shape[0], 1)
    return prediction