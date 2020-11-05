import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertModel


class Sent2Vec:
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = TFBertModel.from_pretrained(model_name)

    def _get_hidden_state(self, sentence: str):
        input_ids = self.tokenizer.encode(sentence)

        # to use the below one for better control over the embeddings
        # input_ids = self.tokenizer.encode_plus(sentence,
        #                                        add_special_tokens=True,
        #                                        max_length=15,
        #                                        pad_to_max_length=True,
        #                                        return_attention_mask=True
        #                                        )

        # keeping the batch size 1
        input_ids = tf.constant(input_ids)[None, :]

        outputs = self.model(input_ids)
        # last hidden state is the first element of the output tuple
        last_hidden_states = outputs[0]

        return last_hidden_states

    def get_sent_embedding(self, sentence: str, use_mean: bool = True):
        sent_embed = self._get_hidden_state(sentence)
        if use_mean:
            sent_embed = np.mean(sent_embed, axis=1)
        return sent_embed


if __name__ == '__main__':
    sent_2_vec = Sent2Vec()
    embeddings = sent_2_vec.get_sent_embedding('hi')
    print(embeddings)
    print(embeddings.shape)
    pass
