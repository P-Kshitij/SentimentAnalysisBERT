import config
import transformers
import torch.nn as nn

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)
    
    def forward(self, ids, mask, token_type_ids):
        '''
        out1 = sequence of hidden states at the last layer 
                size = (batchsize, seqlen, hidden_size)
        out2 = pooled_output: the hidden state of the first token [CLS], 
                trained for sentence-pair-classification task
                size = (batchsize, hidden_size)
        We only take out2 here pooled_output
        '''
        _, out2 = self.bert(
            ids,
            attention_mask = mask,
            token_type_ids=token_type_ids 
        )
        
        bert_out = self.bert_drop(out2)
        output  = self.out(bert_out)
        return output