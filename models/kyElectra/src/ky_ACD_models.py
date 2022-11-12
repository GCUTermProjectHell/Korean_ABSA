import torch
from torch import nn
from utils import MODEL_CLASSES, MODEL_PATH_MAP 


model_name = 'kykim/electra-kor-base' 
config_class, model_class, _ = MODEL_CLASSES[model_name] #config_class, model_class, model_tokenizer
model_path = MODEL_PATH_MAP[model_name]
modelClass_config = config_class.from_pretrained(model_path)
label_id_to_name = ['True', 'False']

class model_ABSA(nn.Module):
    def __init__(self,args, num_label,len_tokenizer):
        super(model_ABSA, self).__init__()
        self.num_label = num_label
        self.model_PLM = model_class.from_pretrained(args.base_model)
        self.model_PLM.resize_token_embeddings(len_tokenizer)

        self.labels_classifier = SimpleClassifier(args,self.num_label)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model_PLM(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None
        )

        sequence_output = outputs[0]
        logits = self.labels_classifier(sequence_output)

        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_label),
                                                labels.view(-1))

        return loss, logits
    
class SimpleClassifier(nn.Module):
    
    def __init__(self,args,num_label):
        super().__init__()
        self.dense = nn.Linear(args.classifier_hidden_size, args.classifier_hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(args.classifier_hidden_size, num_label)

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
        return x



