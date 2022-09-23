from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import streamlit as st

toxic_bert_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
toxic_bert_model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")

def check_toxicity(text: str) -> dict:
    ''' Check the text for toxicity

        Parameters:
        - text: str

        Returns:
        - dict

        Example return dict:
        - {
            "status": 0/1,
            "message": Successful/Unsuccessful,
            "result": {
                "toxic":"0.92"
                "severe_toxic":"0.04"
                "obscene":"0.59"
                "threat":"0.01"
                "insult":"0.19"
                "identity_hate":"0.28"
            }
        }

        "result" can be an empty dictionary in case of "status" 0
    '''
    try:
        inputs = toxic_bert_tokenizer(text, return_tensors='pt')
        outputs = toxic_bert_model(**inputs)
        sigmoid = nn.Sigmoid()
        probabilitis = sigmoid(outputs.logits)
        probabilitis = probabilitis.detach().numpy()
        id2label = toxic_bert_model.config.id2label
        index = 0
        result = {}
        result['status'] = 1
        result['message'] = 'Successful'
        result['result'] = {}
        for _, value in id2label.items():
            result['result'][value] = round(probabilitis[0][index], 2)
            index += 1
    except:
        result = {}
        result['status'] = 0
        result['message'] = 'Unsuccessful'
        result['result'] = {}

    return result


def st_ui():
    ''' Function to render the Streamlit UI.
    '''
    st.title('Check toxicity...')
    text = st.text_input('Paste you text here...', value='')
    button = st.button('Check toxicity...')
    if button:
        result = check_toxicity(text=text)
        st.write('This is the result.')
        st.json(result)

if __name__ == '__main__':
    st_ui()
