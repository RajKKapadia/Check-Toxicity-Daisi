# Check Toxicity

This Daisi is a simple application that will return you probabilities of a piece of text to be toxic. This application uses the state-of-the-art NLP technology by Unitary with the help of HuggingFace🤗 and Transformers.

The technology I have used are:
* [Transformers](https://github.com/huggingface/transformers)
* [unitary/toxic-bert](https://huggingface.co/unitary/toxic-bert)

```python
import pydaisi as pyd

check_toxicity = pyd.Daisi('rajkkapadia/Check Toxicity')
text = 'I will kill you'
result = check_toxicity.check_toxicity(text).value

print(result)
```