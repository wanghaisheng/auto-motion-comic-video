import fasttext
#https://towardsdatascience.com/benchmarking-language-detection-for-nlp-8250ea8b67c
#https://github.com/ageitgey/fastText-windows-binaries/releases/tag/v0.9.1

path_to_pretrained_model = 'assets/lid.176.bin'
def prediction(text):
    text = text.replace('\n', '')
    text = text.replace('\\r', '')

    fmodel = fasttext.load_model(path_to_pretrained_model)
    return fmodel.predict([text])[0][0][0].split('__')[-1]  # ([['__label__en']], [array([0.9331119], dtype=float32)]
    
text ="never forget about china"
print(prediction(text))