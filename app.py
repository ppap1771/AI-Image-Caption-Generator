import torch
import gradio as gr 
import re 
from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel

device='cpu'
encoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
decoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
model_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(device)

def predict(image,max_length=64, num_beams=4):
  image = image.convert('RGB')
  image = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
  clean_text = lambda x: x.replace('<|endoftext|>','').split('\n')[0]
  caption_ids = model.generate(image, max_length = max_length)[0]
  caption_text = clean_text(tokenizer.decode(caption_ids))
  return caption_text 

def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])

css = '''
h1#title {
  text-align: center;
}
h3#header {
  text-align: center;
}
img#overview {
  max-width: 800px;
  max-height: 600px;
}
img#style-image {
  max-width: 1000px;
  max-height: 600px;
}
'''

demo = gr.Interface(fn=predict, inputs = "image", outputs = "text")
with demo:
  gr.Markdown('''<h1 id="title">Image Caption 🖼️</h1>''')
  gr.Markdown('''Made by : Shreyas Dixit''')
  with gr.Column():
        input = gr.inputs.Image(label="Upload your Image", type = 'pil', optional=True)
        output = gr.outputs.Textbox(type="auto",label="Captions")
  btn = gr.Button("Genrate Caption")
  btn.click(fn=predict, inputs=input, outputs=output)

demo.launch()