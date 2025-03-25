import gradio as gr
import cv2 
import tempfile
import matplotlib.pyplot as plt
import glob
import mmcv
import mmrotate  # noqa: F401
from datetime import datetime
from time import *
from PIL import Image
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot, init_detector
import pandas as pd

image_list = glob.glob('./images/*.png')

data = {
    "Prototype": ["Prototype 1", "Prototype 2", "Prototype 3", "Prototype 4"],
    "mAP (%)": [78.1, 78.7, 79.4, 79.5]
}
df = pd.DataFrame(data)

def load_setting(select_value):
    
    option_dic = {}
    
    if select_value == 'Prototype_Model1' :
        option_dic['model_weight'] = './prototype/prototype1.pth'
        option_dic['CONFIG'] = './prototype/prototype1.py'

    elif select_value == 'Prototype_Model2' :
        option_dic['model_weight'] = './prototype/prototype2.pth'
        option_dic['CONFIG'] = './prototype/prototype1.py'

    elif select_value == 'Prototype_Model3' :
        option_dic['model_weight'] = './prototype/prototype3.pth'
        option_dic['CONFIG'] = './prototype/prototype3.py'

    elif select_value == 'Prototype_Model4' :
        option_dic['model_weight'] = './prototype/prototype4.pth'
        option_dic['CONFIG'] = './prototype/prototype4.py'


    return option_dic

def count_bboxes(detection_model,result, score_thr=0.3):
    num_classes = len(result)  
    class_counts = {}
    text = ''
    
    for i in range(num_classes):
        bboxes = result[i]  
        if len(bboxes) > 0:
            valid_bboxes = bboxes[bboxes[:, -1] >= score_thr]
            if len(valid_bboxes) > 0:
                class_counts[detection_model.CLASSES[i]] = len(valid_bboxes)
                text+= f'{detection_model.CLASSES[i]} : {len(valid_bboxes)}\n'
    
    return class_counts , text

def process_img(model_selector,image):
    score_thr = 0.3
    option_dic = load_setting(model_selector)
    detection_model = init_detector(option_dic['CONFIG'], option_dic['model_weight'],device='cpu')
    
    tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name 
    result = inference_detector(detection_model, image)    
    show_result_pyplot(detection_model, image, result, score_thr=score_thr, palette='dota', out_file=tmp_file)
    class_counts , text = count_bboxes(detection_model,result, score_thr=score_thr)
    print(f"{datetime.now().strftime('%Y.%m.%d - %H:%M:%S')} {model_selector}")

    return [Image.open(tmp_file)  ,text]

def interface():
    css="footer {visibility: hidden}"

    with gr.Blocks(css=css) as demo:

        gr.Markdown('### AETEM Model Test' )
        # gr.Markdown('')
        model_selector = gr.Radio(
            choices=["Prototype_Model1",
                     "Prototype_Model2",
                     "Prototype_Model3",
                     "Prototype_Model4"
                     ],
            value="Prototype_Model1",
            label="Select Model"
        )
        

        with gr.Row():
            image_input = gr.Image(type="filepath", label="Inference image")
            image_output = gr.Image(type="filepath", label="Output image")
        
        with gr.Row():
            gr.Examples(image_list, inputs=[image_input])
            output_text = gr.Textbox(label="Output cls")
        
        process_button = gr.Button("Process Image")
        
        process_button.click(
            fn=process_img,
            inputs=[model_selector,image_input
                    ],
            outputs=[image_output,output_text]
        )
        gr.DataFrame(df.sort_values(by=['mAP (%)'],ascending=False), label="Prototype mAP", interactive=False)
    return demo

demo = interface()

demo.launch(server_name="10.100.0.21", server_port=7864,show_api=False)
