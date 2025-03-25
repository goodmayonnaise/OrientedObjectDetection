import gradio as gr
import cv2 
import tempfile
import matplotlib.pyplot as plt
import glob
import mmcv
import mmrotate  # noqa: F401
from datetime import timedelta
from time import *
from PIL import Image
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot, init_detector


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

def process_img(model_selector):
    
    image_list = glob.glob('./images/*.png')
    option_dic = load_setting(model_selector)
    detection_model = init_detector(option_dic['CONFIG'], option_dic['model_weight'],device='cpu')
    view = []
    
    
    for image in image_list :    
    
        tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name 
        result = inference_detector(detection_model, image)    
        show_result_pyplot(detection_model, image, result, score_thr=0.3, palette='dota', out_file=tmp_file)
        
        view.append(Image.open(tmp_file))


    return view  # Gradio에 전달

image_list = glob.glob('./images/*.png')

def create_interface():
    css="footer {visibility: hidden}"

    with gr.Blocks(css=css) as demo:

        gr.Markdown('prototype 1 : 78.1% ,    prototype 2 : 78.7%')
        gr.Markdown('prototype 3 : 79.4% ,    prototype 4 : 79.5%')
        

        model_selector = gr.Radio(
            choices=["Prototype_Model1",
                     "Prototype_Model2",
                     "Prototype_Model3",
                     "Prototype_Model4"
                     ],
            value="Prototype_Model1",
            label="Select OBB Model"
        )
        with gr.Tabs():
            with gr.Tab("Image Processing"):
                with gr.Row():
                    image_input = gr.Gallery(value=image_list, label="Input Images", show_label=False, columns=1, interactive=True,selected_index=0)
                    image_output = gr.Gallery(label="Output Images", show_label=False, columns=1, interactive=False,selected_index=0)
                 
        process_button = gr.Button("Process Image")
        
        process_button.click(
            fn=process_img,
            inputs=[model_selector
                    ],
            outputs=[image_output]
        )
    
    return demo

demo = create_interface()

demo.launch(server_name="10.100.0.21", server_port=7866,show_api=False)
