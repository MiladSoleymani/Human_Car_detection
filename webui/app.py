import sys
import os

sys.path.append(os.getcwd())

import gradio as gr
import time
from utils.labeling_process import label_data


def run_model(
    data_path, save_path, name_of_model, progress=gr.Progress(track_tqdm=True)
):
    # Replace this with your model code
    # Run your model with the provided inputs
    # You can use the data_path, save_path, and name_of_model variables here

    conf = {
        "data_path": data_path,
        "save_path": save_path,
        "yolo_object": name_of_model,
    }

    label_data(conf, progress)

    # Return the output/result of your model
    return "Model execution completed!"


def app():
    data_path_input = gr.Textbox(label="Data Path")
    save_path_input = gr.Textbox(label="Save Path")
    model_name_input = gr.Textbox(label="Name of Model")
    # progress_bar = gr.outputs.ProgressBar(label="Progress")
    output_text = gr.Textbox()

    iface = gr.Interface(
        fn=run_model,
        inputs=[data_path_input, save_path_input, model_name_input],
        outputs=output_text,
        title="Single Page App with Gradio",
        theme="default",
    )
    return iface


app().queue().launch(share=True, debug=True)
