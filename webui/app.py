import gradio as gr


def run_model(data_path, save_path, name_of_model, progress):
    # Replace this with your model code
    # Run your model with the provided inputs
    # You can use the data_path, save_path, and name_of_model variables here

    # Simulating a long-running task
    for _ in range(10):
        # Update the progress bar
        progress.update(progress.value + 10)

    # Return the output/result of your model
    return "Model execution completed!"


def app():
    data_path_input = gr.inputs.Textbox(label="Data Path")
    save_path_input = gr.inputs.Textbox(label="Save Path")
    model_name_input = gr.inputs.Textbox(label="Name of Model")
    progress_bar = gr.outputs.ProgressBar(label="Progress")
    run_button = gr.outputs.Button(label="Run Model")
    output_text = gr.outputs.Textbox()

    iface = gr.Interface(
        fn=run_model,
        inputs=[
            data_path_input,
            save_path_input,
            model_name_input,
            progress_bar,
            run_button,
        ],
        outputs=output_text,
        title="Single Page App with Gradio",
        theme="default",
        layout="vertical",
        verbose=True,
    )
    return iface


if __name__ == "__main__":
    app().launch()
