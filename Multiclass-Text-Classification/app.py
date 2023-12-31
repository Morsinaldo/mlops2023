"""
Python file for gradio interface.
"""

import ktrain
import gradio as gr

def predict_class(text):
    """
    Predict the class of the text.
    """

    predictor = ktrain.load_predictor("bert_trained")
    pred = predictor.predict(text)
    return pred


iface = gr.Interface(
    fn=predict_class,
    inputs=gr.Textbox(),
    outputs="label",
    live=True
)

iface.launch()
