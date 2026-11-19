"""Entry point for the Hugging Face Space.

Serves the Flask web app at / and a Gradio demo (with API) at /gradio,
both on port 7860. Detection runs on CPU; YOLOv8n is fast enough there.
"""
import os

# Spaces enables Gradio server-side rendering, which starts a Node server on
# port 7860 before uvicorn can bind it. The Flask UI doesn't need SSR.
os.environ['GRADIO_SSR_MODE'] = 'false'

import spaces  # Must be imported before torch on ZeroGPU hardware.

import cv2
import gradio as gr
import uvicorn
from a2wsgi import WSGIMiddleware
from fastapi import FastAPI

from app import app as flask_app, detect


@spaces.GPU(duration=10)
def gpu_check():
    """ZeroGPU Spaces need at least one GPU function at startup. Not used for detection."""
    return None


def gradio_detect(image_rgb):
    if image_rgb is None:
        return None, []
    items, annotated = detect(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), items


demo = gr.Interface(
    fn=gradio_detect,
    inputs=gr.Image(type='numpy', label='Food photo'),
    outputs=[gr.Image(label='Detections'), gr.JSON(label='Foods and nutrition per 100 g')],
    title='Indonesian food scanner',
    flagging_mode='never',
)

server = FastAPI()
server = gr.mount_gradio_app(server, demo, path='/gradio', ssr_mode=False)
server.mount('/', WSGIMiddleware(flask_app))

if __name__ == '__main__':
    uvicorn.run(server, host='0.0.0.0', port=int(os.environ.get('GRADIO_SERVER_PORT', 7860)))
