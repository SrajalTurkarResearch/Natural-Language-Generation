#!/usr/bin/env python3
"""
🌟 INTERACTIVE STORY APP
Gradio Web Interface - 30 Lines
Deploy: python story_app.py
"""

import gradio as gr
from rule_based_nlg import EmotionNLG


def story_app(theme, genre, length):
    nlg = EmotionNLG()
    arc = [0, 3, 7, 10, 2] * (length // 5 + 1)
    return nlg.generate_story(arc[:length], theme)


# LAUNCH WEB APP
iface = gr.Interface(
    fn=story_app,
    inputs=[
        gr.Textbox("spaceship", label="Theme"),
        gr.Dropdown(["horror", "romance", "adventure"], label="Genre"),
        gr.Slider(5, 25, 5, label="Length"),
    ],
    outputs=gr.Textbox(label="Generated Story"),
    title="✨ AI Story Generator",
    description="Build tension arcs with emotions!",
)

if __name__ == "__main__":
    iface.launch(share=True)
