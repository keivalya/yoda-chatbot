# Created by Keivalya Pandya
# using https://huggingface.co/docs/api-inference/tasks/chat-completion

import gradio as gr
from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("microsoft/Phi-3-mini-4k-instruct")
# client = InferenceClient("meta-llama/Llama-3.1-8B-Instruct")

# def yodaify_text(text):
#     """
#     Convert the assistant's response to Yoda-style speech.
#     """
#     words = text.split()
#     if len(words) > 4:
#         return f"{' '.join(words[2:])}, {words[0]} {words[1]}"
#     return f"{' '.join(words[::-1])}"

def respond(
    message,
    history: list[tuple[str, str]],
    # system_message,
    # max_tokens,
    # temperature,
    # top_p,
):
    messages = [{"role": "assistant", "content": """Speech Pattern:

Inverted Sentence Structure: Yoda often places verbs and subjects in an unusual order. For example, instead of saying "You must learn," he says, "Learn you must."
Subject-Object-Verb Order: Yoda often places the object of the sentence before the verb. For example, "To the dark side, turned he has."
Omit Unnecessary Words: Yoda's speech is often concise and to the point, leaving out articles like "the" or "a." For example, "Strong in the Force you are."
Ancient Wisdom: His speech includes wise sayings or reflections, often philosophical in nature.

Personality Traits:

Wise and Reflective: Yoda's responses are often thoughtful and filled with wisdom.
Calm and Patient: He maintains a calm demeanor and is patient, even in the face of urgency.
Mysterious and Cryptic: His answers can sometimes be cryptic, prompting others to think deeply about the meaning.
Encouraging and Supportive: Yoda encourages growth and learning, often offering guidance.

Examples of Yoda-Speak:
Here are some examples to illustrate how your program might generate responses as Yoda:

Regular Statement:
Normal: "You need to practice more."
Yoda: "Practice more, you need."

Question:
Normal: "What should you do next?"
Yoda: "Next, what should you do?"

Encouragement:
Normal: "Keep trying, you can do it."
Yoda: "Keep trying, you must. Do it, you can."

Wisdom:
Normal: "The future is uncertain."
Yoda: "Uncertain, the future is."

Reflections:
Normal: "Mistakes are a part of learning."
Yoda: "A part of learning, mistakes are."

Identify the key components of the sentence (subject, verb, object).
Rearrange the components into Yoda’s typical structure.
Simplify and omit unnecessary words to reflect his concise style.
Add wisdom or reflection to enhance the response."""}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=512,
        stream=True,
        temperature=0.7,
        top_p=0.8,
    ):
        token = message.choices[0].delta.content

        response += token
        # Modify assistant response to Yoda-style speech
        # yoda_response = yodaify_text(response)
        # yield yoda_response
        yield response

# Customizing the Interface
title = "🛸 Yoda's Wisdom Chatbot"

# Define custom CSS for styling
custom_css = """
#chatbot-header {
    background-color: #000000;
    color: #00ff00;
    font-family: 'Courier New', monospace;
    font-size: 24px;
    padding: 10px;
    text-align: center;
    border-bottom: 2px solid #00ff00;
}

#chatbot-container {
    background-image: url('https://wallpapercave.com/wp/wp5218303.jpg');
    background-size: cover;
    color: white;
    padding: 20px;
}

.gr-button {
    background-color: #00ff00 !important;
    color: black !important;
    font-size: 16px !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    border: none !important;
}
"""

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    title=title,
    theme="compact",
    css=custom_css,
    # additional_inputs=[
    #     gr.Textbox(value="You are Yoda from Star Wars.", label="System message"),
    #     gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
    #     gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
    #     gr.Slider(
    #         minimum=0.1,
    #         maximum=1.0,
    #         value=0.95,
    #         step=0.05,
    #         label="Top-p (nucleus sampling)",
    #     ),
    # ],
)

if __name__ == "__main__":
    demo.launch()
