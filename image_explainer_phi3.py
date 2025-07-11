# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import argparse
import os
# import readline
# import glob

import onnxruntime_genai as og
import streamlit as st

# streamlit run app_image_to_text_phi3-vision-streamlit.py
# You are a model who is expert in describing architecture diagram. Explain the architecture diagram image that is given.
# Do not make assumptions but describe each component and try to build relationship based on what is in image.
# Also, if architecture diagram is NOT BIAN complained then describe the reason for that.


# You are a model who is expert in describing architecture diagram or image.  Explain and verify is there any challenges in project planning and roadmap
# if you see any minor gap or risk please highlight and explain. also provide remediation steps to mitigate the risk and challenges.


# explain the database table relationship explain how tables are related and generate python SQL code example referring this tables

# https://onnxruntime.ai/docs/genai/tutorials/phi3-v.html
#python app_image_to_text_phi3-vision.py -m Phi-3-vision-128k-instruct-onnx-cpu/Phi-3-vision-128k-instruct


# prompt ==> describe architecture diagram. do not make assumptions but describe each components and try to build relationship based on what is in image
# read this handwritten image and display in to plain english text message. make sure read all references in image and display all

# def _complete(text, state):
#     return (glob.glob(text+'*')+[None])[state]

def run():
    st.title("Business Architecture Validator following BIAN standards")
    image_path = st.text_area("Enter image name for validation (e.g., architecture-diagram.png)")
    text = st.text_area("Enter prompt information for validation (e.g., describe architecture diagram. do not make assumptions but describe each components and try to build relationship based on what is in image)")
    #st.text(anonymized_content)
    #image_path = input("Image Path (leave empty if no image): ")
    if st.button("Submit"):
        print("Loading model...")
        model = og.Model("Phi-3-vision-128k-instruct-onnx-cpu/Phi-3-vision-128k-instruct")
        processor = model.create_multimodal_processor()
        tokenizer_stream = processor.create_stream()
        image = None
        prompt = "<|user|>\n"
        if (len(image_path) == 0):
            print("No image provided")
        else:
            print("Loading image...")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = og.Images.open(image_path)
            prompt += "<|image_1|>\n"

        #text = input("Prompt: ")
        prompt += f"{text}<|end|>\n<|assistant|>\n"
        print("Processing image and prompt...")
        inputs = processor(prompt, images=image)

        print("Generating response...")
        params = og.GeneratorParams(model)
        params.set_inputs(inputs)
        params.set_search_options(max_length=3072)

        generator = og.Generator(model, params)

        response = ""  # Initialize response as an empty string

        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            decoded_token = tokenizer_stream.decode(new_token)
            response += decoded_token  # Now response is defined before it's used
            print(decoded_token, end='', flush=True)

        st.text(response)

        for _ in range(3):
            print()

        # Delete the generator to free the captured graph before creating another one
        del generator

if __name__ == "__main__":
    run()
    # Streamlit UI


    # parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the model")
    # #parser.add_argument("-m", "/Phi-3-vision-128k-instruct-onnx-cpu/Phi-3-vision-128k-instruct", type=str, required=True, help="Path to the model")
    # args = parser.parse_args()


   # https://onnxruntime.ai/docs/genai/tutorials/phi3-v.html
   #python app_image_to_text_phi3-vision.py -m Phi-3-vision-128k-instruct-onnx-cpu/Phi-3-vision-128k-instruct

   # test 1 - https://huggingface.co/spaces/ysharma/Microsoft_Phi-3-Vision-128k
   # test 2 - https://huggingface.co/spaces/gokaygokay/Florence-2
            # https://github.com/kijai/ComfyUI-Florence2/blob/main/nodes.py