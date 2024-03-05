# Project Title: Talking car

A speaking assistant designed for in-car use, leveraging the LLaMA 2 model to facilitate vocal interactions between the car and its users. This notebook provides the foundation for a speech-enabled interface that can understand spoken questions and respond verbally, enhancing the driving experience with intelligent assistance.

## Description

This project integrates speech-to-text and text-to-speech functionalities into a car's infotainment system, using the LLaMA 2 model to process and respond to vocal queries from users. It employs Gradio for user interface creation, NexusRaven for function calling, and integrates various APIs to fetch real-time information, making it a comprehensive solution for creating a responsive and interactive car assistant.

## Features

•	Speech-to-Text and Text-to-Speech: Enables the car assistant to listen to spoken questions and respond audibly, providing a hands-free experience for drivers and passengers.  
•	Intelligent Function Calling with NexusRaven: Implements a sophisticated system for executing commands and retrieving information based on user queries, using the LLaMA 2 model's capabilities.  
•	Dynamic Model Integration: Incorporates multiple models for language recognition, speech processing, and text generation.  
•	User-Friendly Gradio Interface: easy-to-use interface for testing and deploying the speaking assistant within the car's infotainment system.  
•	Real-Time Information Retrieval: Capable of integrating with various APIs to provide up-to-date information on weather, routes, points of interest, and more.

## Requirements

•	Gradio for creating interactive interfaces  
•	Hugging Face Transformers and additional ML models for speech and language processing  
•	NexusRaven for complex function execution  
All required libraries and packages are directly loaded inside the notebook.

## Installation

To set up the speaking assistant in your car's system, follow these steps:  
1.	Run all the cells until the “Interfaces (text and audio)” section.  
2.	Choose between the interfaces which one to run: audio-to-audio or text-to-text.  
### Usage
1.	Model Setup: Begin by loading the necessary models for speech recognition, language processing, and text-to-speech conversion as detailed in the "Models loads" section.  
2.	Function Definition: Customize the assistant's responses and capabilities by defining functions in the "Function calling with NexusRaven" section.  
3.	Interface Configuration: Choose the Gradio interface that suits your in-car system, following setup instructions in the "Interfaces (text and audio)" section.  
4.	Activation: Execute one of the interface to start the speaking assistant, enabling vocal interactions within the car.  

## Authors and acknowledgment

Sasan Jafarnejad  
Abigail Berthe--Pardo  
